import os, sys, math, json, random, logging, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
train.py  –  Model 6 training script
--------------------------------------
Every hyperparameter comes from config.yaml.
Routing is NOT applied during training; every tile trains every head and the concluder.

Run:
    cd D:\Project\model6_earlyexit
    python train.py [--config config.yaml] [--device cuda]
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

sys.path.insert(0, "D:/Project/model4_cascade")
sys.path.insert(0, "D:/Project/model6_earlyexit")

from m4_dataset import (
    Stage2Dataset,          # reuse augmentation logic from Model 4
    CLASS_NAMES, LABEL_ENC_4,
    get_heavy_aug, get_light_aug, get_val_transform, make_16_crops,
)
from model6 import EarlyExitModel
from routing import route_batch


# ── Dataset reuse ─────────────────────────────────────────────────────────────
# Stage2Dataset already supports all 4 classes + heavy aug for rare classes.
# We reuse it as-is; Model 6 forward takes (crops_224, img_256).

# ── Loss helpers ──────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    """Focal Binary Cross-Entropy for one-vs-rest sigmoid heads."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logit, target.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pt   = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        return loss.mean()


class FocalCE(nn.Module):
    """Focal Cross-Entropy for the concluder."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Main training routine ─────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weights(df: pd.DataFrame, classes: list, device) -> dict:
    """Compute pos_weight = (N - n_pos) / n_pos for each class."""
    N = len(df)
    pw = {}
    for cls in classes:
        n_pos = (df["label"] == cls).sum()
        pw[cls] = torch.tensor([(N - n_pos) / max(n_pos, 1)], dtype=torch.float32, device=device)
    return pw


def build_losses(cfg: dict, train_df: pd.DataFrame, device, classes: list):
    t_cfg = cfg["training"]
    gamma = t_cfg["focal_gamma"]
    focal_heads = set(t_cfg["focal_heads"])
    pw_map = compute_pos_weights(train_df, classes, device) if t_cfg["pos_weight_auto"] else {}

    head_loss_fns = {}
    for cls in classes:
        if cls in focal_heads:
            head_loss_fns[cls] = FocalBCE(gamma=gamma)
        else:
            pw = pw_map.get(cls)
            head_loss_fns[cls] = FocalBCE(gamma=1.0, pos_weight=pw)

    counts = np.array([(train_df["label"] == c).sum() for c in classes], dtype=np.float32)
    ce_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32, device=device)
    ce_weights = ce_weights / ce_weights.sum() * len(classes)
    concluder_loss = FocalCE(gamma=gamma, weight=ce_weights)

    return head_loss_fns, concluder_loss


def compute_total_loss(outputs: dict, labels: torch.Tensor,
                       head_loss_fns: dict, concluder_loss_fn,
                       loss_weights: dict, classes: list) -> dict:
    """
    Compute all 5 loss terms and return a dict with individual + total.
    Head targets are one-vs-rest: 1 if tile label == that class.
    """
    loss_dict = {}
    total = torch.tensor(0.0, device=labels.device)

    head_key_map = {
        "no_feature" : "head_no_feature",
        "dispersive" : "head_dispersive",
        "cohesive"   : "head_cohesive",
        "noncohesive": "head_noncohesive",
    }

    for cls in classes:
        target = (labels == LABEL_ENC_4[cls]).long()
        logit  = outputs[head_key_map[cls]]
        lw     = loss_weights.get(f"head_{cls}", 1.0)
        l      = head_loss_fns[cls](logit, target) * lw
        loss_dict[f"loss_head_{cls}"] = l.item()
        total = total + l

    cl = concluder_loss_fn(outputs["concluder"], labels) * loss_weights.get("concluder", 1.0)
    loss_dict["loss_concluder"] = cl.item()
    total = total + cl
    loss_dict["total"] = total

    return loss_dict


def evaluate_on_loader(model: nn.Module, loader: DataLoader,
                       routing_cfg: dict, device) -> dict:
    model.eval()
    all_true, all_pred = [], []
    use_amp = (torch.device(device).type == "cuda")
    with torch.no_grad():
        for crops_224, img_256, labels in loader:
            crops_224 = crops_224.to(device)
            img_256   = img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(crops_224, img_256)
            decisions = route_batch(outputs, routing_cfg)
            pred_classes = [d["pred_class"] for d in decisions]
            all_true.extend(labels.tolist())
            all_pred.extend([LABEL_ENC_4[p] for p in pred_classes])
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return {"macro_f1": f1, "y_true": all_true, "y_pred": all_pred}


def build_sampler(df: pd.DataFrame, classes: list):
    counts = df["label"].value_counts()
    class_wt = {c: 1.0 / counts.get(c, 1) for c in classes}
    sample_wt = df["label"].map(class_wt).values
    return WeightedRandomSampler(weights=sample_wt, num_samples=len(sample_wt), replacement=True)


def get_cosine_lr(optimizer, epoch: int, total_epochs: int, warmup: int,
                  base_lrs: list):
    if epoch < warmup:
        factor = (epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        factor   = 0.5 * (1 + math.cos(math.pi * progress))
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    seed_everything(cfg["training"]["seed"])
    classes = cfg["classes"]
    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_path = cfg["logging"]["log_file"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger()
    log.info(f"Model 6  |  device={device}  |  config={args.config}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(cfg["paths"]["train_csv"])
    val_df   = pd.read_csv(cfg["paths"]["val_csv"])
    log.info(f"Train: {len(train_df)} tiles  |  Val: {len(val_df)} tiles")
    log.info("Train class dist:\n" + str(train_df["label"].value_counts()))

    train_ds = Stage2Dataset(train_df, mode="train")
    val_ds   = Stage2Dataset(val_df,   mode="val")

    sampler    = build_sampler(train_df, classes)
    dl_cfg     = cfg["dataloader"]
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EarlyExitModel(cfg, device=str(device)).to(device)
    log.info("EarlyExitModel loaded")

    m4_path = cfg["paths"].get("m4_weights_s2")
    if m4_path and os.path.exists(m4_path):
        log.info(f"Loading pretrained Model 4 backbone weights from {m4_path}...")
        model.load_m4_backbone(m4_path)
    else:
        log.warning(f"No valid pretrained backbone path found at {m4_path}; starting with fresh weights.")

    # ── Optimiser: 3 param groups with different LRs ──────────────────────────
    t_cfg   = cfg["training"]
    head_params = (
        list(model.net_stage1.parameters()) +
        list(model.head_no_feature.parameters()) +
        list(model.net_stage2.parameters()) +
        list(model.head_dispersive.parameters()) +
        list(model.net_stage3.parameters()) +
        list(model.head_cohesive.parameters()) +
        list(model.net_stage4.parameters()) +
        list(model.head_noncohesive.parameters()) +
        list(model.net_concluder.parameters()) +
        list(model.concluder.parameters()) +
        list(model.proj_a.parameters()) +
        list(model.proj_b.parameters()) +
        list(model.gate.parameters())
    )
    phikon_b_params = [p for p in model.phikon_b.parameters() if p.requires_grad]
    cnn_params      = list(model.cnn.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": t_cfg["lr_head"]},
        {"params": phikon_b_params, "lr": t_cfg["lr_phikon_b"]},
        {"params": cnn_params,      "lr": t_cfg["lr_cnn"]},
    ], weight_decay=t_cfg["weight_decay"])

    base_lrs = [t_cfg["lr_head"], t_cfg["lr_phikon_b"], t_cfg["lr_cnn"]]

    # ── Losses ────────────────────────────────────────────────────────────────
    head_loss_fns, concluder_loss = build_losses(cfg, train_df, device, classes)
    loss_weights = t_cfg["loss_weights"]

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1    = 0.0
    no_improve = 0
    epochs     = t_cfg["epochs"]
    patience   = t_cfg["patience"]
    print_every = cfg["logging"]["print_every_n_batches"]

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        get_cosine_lr(optimizer, epoch, epochs, t_cfg["warmup_epochs"], base_lrs)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (crops_224, img_256, labels) in enumerate(train_loader):
            crops_224 = crops_224.to(device)
            img_256   = img_256.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(crops_224, img_256)
                loss_d  = compute_total_loss(outputs, labels, head_loss_fns,
                                             concluder_loss, loss_weights, classes)

            scaler.scale(loss_d["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_d["total"].item()
            if (batch_idx + 1) % print_every == 0:
                log.info(
                    f"Ep {epoch+1}/{epochs}  batch {batch_idx+1}/{len(train_loader)}"
                    f"  loss={loss_d['total'].item():.4f}"
                    f"  [nf={loss_d['loss_head_no_feature']:.3f}"
                    f"  dis={loss_d['loss_head_dispersive']:.3f}"
                    f"  coh={loss_d['loss_head_cohesive']:.3f}"
                    f"  nc={loss_d['loss_head_noncohesive']:.3f}"
                    f"  ce={loss_d['loss_concluder']:.3f}]"
                )

        avg_loss = epoch_loss / len(train_loader)
        val_res  = evaluate_on_loader(model, val_loader, cfg["routing"], device)
        val_f1   = val_res["macro_f1"]
        elapsed  = time.time() - t0
        log.info(f"Epoch {epoch+1:3d}  train_loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}  ({elapsed:.0f}s)")

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            ckpt_path  = os.path.join(cfg["paths"]["save_dir"], "best_m6.pt")
            torch.save(model.state_dict(), ckpt_path)
            # Save config snapshot alongside checkpoint
            with open(os.path.join(cfg["paths"]["save_dir"], "config_used.yaml"), "w") as f:
                yaml.dump(cfg, f)
            log.info(f"  *** New best val macro-F1={best_f1:.4f}  saved to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    log.info(f"Training complete. Best val macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
