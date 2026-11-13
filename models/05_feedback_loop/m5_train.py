import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m5_train.py
-----------
Train Model 5: Recurrent Feedback / Looping Dual-Stage Network.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from m5_model import Model5LoopNetwork, MultiTaskLoopLoss
from m5_dataset import OSCCLoopDataset, CLASS_NAMES

LOG_FILE = Path("D:/Project/model5_loop/training.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_class_weights(df, device):
    counts = df["label"].value_counts()
    total  = len(df)
    weights = []
    for cls in CLASS_NAMES:
        cnt = counts.get(cls, 1)
        base = total / (4 * cnt)
        if cls in ["dispersive", "noncohesive"]:
            base *= 1.5
        weights.append(base)
    weights = np.array(weights)
    weights = weights / weights.sum() * 4.0
    return torch.tensor(weights, dtype=torch.float32).to(device)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []
    for crops, img_256, labels in loader:
        crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            final_logits, _ = model(crops, img_256)
        preds = final_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return macro_f1, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",  default="model4_cascade/m4_train.csv")
    ap.add_argument("--val-csv",    default="hybridpath_model/split_val.csv")
    ap.add_argument("--out-dir",    default="model5_loop/weights")
    ap.add_argument("--epochs",     type=int,   default=25)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--unfreeze",   type=int,   default=3)
    ap.add_argument("--patience",   type=int,   default=3)
    ap.add_argument("--workers",    type=int,   default=0)
    ap.add_argument("--resume",     action="store_true")
    args = ap.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Model 5 Training Started at {time.ctime()} ===\n")

    log(f"Device : {device} (AMP={use_amp})")

    train_path = Path(args.train_csv)
    if not train_path.exists():
        train_path = Path("hybridpath_model/split_train.csv")
        log(f"Note: {args.train_csv} not found, falling back to {train_path}")

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(args.val_csv)

    train_ds = OSCCLoopDataset(df_train, mode="train")
    val_ds   = OSCCLoopDataset(df_val,   mode="val")

    log(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device == "cuda"))

    model = Model5LoopNetwork(unfreeze_last_n=args.unfreeze, device=device).to(device)

    class_weights = build_class_weights(df_train, device)
    criterion     = MultiTaskLoopLoss(class_weights=class_weights, gamma=2.0, aux_weight=0.3)
    scaler        = torch.amp.GradScaler("cuda", enabled=use_amp)

    adapt_params = [p for p in model.phikon_adapt.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("phikon_") and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": adapt_params, "lr": 1e-5},
        {"params": other_params, "lr": 2e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-6
    )

    best_f1   = 0.0
    start_ep  = 1
    ckpt_path = save_dir / "model5_best.pt"

    log(f"\nTraining Model 5 (Recurrent Feedback Looping Dual-Stage Network)...")
    log(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Patience: {args.patience}")
    wait = 0

    for epoch in range(start_ep, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_b = 0.0, 0
        total_batches = len(train_loader)

        for b_idx, (crops, img_256, labels) in enumerate(train_loader, start=1):
            crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    final_logits, early_logits = model(crops, img_256)
                    loss = criterion(final_logits, early_logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                final_logits, early_logits = model(crops, img_256)
                loss = criterion(final_logits, early_logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_b += 1

            if b_idx % 40 == 0 or b_idx == total_batches:
                log(f"  [Ep {epoch:2d}] Batch {b_idx:4d}/{total_batches} | Running Loss: {total_loss/n_b:.4f}")

        scheduler.step()
        macro_f1, per_class = evaluate(model, val_loader, device, use_amp)
        elapsed = time.time() - t0
        log(f"\n>>> Ep {epoch:2d} | Loss: {total_loss/n_b:.4f} | Macro-F1: {macro_f1:.4f} | "
            f"Per-class: {per_class.round(3)} | Time: {elapsed:.0f}s")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            wait = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_f1": best_f1,
                "per_class": per_class.tolist()
            }, ckpt_path)
            log(f"  [BEST] Model 5 Best saved! (Macro-F1: {best_f1:.4f})\n")
        else:
            wait += 1
            log(f"  (No improvement for {wait}/{args.patience} epochs)\n")
            if wait >= args.patience:
                log(f"\nEarly stop triggered at epoch {epoch}.")
                break

    log(f"\nModel 5 Training Complete! Best Macro-F1: {best_f1:.4f}\n")


if __name__ == "__main__":
    main()
