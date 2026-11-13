import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_train_stage2.py
------------------
Train Stage 2 specialist of Model 4:
Trained on all 4 classes, with emphasis/heavy loss weights on dispersive & noncohesive.
Outputs full 4-class classification.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from m4_stage2_model import Stage2Specialist4Class, FocalLoss
from m4_dataset import Stage2Dataset, CLASS_NAMES


def build_class_weights(df, device):
    # Heavy weights for dispersive (1) and noncohesive (3)
    counts = df["label"].value_counts()
    total  = len(df)
    weights = []
    for cls in CLASS_NAMES:
        cnt = counts.get(cls, 1)
        base = total / (4 * cnt)
        if cls in ["dispersive", "noncohesive"]:
            base *= 1.5  # extra boost for the remaining two classes
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
            logits = model(crops, img_256)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return macro_f1, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",  default="model4_cascade/m4_train.csv")
    ap.add_argument("--val-csv",    default="hybridpath_model/split_val.csv")
    ap.add_argument("--out-dir",    default="model4_cascade/weights_s2")
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--unfreeze",   type=int,   default=3)
    ap.add_argument("--patience",   type=int,   default=8)
    ap.add_argument("--workers",    type=int,   default=0)
    ap.add_argument("--resume",     action="store_true")
    args = ap.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}", flush=True)

    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)

    train_ds = Stage2Dataset(df_train, mode="train")
    val_ds   = Stage2Dataset(df_val,   mode="val")

    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device == "cuda"))

    model = Stage2Specialist4Class(num_classes=4, unfreeze_last_n=args.unfreeze, device=device).to(device)

    class_weights = build_class_weights(df_train, device)
    criterion     = FocalLoss(gamma=2.0, weight=class_weights)
    scaler        = torch.amp.GradScaler("cuda", enabled=use_amp)

    phikon_b_params = [p for p in model.phikon_b.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": phikon_b_params,           "lr": 1e-5},
        {"params": model.cnn.parameters(),    "lr": 1e-4},
        {"params": model.proj_a.parameters(), "lr": 3e-4},
        {"params": model.proj_b.parameters(), "lr": 3e-4},
        {"params": model.gate.parameters(),   "lr": 3e-4},
        {"params": model.head.parameters(),   "lr": 3e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-6
    )

    best_f1   = 0.0
    start_ep  = 1
    ckpt_path = save_dir / "stage2_best.pt"

    print(f"\nTraining Stage 2 Specialist (4-Class with Focus on Dispersive & Noncohesive)...", flush=True)
    wait = 0
    for epoch in range(start_ep, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_b = 0.0, 0
        for b_idx, (crops, img_256, labels) in enumerate(train_loader, start=1):
            crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, img_256)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(crops, img_256)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_b += 1

            if b_idx % 40 == 0 or b_idx == len(train_loader):
                print(f"  [Ep {epoch:2d}] Batch {b_idx:3d}/{len(train_loader)} | Running Loss: {total_loss/n_b:.4f}", flush=True)

        scheduler.step()
        macro_f1, per_class = evaluate(model, val_loader, device, use_amp)
        print(f"\n>>> Ep {epoch:2d} | Loss: {total_loss/n_b:.4f} | Macro-F1: {macro_f1:.4f} | Per-class: {per_class.round(3)} | Time: {time.time()-t0:.0f}s", flush=True)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            wait = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_f1": best_f1}, ckpt_path)
            print(f"  [BEST] Stage 2 Best saved! (Macro-F1: {best_f1:.4f})\n", flush=True)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch}.", flush=True)
                break

    print(f"Stage 2 Training Complete! Best Macro-F1: {best_f1:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
