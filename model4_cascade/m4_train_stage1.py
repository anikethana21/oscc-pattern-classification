import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_train_stage1.py
------------------
Train Stage 1 of Model 4:
STRICTLY trained only on the first two classes: cohesive (0) and no_feature (1).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from m4_stage1_model import Stage1WithBackbone
from m4_dataset import Stage1Dataset, NAMED_COLS


def compute_named_stats(df):
    avail = [c for c in NAMED_COLS if c in df.columns]
    if not avail:
        return None, None
    vals = df[avail].fillna(0.0).values.astype(np.float32)
    mean = vals.mean(axis=0)
    std  = vals.std(axis=0) + 1e-8
    full_mean = np.zeros(len(NAMED_COLS), dtype=np.float32)
    full_std  = np.ones(len(NAMED_COLS),  dtype=np.float32)
    for i, c in enumerate(NAMED_COLS):
        if c in avail:
            idx = avail.index(c)
            full_mean[i] = mean[idx]
            full_std[i]  = std[idx]
    return full_mean, full_std


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []
    for crops, named, labels in loader:
        crops  = crops.to(device)
        named  = named.to(device)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(crops, named)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    return macro_f1, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",   default="model4_cascade/m4_train.csv")
    ap.add_argument("--val-csv",     default="hybridpath_model/split_val.csv")
    ap.add_argument("--out-dir",     default="model4_cascade/weights_s1")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--batch-size",  type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--patience",    type=int,   default=8)
    ap.add_argument("--workers",     type=int,   default=0)
    ap.add_argument("--resume",      action="store_true")
    args = ap.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}", flush=True)

    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)

    # Filter val set for binary evaluation
    df_val_binary = df_val[df_val["label"].isin(["cohesive", "no_feature"])].copy()

    named_mean, named_std = compute_named_stats(df_train)
    if named_mean is not None:
        np.save(save_dir / "named_mean.npy", named_mean)
        np.save(save_dir / "named_std.npy",  named_std)

    train_ds = Stage1Dataset(df_train, mode="train", named_mean=named_mean, named_std=named_std)
    val_ds   = Stage1Dataset(df_val_binary, mode="val", named_mean=named_mean, named_std=named_std)

    print(f"Train binary tiles (cohesive + no_feature): {len(train_ds)}", flush=True)
    print(f"Val binary tiles: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device == "cuda"))

    model = Stage1WithBackbone(device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    start_ep = 1
    ckpt_path = save_dir / "stage1_best.pt"

    print(f"\nTraining Stage 1 (First Two Classes Only)...", flush=True)
    wait = 0
    for epoch in range(start_ep, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_b = 0.0, 0
        for b_idx, (crops, named, labels) in enumerate(train_loader, start=1):
            crops, named, labels = crops.to(device), named.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, named)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(crops, named)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_b += 1

        macro_f1, per_class = evaluate(model, val_loader, device, use_amp)
        print(f"  [Ep {epoch:2d}/{args.epochs}] Loss: {total_loss/n_b:.4f} | Binary Macro-F1: {macro_f1:.4f} | Time: {time.time()-t0:.0f}s", flush=True)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            wait = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_f1": best_f1}, ckpt_path)
            print(f"    -> Saved Stage 1 Best (F1: {best_f1:.4f})", flush=True)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch}.", flush=True)
                break

    print(f"Stage 1 Training Complete! Best F1: {best_f1:.4f}\n", flush=True)


if __name__ == "__main__":
    main()
