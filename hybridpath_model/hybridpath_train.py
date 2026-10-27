import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
hybridpath_train.py
-------------------
Optimized training script for HybridPath with CUDA AMP (FP16).
Supports resuming from existing checkpoint and ascii-safe logging.

Usage:
    python hybridpath_train.py --train split_train.csv --val split_val.csv
"""

import argparse
import json
import time
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from hybridpath_model import HybridPath, FocalLoss
from hybridpath_dataset import OSCCDataset

CLASS_NAMES = ["cohesive", "dispersive", "no_feature", "noncohesive"]


def build_class_weights(df_train, device):
    counts  = df_train["label"].value_counts()
    counts  = counts.reindex(CLASS_NAMES).fillna(1).values.astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for crops, img_256, labels in loader:
            crops   = crops.to(device)
            img_256 = img_256.to(device)
            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, img_256)
            else:
                logits = model(crops, img_256)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,
                         labels=list(range(4)), zero_division=0)
    return macro_f1, per_class, all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",      default="split_train.csv")
    ap.add_argument("--val",        default="split_val.csv")
    ap.add_argument("--save-dir",   default="weights")
    ap.add_argument("--epochs",     type=int,   default=15)
    ap.add_argument("--patience",   type=int,   default=5)
    ap.add_argument("--batch-size", type=int,   default=8)
    ap.add_argument("--workers",    type=int,   default=0)
    ap.add_argument("--unfreeze",   type=int,   default=3,
                    help="Number of Phikon blocks to unfreeze (from the end)")
    ap.add_argument("--device",     default=None)
    ap.add_argument("--resume",     action="store_true", default=True,
                    help="Resume from existing hybridpath_best.pt if present")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    print(f"Device: {device}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # -- Data ----------------------------------------------------------------
    df_train = pd.read_csv(args.train)
    df_val   = pd.read_csv(args.val)
    print(f"\nTrain original tiles: {len(df_train)}", flush=True)
    print(f"Val   tiles         : {len(df_val)}", flush=True)
    print(f"\nTrain class distribution (before augmentation):", flush=True)
    print(df_train["label"].value_counts().to_string(), flush=True)

    train_ds = OSCCDataset(df_train, mode="train")
    val_ds   = OSCCDataset(df_val,   mode="val")

    print(f"\nTrain tiles after augmentation multiplier: {len(train_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    # -- Model ---------------------------------------------------------------
    model = HybridPath(
        num_classes=4,
        unfreeze_last_n=args.unfreeze,
        device=device,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params    : {total:,}", flush=True)
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)", flush=True)

    # -- Loss & AMP ----------------------------------------------------------
    class_weights = build_class_weights(df_train, device)
    print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}", flush=True)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- Optimizer (differential LR) -----------------------------------------
    phikon_b_params = [
        p for p in model.phikon_b.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW([
        {"params": phikon_b_params,              "lr": 1e-5},
        {"params": model.cnn.parameters(),       "lr": 1e-4},
        {"params": model.proj_a.parameters(),    "lr": 3e-4},
        {"params": model.proj_b.parameters(),    "lr": 3e-4},
        {"params": model.gate.parameters(),      "lr": 3e-4},
        {"params": model.head.parameters(),      "lr": 3e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-6
    )

    # -- Check Resume --------------------------------------------------------
    best_f1 = 0.0
    start_epoch = 1
    log_rows = []
    ckpt_path = save_dir / "hybridpath_best.pt"

    if args.resume and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            best_f1 = ckpt.get("best_f1", 0.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"\n[RESUME] Successfully loaded checkpoint from epoch {ckpt.get('epoch', 0)} (best_f1={best_f1:.4f})", flush=True)
            print(f"Continuing training from epoch {start_epoch}...", flush=True)
        except Exception as e:
            print(f"\n[RESUME] Could not load checkpoint ({e}), starting fresh.", flush=True)

    # Load existing log if resuming
    log_path = save_dir / "training_log.csv"
    if args.resume and log_path.exists():
        try:
            log_rows = pd.read_csv(log_path).to_dict("records")
        except Exception:
            log_rows = []

    wait = 0
    total_train_batches = len(train_loader)

    print(f"\n{'='*70}", flush=True)
    print(f"TRAINING HybridPath (AMP FP16) | {args.epochs} epochs | patience={args.patience}", flush=True)
    print(f"Total Batches per epoch: Train={total_train_batches}, Val={len(val_loader)}", flush=True)
    print(f"{'='*70}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for b_idx, (crops, img_256, labels) in enumerate(train_loader, start=1):
            crops   = crops.to(device)
            img_256 = img_256.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, img_256)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(crops, img_256)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if b_idx % 40 == 0 or b_idx == total_train_batches:
                cur_loss = total_loss / n_batches
                elapsed_b = time.time() - t0
                print(f"  [Ep {epoch:2d}/{args.epochs}] Batch {b_idx:3d}/{total_train_batches} | "
                      f"Running Loss: {cur_loss:.4f} | Elapsed: {elapsed_b:.0f}s", flush=True)

        scheduler.step()

        # -- Validate --
        macro_f1, per_class, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
        avg_loss = total_loss / max(n_batches, 1)
        elapsed  = time.time() - t0

        disp_f1 = per_class[1] if len(per_class) > 1 else 0.0
        nc_f1   = per_class[3] if len(per_class) > 3 else 0.0
        coh_f1  = per_class[0] if len(per_class) > 0 else 0.0
        nf_f1   = per_class[2] if len(per_class) > 2 else 0.0

        print(
            f"\n>>> Epoch {epoch:2d}/{args.epochs} Summary: "
            f"Loss={avg_loss:.4f} | Macro-F1={macro_f1:.4f} | "
            f"Coh={coh_f1:.4f} | Disp={disp_f1:.4f} | NF={nf_f1:.4f} | NC={nc_f1:.4f} | "
            f"Time={elapsed:.0f}s", flush=True
        )

        log_rows.append({
            "epoch": epoch,
            "loss": avg_loss,
            "macro_f1": macro_f1,
            "f1_cohesive": coh_f1,
            "f1_dispersive": disp_f1,
            "f1_no_feature": nf_f1,
            "f1_noncohesive": nc_f1,
        })
        # Save log after every epoch so it's always persistent
        pd.DataFrame(log_rows).to_csv(save_dir / "training_log.csv", index=False)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            wait    = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1":         best_f1,
                "per_class_f1":    per_class.tolist(),
            }, save_dir / "hybridpath_best.pt")
            print(f"  [BEST] New Best Model Saved! (Macro-F1={best_f1:.4f})\n", flush=True)
        else:
            wait += 1
            print(f"  (No improvement for {wait}/{args.patience} epochs)\n", flush=True)
            if wait >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!", flush=True)
                break

    config = {
        "model_name":    "HybridPath",
        "version":       "1.0",
        "num_classes":   4,
        "classes":       CLASS_NAMES,
        "backbone":      "owkin/phikon",
        "frozen_blocks": f"0-{12 - args.unfreeze - 1}",
        "unfrozen_blocks": f"{12 - args.unfreeze}-11",
        "cnn_input":     256,
        "fused_dim":     512,
        "activation":    "GELU",
        "loss":          "FocalLoss(gamma=2.0)",
        "precision":     "AMP FP16",
        "best_val_macro_f1": best_f1,
        "total_params":      total,
        "trainable_params":  trainable,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"TRAINING COMPLETE | Best Val Macro-F1: {best_f1:.4f}", flush=True)
    print(f"Checkpoint saved : {save_dir}/hybridpath_best.pt", flush=True)
    print(f"Training history : {save_dir}/training_log.csv", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
