"""
m4_3_2_train.py
---------------
Training script for Model 4.3.2 (Cleaned dataset from new Durgesh + Amrutha data with outlier quarantine).
- Architecture: Model 4 Specialist 4-Class Network (Phikon Frozen A + Phikon Fine-tuned B + TileCNN C + Dynamic Gating).
- Pretrained Backbone Weights: Initializes from champion Model 4.3 (D:/Project/model4_3_retrained/weights/model4_3_best.pt).
- Early Stopping: Strict patience = 3 (stops early if no validation Macro-F1 improvement for 3 consecutive epochs).
- Precision: FP16 Automatic Mixed Precision (AMP) with GradScaler.
- Loss: Multi-class Focal Loss (gamma=2.0).
- Optimizer: AdamW with Cosine Annealing scheduler.
- Memory Optimization: Micro-batch 2 + Accum 2 + explicit torch.cuda.empty_cache() between epochs.
- Resume capability: Supports starting from epoch N with existing best F1 checkpoint.
"""

import os
import sys
import time
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Project/model4_cascade")
from m4_stage2_model import Stage2Specialist4Class, FocalLoss
from m4_dataset import Stage2Dataset, CLASS_NAMES


def train_model4_3_2(
    train_csv: str = "D:/Project/model4_3_2_retrained/train.csv",
    val_csv: str = "D:/Project/model4_3_2_retrained/val.csv",
    pretrained_weights: str = "D:/Project/model4_3_2_retrained/weights/model4_3_2_best.pt",
    weights_out_dir: str = "D:/Project/model4_3_2_retrained/weights",
    start_epoch: int = 3,
    epochs: int = 25,
    batch_size: int = 2,
    accum_steps: int = 2,
    lr: float = 1e-4,
    patience: int = 3,
    initial_best_f1: float = 0.8713,
    initial_patience_cnt: int = 1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"      RESUMING MODEL 4.3.2 TRAINING (EPOCH {start_epoch}-{epochs})")
    print(f"==================================================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Current Best Macro-F1: {initial_best_f1:.4f} | Current Patience Count: {initial_patience_cnt}/{patience}", flush=True)
    print(f"Loading Weights: {pretrained_weights}", flush=True)
    print(f"Micro Batch: {batch_size} | Accum Steps: {accum_steps} | Effective Batch Size: {batch_size * accum_steps}", flush=True)

    # Datasets
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train Tiles: {len(df_train)} | Val Tiles: {len(df_val)}", flush=True)

    train_ds = Stage2Dataset(df_train, mode="train")
    val_ds = Stage2Dataset(df_val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model 4 architecture
    model = Stage2Specialist4Class(num_classes=4, unfreeze_last_n=3, device=str(device)).to(device)

    # Load weights
    if os.path.exists(pretrained_weights):
        print(f"\n[+] Loading checkpoint weights from {pretrained_weights}...", flush=True)
        ckpt = torch.load(pretrained_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"[+] Loaded weights successfully!", flush=True)
    else:
        print(f"[!] Checkpoint not found at {pretrained_weights}. Checking Model 4.3 champion fallback...", flush=True)
        fallback = "D:/Project/model4_3_retrained/weights/model4_3_best.pt"
        ckpt = torch.load(fallback, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)

    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.phikon_b.parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 1e-2},
        {"params": model.cnn.parameters(), "lr": lr, "weight_decay": 1e-2},
        {"params": model.proj_a.parameters(), "lr": lr, "weight_decay": 1e-2},
        {"params": model.proj_b.parameters(), "lr": lr, "weight_decay": 1e-2},
        {"params": model.gate.parameters(), "lr": lr, "weight_decay": 1e-2},
        {"params": model.head.parameters(), "lr": lr, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Step scheduler up to start_epoch
    for _ in range(start_epoch - 1):
        scheduler.step()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(weights_out_dir, exist_ok=True)
    best_weight_path = os.path.join(weights_out_dir, "model4_3_2_best.pt")
    last_weight_path = os.path.join(weights_out_dir, "model4_3_2_last.pt")

    best_val_f1 = initial_best_f1
    patience_cnt = initial_patience_cnt
    
    # Load history if exists
    history_csv = os.path.join(weights_out_dir, "training_log.csv")
    if os.path.exists(history_csv):
        history = pd.read_csv(history_csv).to_dict(orient="records")
    else:
        history = []

    total_train_batches = len(train_loader)
    print(f"\nResuming training from Epoch {start_epoch} to {epochs} ({total_train_batches} micro-batches per epoch)...", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (crops, img_256, labels) in enumerate(train_loader, 1):
            crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops, img_256)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == total_train_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

            if step % 500 == 0 or step == total_train_batches:
                avg_so_far = total_loss / step
                print(f"  Epoch [{epoch:02d}] Step [{step:04d}/{total_train_batches:04d}] - Running Loss: {avg_so_far:.4f}", flush=True)

        scheduler.step()
        train_loss = total_loss / total_train_batches

        # Validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for crops, img_256, labels in val_loader:
                crops, img_256 = crops.to(device), img_256.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(crops, img_256)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, zero_division=0)
        elapsed = time.time() - t0

        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}", flush=True)
        print(f"    Per-class F1: cohesive={per_class_f1[0]:.4f}, dispersive={per_class_f1[1]:.4f}, no_feature={per_class_f1[2]:.4f}, noncohesive={per_class_f1[3]:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "cohesive_f1": per_class_f1[0],
            "dispersive_f1": per_class_f1[1],
            "no_feature_f1": per_class_f1[2],
            "noncohesive_f1": per_class_f1[3],
        })
        pd.DataFrame(history).to_csv(os.path.join(weights_out_dir, "training_log.csv"), index=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0  # Reset patience counter on new high
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_acc": val_acc,
                "classes": CLASS_NAMES,
            }, best_weight_path)
            print(f"  [+] NEW BEST! Checkpoint saved to: {best_weight_path} (F1: {best_val_f1:.4f})", flush=True)
        else:
            patience_cnt += 1
            print(f"  [-] No improvement for {patience_cnt}/{patience} epoch(s). (Best F1: {best_val_f1:.4f})", flush=True)
            if patience_cnt >= patience:
                print(f"\n[!] Early stopping triggered: No validation improvement for {patience} consecutive epochs.", flush=True)
                break

    # Save last weights
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_weight_path)
    print(f"\n==================================================", flush=True)
    print(f"Training Complete! Final Champion Macro-F1: {best_val_f1:.4f}", flush=True)
    print(f"Best Model Saved: {best_weight_path}", flush=True)
    print(f"==================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--start-epoch", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    args = parser.parse_args()

    train_model4_3_2(
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps
    )
