"""
m4_3_1_train.py
---------------
Training script for Model 4.3.1 (Retrained on merged 5,491 tiles from new Durgesh + Amrutha data).
- Architecture: Model 4 Specialist 4-Class Network (Phikon Frozen A + Phikon Fine-tuned B + TileCNN C + Dynamic Gating).
- Pretrained Backbone Weights: Initializes from champion Model 4.3 (D:/Project/model4_3_retrained/weights/model4_3_best.pt).
- Early Stopping: Strict patience = 3 (stops early if no validation Macro-F1 improvement for 3 consecutive epochs).
- Precision: FP16 Automatic Mixed Precision (AMP) with GradScaler.
- Loss: Multi-class Focal Loss (gamma=2.0).
- Optimizer: AdamW with Cosine Annealing scheduler.
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


def train_model4_3_1(
    train_csv: str = "D:/Project/model4_3_1_retrained/train.csv",
    val_csv: str = "D:/Project/model4_3_1_retrained/val.csv",
    pretrained_weights: str = "D:/Project/model4_3_retrained/weights/model4_3_best.pt",
    weights_out_dir: str = "D:/Project/model4_3_1_retrained/weights",
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"      TRAINING MODEL 4.3.1 (4-CLASS SPECIALIST)")
    print(f"==================================================")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Patience: {patience} (Strict 3-epoch early stopping)")
    print(f"Pretrained Weights: {pretrained_weights}")
    print(f"Batch Size: {batch_size} | Base LR: {lr}")

    # Datasets
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train Tiles: {len(df_train)} | Val Tiles: {len(df_val)}")

    train_ds = Stage2Dataset(df_train, mode="train")
    val_ds = Stage2Dataset(df_val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model 4 architecture
    model = Stage2Specialist4Class(num_classes=4, unfreeze_last_n=3, device=str(device)).to(device)

    # Transfer learn from champion Model 4.3 weights
    if os.path.exists(pretrained_weights):
        print(f"\n[+] Loading backbone & head weights from Model 4.3 champion checkpoint...")
        ckpt = torch.load(pretrained_weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"[+] Loaded 100% of weights successfully!")
    else:
        print(f"[!] Warning: Pretrained weights not found at {pretrained_weights}. Checking Model 4 fallback...")
        fallback = "D:/Project/model4_cascade/weights_s2/stage2_best.pt"
        if os.path.exists(fallback):
            ckpt = torch.load(fallback, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"], strict=True)
            print(f"[+] Loaded fallback Model 4 Stage 2 weights!")

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
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(weights_out_dir, exist_ok=True)
    best_weight_path = os.path.join(weights_out_dir, "model4_3_1_best.pt")
    last_weight_path = os.path.join(weights_out_dir, "model4_3_1_last.pt")

    # Initial Zero-Shot Evaluation of preloaded weights on this validation set
    print("\n--- Pre-training Baseline Evaluation on New Val Set ---")
    model.eval()
    init_preds, init_targets = [], []
    with torch.no_grad():
        for crops, img_256, labels in val_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops, img_256)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            init_preds.extend(preds)
            init_targets.extend(labels.numpy())

    init_f1 = f1_score(init_targets, init_preds, average="macro", zero_division=0)
    init_acc = accuracy_score(init_targets, init_preds)
    print(f"Preloaded Baseline -> Val Acc: {init_acc*100:.2f}% | Val Macro-F1: {init_f1:.4f}")

    best_val_f1 = init_f1
    patience_cnt = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for crops, img_256, labels in train_loader:
            crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops, img_256)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
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

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")
        print(f"  Per-class F1: cohesive={per_class_f1[0]:.4f}, dispersive={per_class_f1[1]:.4f}, no_feature={per_class_f1[2]:.4f}, noncohesive={per_class_f1[3]:.4f}")

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
            print(f"  [+] NEW BEST! Checkpoint saved to: {best_weight_path} (F1: {best_val_f1:.4f})")
        else:
            patience_cnt += 1
            print(f"  [-] No improvement for {patience_cnt}/{patience} epoch(s). (Best F1: {best_val_f1:.4f})")
            if patience_cnt >= patience:
                print(f"\n[!] Early stopping triggered: No validation improvement for {patience} consecutive epochs.")
                break

    # Save last weights
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_weight_path)
    print(f"\n==================================================")
    print(f"Training Complete! Final Champion Macro-F1: {best_val_f1:.4f}")
    print(f"Best Model Saved: {best_weight_path}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train_model4_3_1(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)

