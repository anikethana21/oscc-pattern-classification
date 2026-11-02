"""
m4_2_train.py
-------------
Training script for Model 4.2 Sub-classifier.
Uses:
  - FP16 AMP (torch.cuda.amp.autocast) with GradScaler for 8GB VRAM efficiency.
  - Multi-class Focal Loss (gamma=2.0).
  - Cosine annealing learning rate scheduler.
  - Checkpoint tracking based on validation Macro-F1.
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Project/model4_2_subclassifier")
from m4_2_model import Model4_2_Classifier, FocalLoss
from m4_2_dataset import Model4_2_Dataset


def train_model4_2(
    csv_path: str = "D:/Project/model4_2_subclassifier/seniors_balanced_train.csv",
    classes_json: str = "D:/Project/model4_2_subclassifier/classes.json",
    weights_dir: str = "D:/Project/model4_2_subclassifier/weights",
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 7
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Training Model 4.2 on {device} ===")

    with open(classes_json, "r") as f:
        classes = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Target Sub-classes ({num_classes}): {classes}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total tiles.")

    # Train / Val Split (85% train, 15% val stratified)
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df["label"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train set: {len(df_train)} tiles | Val set: {len(df_val)} tiles")

    train_ds = Model4_2_Dataset(df_train, class_to_idx, mode="train")
    val_ds = Model4_2_Dataset(df_val, class_to_idx, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model 4.2
    model = Model4_2_Classifier(num_classes=num_classes, unfreeze_last_n=3, device=str(device)).to(device)

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

    best_val_f1 = 0.0
    patience_cnt = 0
    os.makedirs(weights_dir, exist_ok=True)
    best_weight_path = os.path.join(weights_dir, "model4_2_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for crops_224, img_256, labels in train_loader:
            crops_224 = crops_224.to(device)
            img_256 = img_256.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops_224, img_256)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for crops_224, img_256, labels in val_loader:
                crops_224 = crops_224.to(device)
                img_256 = img_256.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(crops_224, img_256)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "val_macro_f1": best_val_f1,
                "epoch": epoch
            }, best_weight_path)
            print(f"  --> Saved new best checkpoint (F1: {best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"\n[+] Training Complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f"[+] Saved model to: {best_weight_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="D:/Project/model4_2_subclassifier/seniors_balanced_train.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model4_2(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

