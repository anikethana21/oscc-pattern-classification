"""
m4_3_2_1_train.py
-----------------
Trains Model 4.3.2.1: 3-Class Cancer Sub-Specialist Network.
- Classes (3): cohesive, dispersive, noncohesive (no_feature dropped).
- Architecture: Model 4 Specialist (Dual Phikon ViT + TileCNN + Gating + 3-Class Head).
- Pretrained Weights: Warm-started from champion Model 4.3.2 (with head sliced for the 3 cancer classes).
- Fast execution on RTX 4070 (1,030 train tiles, ~60s per epoch).
- Strict Early Stopping (patience = 3).
"""

import os
import sys
import time
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Project/model4_cascade")
from m4_stage2_model import Stage2Specialist4Class, FocalLoss

CLASS_NAMES_3 = ["cohesive", "dispersive", "noncohesive"]
LABEL_ENC_3 = {n: i for i, n in enumerate(CLASS_NAMES_3)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_16_crops(img_tensor_512: torch.Tensor) -> torch.Tensor:
    crops = []
    for r in range(4):
        for c in range(4):
            crop = img_tensor_512[:, r*128:(r+1)*128, c*128:(c+1)*128]
            crop = F.interpolate(crop.unsqueeze(0), size=(224, 224),
                                 mode="bilinear", align_corners=False).squeeze(0)
            crops.append(crop)
    return torch.stack(crops, dim=0)


class Stage2Dataset3Class(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        if mode == "train":
            self.aug = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img_512 = img.resize((512, 512), Image.Resampling.BILINEAR)
        arr_512 = np.array(img_512, dtype=np.uint8)

        t_512 = self.aug(image=arr_512)["image"]
        crops = make_16_crops(t_512)
        img_256 = F.interpolate(t_512.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).squeeze(0)

        label = LABEL_ENC_3[row["label"]]
        return crops, img_256, torch.tensor(label, dtype=torch.long)


def train_model4_3_2_1(
    train_csv: str = "D:/Project/model4_3_2_1_retrained/train.csv",
    val_csv: str = "D:/Project/model4_3_2_1_retrained/val.csv",
    pretrained_weights: str = "D:/Project/model4_3_2_retrained/weights/model4_3_2_best.pt",
    weights_out_dir: str = "D:/Project/model4_3_2_1_retrained/weights",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    patience: int = 3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================", flush=True)
    print(f"      TRAINING MODEL 4.3.2.1 (3-CLASS SPECIALIST)", flush=True)
    print(f"==================================================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Classes (3): {CLASS_NAMES_3}", flush=True)
    print(f"Batch Size: {batch_size} | Base LR: {lr} | Patience: {patience}", flush=True)

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train Tiles: {len(df_train)} | Val Tiles: {len(df_val)}", flush=True)

    train_ds = Stage2Dataset3Class(df_train, mode="train")
    val_ds = Stage2Dataset3Class(df_val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize 3-class specialist architecture
    model = Stage2Specialist4Class(num_classes=3, unfreeze_last_n=3, device=str(device)).to(device)

    # Transfer learn from Model 4.3.2 champion
    if os.path.exists(pretrained_weights):
        print(f"\n[+] Warm-starting from Model 4.3.2 champion ({pretrained_weights})...", flush=True)
        ckpt = torch.load(pretrained_weights, map_location=device, weights_only=False)
        m432_state = ckpt["model_state"]

        model_dict = model.state_dict()
        transferred = 0
        for k, v in m432_state.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    transferred += 1
                elif "head.6" in k:  # Classification linear layer
                    # Old: [4, 128] (cohesive=0, dispersive=1, no_feature=2, noncohesive=3)
                    # New: [3, 128] (cohesive=0, dispersive=1, noncohesive=2)
                    if "weight" in k:
                        model_dict[k] = v[[0, 1, 3], :]
                        print(f"  [+] Sliced head weight [3, 128] from 4.3.2", flush=True)
                    elif "bias" in k:
                        model_dict[k] = v[[0, 1, 3]]
                        print(f"  [+] Sliced head bias [3] from 4.3.2", flush=True)
                    transferred += 1

        model.load_state_dict(model_dict)
        print(f"[+] Successfully loaded {transferred} parameters into Model 4.3.2.1!", flush=True)

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
    best_weight_path = os.path.join(weights_out_dir, "model4_3_2_1_best.pt")

    # Initial Zero-Shot Baseline
    print("\n--- Zero-Shot Pre-training Baseline on Val Set ---", flush=True)
    model.eval()
    init_preds, init_targets = [], []
    with torch.no_grad():
        for crops, img_256, labels in val_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(crops, img_256)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            init_preds.extend(preds)
            init_targets.extend(labels.numpy())

    init_f1 = f1_score(init_targets, init_preds, average="macro", zero_division=0)
    init_acc = accuracy_score(init_targets, init_preds)
    print(f"Preloaded Baseline -> Val Acc: {init_acc*100:.2f}% | Val Macro-F1: {init_f1:.4f}", flush=True)

    best_val_f1 = init_f1
    patience_cnt = 0
    history = []

    print(f"\nStarting training loop across {epochs} epochs...", flush=True)

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        model.train()
        total_loss = 0.0

        for crops, img_256, labels in train_loader:
            crops, img_256, labels = crops.to(device), img_256.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(crops, img_256)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for crops, img_256, labels in val_loader:
                crops, img_256 = crops.to(device), img_256.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, img_256)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, zero_division=0)
        elapsed = time.time() - t0

        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}", flush=True)
        print(f"    Per-class F1: cohesive={per_class_f1[0]:.4f}, dispersive={per_class_f1[1]:.4f}, noncohesive={per_class_f1[2]:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "cohesive_f1": per_class_f1[0],
            "dispersive_f1": per_class_f1[1],
            "noncohesive_f1": per_class_f1[2],
        })
        pd.DataFrame(history).to_csv(os.path.join(weights_out_dir, "training_log.csv"), index=False)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_acc": val_acc,
                "classes": CLASS_NAMES_3,
            }, best_weight_path)
            print(f"  [+] NEW BEST! Checkpoint saved to: {best_weight_path} (F1: {best_val_f1:.4f})", flush=True)
        else:
            patience_cnt += 1
            print(f"  [-] No improvement for {patience_cnt}/{patience} epoch(s). (Best F1: {best_val_f1:.4f})", flush=True)
            if patience_cnt >= patience:
                print(f"\n[!] Early stopping triggered: No validation improvement for {patience} consecutive epochs.", flush=True)
                break

    print(f"\n==================================================", flush=True)
    print(f"Training Complete! Champion Macro-F1: {best_val_f1:.4f}", flush=True)
    print(f"Saved: {best_weight_path}", flush=True)
    print(f"==================================================", flush=True)

    # Run full final evaluation
    print("\n--- Running Final Evaluation & Generating Artifacts ---", flush=True)
    ckpt = torch.load(best_weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_preds, eval_targets, eval_probs = [], [], []
    with torch.no_grad():
        for crops, img_256, labels in val_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(crops, img_256)
                probs = F.softmax(logits, dim=1)
            eval_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            eval_targets.extend(labels.numpy())
            eval_probs.extend(probs.cpu().numpy())

    eval_probs = np.array(eval_probs)
    final_acc = accuracy_score(eval_targets, eval_preds)
    final_f1 = f1_score(eval_targets, eval_preds, average="macro", zero_division=0)

    print(f"\nFinal Val Accuracy: {final_acc*100:.2f}% | Final Macro-F1: {final_f1:.4f}", flush=True)
    print("\nClassification Report:")
    print(classification_report(eval_targets, eval_preds, target_names=CLASS_NAMES_3, digits=4), flush=True)

    # Confusion matrix
    cm = confusion_matrix(eval_targets, eval_preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES_3, yticklabels=CLASS_NAMES_3, ax=ax)
    ax.set_title(f"Model 4.3.2.1 Confusion Matrix (Val Acc: {final_acc*100:.2f}%, F1: {final_f1:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join("D:/Project/model4_3_2_1_retrained", "confusion_matrix_model4_3_2_1.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[+] Confusion matrix saved to: {cm_path}", flush=True)

    # Sorted predictions CSV
    records = []
    for i, row in df_val.iterrows():
        pred_c = CLASS_NAMES_3[eval_preds[i]]
        true_c = CLASS_NAMES_3[eval_targets[i]]
        conf = float(eval_probs[i][eval_preds[i]])
        records.append({
            "path": row["path"],
            "filename": os.path.basename(row["path"]),
            "true_label": true_c,
            "pred_label": pred_c,
            "confidence": conf,
            "is_correct": (pred_c == true_c),
            "prob_cohesive": float(eval_probs[i][0]),
            "prob_dispersive": float(eval_probs[i][1]),
            "prob_noncohesive": float(eval_probs[i][2]),
        })

    df_out = pd.DataFrame(records).sort_values(by="confidence", ascending=True)
    csv_path = os.path.join("D:/Project/model4_3_2_1_retrained", "val_predictions_sorted.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[+] Predictions CSV saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_model4_3_2_1(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)

