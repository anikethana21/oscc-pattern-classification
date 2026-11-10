"""
train_model9_3.py
-----------------
Training, Validation, and Evaluation for Model 9.3:
- Tri-Branch Multi-Prototype Sub-Clustering Net (2 sub-centroids per class = 6 total).
- Captures biological intra-class sub-types (e.g. single cells vs micro-nests).
- Nearest sub-prototype max-pooling classification head.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np

sys.path.insert(0, r"D:\Project\model9_hybrid_cluster")
from model9_architecture import Model9_3_MultiProtoNet, CLASSES_3

LABEL_ENC = {n: i for i, n in enumerate(CLASSES_3)}
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


class Model9Dataset(Dataset):
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
        crops_16 = make_16_crops(t_512)
        label = LABEL_ENC[row["label"]]
        return crops_16, t_512, torch.tensor(label, dtype=torch.long), row["path"]


def train_model9_3(
    train_csv=r"D:\Project\model9_hybrid_cluster\train.csv",
    val_csv=r"D:\Project\model9_hybrid_cluster\val.csv",
    out_dir=r"D:\Project\model9_hybrid_cluster",
    epochs=10,
    batch_size=4,
    phikon_lr=1.5e-5,
    head_lr=1e-4,
    patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=================================================================", flush=True)
    print("      TRAINING MODEL 9.3: MULTI-PROTOTYPE SUB-CLUSTERING NET", flush=True)
    print("=================================================================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Classes: {CLASSES_3} (2 Sub-Prototypes per class = 6 Centroids Total)", flush=True)

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    train_ds = Model9Dataset(df_train, mode="train")
    val_ds = Model9Dataset(df_val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = Model9_3_MultiProtoNet(num_classes=3, prototypes_per_class=2, unfreeze_blocks=4, device=str(device)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    phikon_b_trainable = [p for p in model.backbone.phikon_b.parameters() if p.requires_grad]
    other_params = (
        list(model.backbone.cnn.parameters()) +
        list(model.backbone.proj_a.parameters()) +
        list(model.backbone.proj_b.parameters()) +
        list(model.backbone.gate.parameters()) +
        list(model.proj.parameters()) +
        [model.prototypes, model.log_temp]
    )

    optimizer = torch.optim.AdamW([
        {"params": phikon_b_trainable, "lr": phikon_lr, "weight_decay": 1e-2},
        {"params": other_params, "lr": head_lr, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_weights_path = os.path.join(weights_dir, "model9_3_best.pt")

    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        model.train()
        total_loss = 0.0

        for crops_16, img_512, labels, _ in train_loader:
            crops_16, img_512, labels = crops_16.to(device), img_512.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _, _ = model(crops_16, img_512)
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
            for crops_16, img_512, labels, _ in val_loader:
                crops_16, img_512 = crops_16.to(device), img_512.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, _, _, _ = model(crops_16, img_512)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, zero_division=0)
        elapsed = time.time() - t0

        print(f"\nModel 9.3 | Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}", flush=True)
        print(f"    Per-Class F1: Cohesive: {per_class_f1[0]:.4f} | Dispersive: {per_class_f1[1]:.4f} | Noncohesive: {per_class_f1[2]:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_acc": best_val_acc,
                "classes": CLASSES_3,
            }, best_weights_path)
            print(f"  [+] NEW BEST MODEL 9.3! (F1: {best_val_f1:.4f}, Acc: {best_val_acc*100:.2f}%) -> {best_weights_path}", flush=True)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    # Full Evaluation & Artifact Generation
    print("\n--- Generating Model 9.3 Visualizations & Artifacts ---", flush=True)
    ckpt = torch.load(best_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_preds, eval_targets, eval_probs, all_latents, eval_paths = [], [], [], [], []
    with torch.no_grad():
        for crops_16, img_512, labels, paths in val_loader:
            crops_16, img_512 = crops_16.to(device), img_512.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, z, c, _ = model(crops_16, img_512)
                probs = F.softmax(logits, dim=1)
            eval_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            eval_targets.extend(labels.numpy())
            eval_probs.extend(probs.cpu().numpy())
            all_latents.extend(z.cpu().numpy())
            eval_paths.extend(paths)

    final_acc = accuracy_score(eval_targets, eval_preds)
    final_f1 = f1_score(eval_targets, eval_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(eval_targets, eval_preds, average=None, zero_division=0)
    eval_probs = np.array(eval_probs)
    all_latents = np.array(all_latents)

    # 1. Confusion Matrix
    cm = confusion_matrix(eval_targets, eval_preds)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES_3, yticklabels=CLASSES_3, ax=ax,
                annot_kws={"size": 13, "fontweight": "bold"})
    ax.set_title(f"Model 9.3: Multi-Prototype Net Confusion Matrix\nVal Acc: {final_acc*100:.2f}% | Macro-F1: {final_f1:.4f}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_model9_3.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # 2. Multi-Class ROC Curves
    y_true_bin = label_binarize(eval_targets, classes=[0, 1, 2])
    plt.figure(figsize=(7.5, 6.2))
    colors = ["#2ca02c", "#d62728", "#8c564b"]
    for i, (cls_name, color) in enumerate(zip(CLASSES_3, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], eval_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name.capitalize()} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("Model 9.3: Multi-Prototype ROC Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", fontsize=10.5)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve_model9_3.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. 2D Cluster Map with Sub-Prototypes
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(all_latents)
    plt.figure(figsize=(8.5, 7))
    class_markers = {"cohesive": "o", "dispersive": "^", "noncohesive": "s"}
    for idx, cls_name in enumerate(CLASSES_3):
        mask = (np.array(eval_targets) == idx)
        plt.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=colors[idx], label=cls_name.capitalize(),
            alpha=0.80, edgecolors="none", s=55, marker=class_markers[cls_name]
        )
    plt.title(f"Model 9.3 Multi-Prototype Latent Sub-Clusters\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%",
              fontsize=12, fontweight="bold")
    plt.xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    cluster_path = os.path.join(out_dir, "clusters_2d_model9_3.png")
    plt.savefig(cluster_path, dpi=300)
    plt.close()

    metrics = {
        "model": "Model 9.3 (Multi-Prototype Sub-Clustering Net)",
        "val_accuracy": float(final_acc),
        "val_macro_f1": float(final_f1),
        "per_class_f1": {
            "cohesive": float(per_class_f1[0]),
            "dispersive": float(per_class_f1[1]),
            "noncohesive": float(per_class_f1[2])
        }
    }
    with open(os.path.join(out_dir, "model9_3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return final_acc, final_f1


if __name__ == "__main__":
    train_model9_3()

