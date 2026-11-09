"""
train_model7.py
---------------
Training, Evaluation, and Cluster Visualization for Model 7.
- Prototypical Phikon Cluster Network with Geometric Outlier Pruning.
- Produces:
  1. Best model weights: model7_best.pt
  2. Confusion Matrix: confusion_matrix_model7.png
  3. Multi-Class ROC Curves with AUC: roc_curve_model7.png
  4. 2D Cluster Visualization: clusters_2d_model7.png
  5. Predictions sorted CSV: val_predictions_sorted.csv
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

sys.path.insert(0, r"D:\Project\model7_phikon_cluster")
from model7_cluster_net import Model7PhikonCluster, OutlierPrunedPrototypicalLoss, CLASSES_3

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


class Model7Dataset(Dataset):
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
        label = LABEL_ENC[row["label"]]
        return crops, torch.tensor(label, dtype=torch.long), row["path"]


def train_model7(
    train_csv=r"D:\Project\model7_phikon_cluster\train.csv",
    val_csv=r"D:\Project\model7_phikon_cluster\val.csv",
    out_dir=r"D:\Project\model7_phikon_cluster",
    epochs=12,
    batch_size=4,
    lr=1e-4,
    patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================================", flush=True)
    print("      TRAINING MODEL 7: PHIKON DEEP CLUSTERING", flush=True)
    print("==================================================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Classes (3 Cancer Patterns): {CLASSES_3}", flush=True)
    print(f"Batch Size: {batch_size} | Base LR: {lr} | Patience: {patience}", flush=True)

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    print(f"Train Tiles: {len(df_train)} | Val Tiles: {len(df_val)}", flush=True)

    train_ds = Model7Dataset(df_train, mode="train")
    val_ds = Model7Dataset(df_val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = Model7PhikonCluster(num_classes=3, device=str(device)).to(device)

    # Distinct learning rates: low for unfrozen ViT layers, standard for cluster head
    phikon_params = [p for p in model.phikon.parameters() if p.requires_grad]
    head_params = list(model.cluster_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": phikon_params, "lr": 1.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": lr, "weight_decay": 1e-2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = OutlierPrunedPrototypicalLoss(outlier_threshold=0.20)

    best_val_f1 = 0.0
    patience_cnt = 0
    weights_path = os.path.join(out_dir, "weights", "model7_best.pt")

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_outliers = 0

        for crops, labels, _ in train_loader:
            crops, labels = crops.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                z, logits, prototypes = model(crops)
                loss, n_out = criterion(logits, labels, z, prototypes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_outliers += n_out

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for crops, labels, _ in val_loader:
                crops = crops.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    z, logits, _ = model(crops)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        per_class_f1 = f1_score(val_targets, val_preds, average=None, zero_division=0)
        elapsed = time.time() - t0

        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.0f}s) | Train Loss: {train_loss:.4f} | Outliers Pruned: {total_outliers} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}", flush=True)
        print(f"    Per-class F1: cohesive={per_class_f1[0]:.4f}, dispersive={per_class_f1[1]:.4f}, noncohesive={per_class_f1[2]:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_f1,
                "val_acc": val_acc,
                "classes": CLASSES_3,
            }, weights_path)
            print(f"  [+] NEW BEST! Checkpoint saved to: {weights_path} (F1: {best_val_f1:.4f})", flush=True)
        else:
            patience_cnt += 1
            print(f"  [-] No improvement for {patience_cnt}/{patience} epoch(s). (Best F1: {best_val_f1:.4f})", flush=True)
            if patience_cnt >= patience:
                print(f"\n[!] Early stopping triggered: No validation improvement for {patience} consecutive epochs.", flush=True)
                break

    print(f"\n==================================================", flush=True)
    print(f"Training Complete! Champion Macro-F1: {best_val_f1:.4f}", flush=True)
    print(f"==================================================", flush=True)

    # Full Evaluation & Visual Artifacts Generation
    print("\n--- Generating Model 7 Visualizations & Artifacts ---", flush=True)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_preds, eval_targets, eval_probs, all_embeddings, eval_paths = [], [], [], [], []
    with torch.no_grad():
        for crops, labels, paths in val_loader:
            crops = crops.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                z, logits, _ = model(crops)
                probs = F.softmax(logits, dim=1)
            eval_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            eval_targets.extend(labels.numpy())
            eval_probs.extend(probs.cpu().numpy())
            all_embeddings.extend(z.cpu().numpy())
            eval_paths.extend(paths)

    eval_probs = np.array(eval_probs)
    all_embeddings = np.array(all_embeddings)
    final_acc = accuracy_score(eval_targets, eval_preds)
    final_f1 = f1_score(eval_targets, eval_preds, average="macro", zero_division=0)

    print(f"\nFinal Val Accuracy: {final_acc*100:.2f}% | Final Macro-F1: {final_f1:.4f}", flush=True)
    print("\nClassification Report:")
    print(classification_report(eval_targets, eval_preds, target_names=CLASSES_3, digits=4), flush=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(eval_targets, eval_preds)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_3, yticklabels=CLASSES_3, ax=ax, annot_kws={"size": 12})
    ax.set_title(f"Model 7 Confusion Matrix (Val Acc: {final_acc*100:.2f}%, F1: {final_f1:.4f})", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_model7.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[+] Confusion matrix saved to: {cm_path}", flush=True)

    # 2. Multi-Class ROC Curves
    y_true_bin = label_binarize(eval_targets, classes=[0, 1, 2])
    plt.figure(figsize=(7, 6))
    colors = ["#2ca02c", "#d62728", "#8c564b"]
    for i, (cls_name, color) in enumerate(zip(CLASSES_3, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], eval_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.0, label=f"{cls_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve - Model 7 Prototypical Clusters", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="-", alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve_model7.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[+] ROC curve saved to: {roc_path}", flush=True)

    # 3. 2D Cluster Visualization (PCA Projection)
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_embeddings)
    plt.figure(figsize=(8, 6.5))
    class_markers = {"cohesive": "o", "dispersive": "^", "noncohesive": "s"}
    for idx, cls_name in enumerate(CLASSES_3):
        mask = (np.array(eval_targets) == idx)
        plt.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            c=colors[idx], label=cls_name,
            alpha=0.85, edgecolors="k", s=65, marker=class_markers[cls_name]
        )

    plt.title(f"Model 7 Latent Prototypical Clusters (Explained Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    cluster_path = os.path.join(out_dir, "clusters_2d_model7.png")
    plt.savefig(cluster_path, dpi=300)
    plt.close()
    print(f"[+] 2D Cluster map saved to: {cluster_path}", flush=True)

    # 4. Predictions Sorted CSV
    records = []
    for i in range(len(eval_targets)):
        pred_c = CLASSES_3[eval_preds[i]]
        true_c = CLASSES_3[eval_targets[i]]
        conf = float(eval_probs[i][eval_preds[i]])
        records.append({
            "path": eval_paths[i],
            "filename": os.path.basename(eval_paths[i]),
            "true_label": true_c,
            "pred_label": pred_c,
            "confidence": conf,
            "is_correct": (pred_c == true_c),
            "prob_cohesive": float(eval_probs[i][0]),
            "prob_dispersive": float(eval_probs[i][1]),
            "prob_noncohesive": float(eval_probs[i][2]),
        })

    df_out = pd.DataFrame(records).sort_values(by="confidence", ascending=True)
    csv_path = os.path.join(out_dir, "val_predictions_sorted.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[+] Predictions CSV saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_model7(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)

