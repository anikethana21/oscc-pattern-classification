"""
train_model109_4.py
-------------------
Training and Validation Suite for Model 109.4:
- 5-Class Multi-Prototype Deep Metric Clustering Network (10 Sub-Centroids).
- Captures intra-class biological heterogeneity (e.g. cohesive sheets vs nests, dysplasia grades).
- Outlier Radius Gate (R_c = mu_c + 2.5 * sigma_c) rejection engine.
- Outputs 5x5 confusion matrix, 2D PCA cluster map, ROC curves, metrics JSON, and outlier audit.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

sys.path.insert(0, r"D:\Project\model109_4_cluster")
from model109_4_architecture import Model109_4_MultiProtoNet, OutlierRadiusGate, CLASSES_5

LABEL_ENC = {n: i for i, n in enumerate(CLASSES_5)}
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


class OSCC109Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        if mode == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["file_path"]).convert("RGB")
        img_512 = img.resize((512, 512), Image.Resampling.BILINEAR)

        t_512 = self.transform(img_512)
        crops_16 = make_16_crops(t_512)
        label = int(row["label"])
        return crops_16, t_512, torch.tensor(label, dtype=torch.long), row["file_path"], row["class_name"]


def train_model109_4(
    manifest_csv=r"D:\Project\dataset_20x_master\manifest_109_4.csv",
    save_dir=r"D:\Project\model109_4_cluster",
    epochs=10,
    batch_size=4,
    patience=3,
    lr_proto=1e-3,
    lr_head=1e-4,
    lr_backbone=2e-5,
    unfreeze_blocks=4,
):
    start_time = time.time()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    weights_dir = save_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      LAUNCHING MODEL 109.4 (5-CLASS MULTI-PROTOTYPE CLUSTERING)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Runtime: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Manifest
    df = pd.read_csv(manifest_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    outlier_df = df[df["split"] == "outlier_eval"].reset_index(drop=True)
    print(f"Train samples (bio): {len(train_df)} | Val samples (bio): {len(val_df)} | Outlier Eval: {len(outlier_df)}")
    print("Train Class Distribution:\n", train_df["class_name"].value_counts())

    train_ds = OSCC109Dataset(train_df, mode="train")
    val_ds   = OSCC109Dataset(val_df, mode="val")
    outlier_ds = OSCC109Dataset(outlier_df, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    outlier_loader = DataLoader(outlier_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

    # Initialize Model
    print("\nInstantiating Model 109.4 Multi-Prototype Network (10 Sub-Centroids)...")
    model = Model109_4_MultiProtoNet(num_classes=5, prototypes_per_class=2,
                                     unfreeze_blocks=unfreeze_blocks, device=device).to(device)

    # Parameter groups
    proto_params = [model.prototypes, model.log_temp]
    backbone_params = [p for p in model.phikon_b.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("phikon_b") and n != "prototypes" and n != "log_temp" and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": proto_params, "lr": lr_proto, "weight_decay": 1e-4},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    print("\nStarting Training Execution...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for step, (crops_16, img_512, labels, _, _) in enumerate(train_loader):
            crops_16 = crops_16.to(device, non_blocking=True)
            img_512  = img_512.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, z, c, _ = model(crops_16, img_512)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                temp_val = float(torch.exp(model.log_temp).item())
                print(f"  [Epoch {epoch:02d}/{epochs:02d} | Step {step+1:03d}/{len(train_loader):03d}] "
                      f"Step Loss: {loss.item():.4f} | Temp: {temp_val:.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1  = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for crops_16, img_512, labels, _, _ in val_loader:
                crops_16 = crops_16.to(device, non_blocking=True)
                img_512  = img_512.to(device, non_blocking=True)
                labels   = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    logits, _, _, _ = model(crops_16, img_512)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                val_probs.append(probs)
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_ds)
        val_acc  = accuracy_score(val_targets, val_preds)
        val_f1   = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        epoch_dur = (time.time() - epoch_start) / 60.0

        print(f"\n>>> [EPOCH {epoch:02d}/{epochs:02d} COMPLETE ({epoch_dur:.2f} min)]")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train Macro-F1: {train_f1:.4f}")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val Macro-F1:   {val_f1:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "duration_min": epoch_dur
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            best_path = weights_dir / "model109_4_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "classes": CLASSES_5,
            }, best_path)
            print(f"    *** NEW BEST MODEL SAVED: {best_path.name} (Val Macro-F1: {val_f1:.4f}) ***\n")
        else:
            no_improve += 1
            print(f"    No improvement for {no_improve}/{patience} epochs.\n")
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_time = (time.time() - start_time) / 60.0
    print(f"\nModel 109.4 Training Complete in {total_time:.2f} min. Best Epoch: {best_epoch} (Macro-F1: {best_val_f1:.4f})")

    # Evaluation with Best Weights
    print("\nRunning Final Evaluation on Best Model...")
    best_weights = torch.load(weights_dir / "model109_4_best.pt", map_location=device)
    model.load_state_dict(best_weights["model_state_dict"])
    model.eval()

    all_preds, all_targets, all_probs, all_embeddings, all_paths = [], [], [], [], []
    with torch.no_grad():
        for crops_16, img_512, labels, paths, _ in val_loader:
            crops_16 = crops_16.to(device, non_blocking=True)
            img_512  = img_512.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits, z, c, _ = model(crops_16, img_512)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_embeddings.append(z.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_embeddings = np.vstack(all_embeddings)
    final_prototypes = F.normalize(model.prototypes, p=2, dim=-1).detach().cpu().numpy()

    final_acc = accuracy_score(all_targets, all_preds)
    final_f1  = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report_dict = classification_report(all_targets, all_preds, target_names=CLASSES_5, output_dict=True, zero_division=0)
    print("\n=== FINAL MODEL 109.4 CLASSIFICATION REPORT ===")
    print(classification_report(all_targets, all_preds, target_names=CLASSES_5, zero_division=0))

    # Confusion Matrix Plot (Pure Matplotlib)
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Purples)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASSES_5)),
        yticks=np.arange(len(CLASSES_5)),
        xticklabels=CLASSES_5,
        yticklabels=CLASSES_5,
        title=f"Model 109.4 Confusion Matrix (5 Biological Classes)\nAcc: {final_acc*100:.2f}% | Macro-F1: {final_f1:.4f}",
        ylabel="True Class",
        xlabel="Predicted Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    cm_path = save_path / "confusion_matrix_model109_4.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix: {cm_path}")

    # ROC Curves
    y_bin = label_binarize(all_targets, classes=list(range(5)))
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(CLASSES_5):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model 109.4 Multi-Class ROC Curves (5 Biological Classes)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = save_path / "roc_curve_model109_4.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curves: {roc_path}")

    # Fit Outlier Radius Gate on biological embeddings
    print("\nFitting Outlier Radius Gate (R_c = mu_c + 2.5 * sigma_c)...")
    embs_by_class = {}
    for i, cname in enumerate(CLASSES_5):
        embs_by_class[cname] = all_embeddings[all_targets == i]

    outlier_gate = OutlierRadiusGate(n_sigma=2.5)
    outlier_gate.fit(embs_by_class)
    for cname in CLASSES_5:
        print(f"  Class {cname:<12}: Radius R = {outlier_gate.radii[cname]:.4f}")

    # Evaluate Outlier Gate on Held-out Outliers (no_feature & blank)
    print("\nEvaluating Outlier Gate on Held-out Outliers (no_feature, blank)...")
    outlier_embeddings, outlier_paths, outlier_types = [], [], []
    with torch.no_grad():
        for crops_16, img_512, _, paths, cnames in outlier_loader:
            crops_16 = crops_16.to(device, non_blocking=True)
            img_512  = img_512.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                _, z, _, _ = model(crops_16, img_512)
            outlier_embeddings.append(z.cpu().numpy())
            outlier_paths.extend(paths)
            outlier_types.extend(cnames)

    outlier_embeddings = np.vstack(outlier_embeddings)
    rejected_count = 0
    outlier_records = []

    for i in range(len(outlier_embeddings)):
        z_sample = outlier_embeddings[i]
        res = outlier_gate.evaluate(z_sample)
        if res["is_outlier"]:
            rejected_count += 1
        outlier_records.append({
            "file_path": outlier_paths[i],
            "raw_class": outlier_types[i],
            "closest_bio_class": res["closest_class"],
            "distance": res["min_dist"],
            "threshold_radius": res["cluster_radius"],
            "rejected_as_outlier": res["is_outlier"]
        })

    rejection_rate = (rejected_count / len(outlier_embeddings)) * 100.0
    print(f"Outlier Gate Detection Rate: {rejected_count}/{len(outlier_embeddings)} ({rejection_rate:.2f}%) successfully rejected!")

    outlier_audit_path = save_path / "outlier_rejection_audit_109_4.csv"
    pd.DataFrame(outlier_records).to_csv(outlier_audit_path, index=False)

    # 2D PCA Cluster Map
    print("\nGenerating 2D PCA Latent Metric Cluster Map...")
    pca = PCA(n_components=2, random_state=42)
    pca_bio = pca.fit_transform(all_embeddings)
    pca_proto = pca.transform(final_prototypes)
    pca_outlier = pca.transform(outlier_embeddings[:200])

    plt.figure(figsize=(10, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, cname in enumerate(CLASSES_5):
        mask = (all_targets == i)
        plt.scatter(pca_bio[mask, 0], pca_bio[mask, 1], c=colors[i], label=f"{cname} (n={mask.sum()})", alpha=0.6, s=35)
        p_idx1 = i * 2
        p_idx2 = i * 2 + 1
        plt.scatter(pca_proto[p_idx1, 0], pca_proto[p_idx1, 1], marker="*", c=colors[i], edgecolors="black", s=250, label=f"{cname} Sub-Centroids" if i == 0 else None)
        plt.scatter(pca_proto[p_idx2, 0], pca_proto[p_idx2, 1], marker="*", c=colors[i], edgecolors="black", s=250)

    plt.scatter(pca_outlier[:, 0], pca_outlier[:, 1], c="gray", marker="x", alpha=0.35, s=20, label=f"Rejected Outliers (Blank/No-Feature, {rejection_rate:.1f}%)")

    plt.title(f"Model 109.4 Multi-Prototype Deep Metric Clustering (20x OSCC)\n10 Sub-Centroids + Outlier Radius Gate (Rejection: {rejection_rate:.1f}%)")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    pca_plot_path = save_path / "clusters_2d_model109_4.png"
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    print(f"Saved 2D PCA Cluster Map: {pca_plot_path}")

    # Save Predictions CSV
    pred_df = pd.DataFrame({
        "file_path": all_paths,
        "true_label": [CLASSES_5[t] for t in all_targets],
        "pred_label": [CLASSES_5[p] for p in all_preds],
        "is_correct": all_targets == all_preds
    })
    for i, cname in enumerate(CLASSES_5):
        pred_df[f"prob_{cname}"] = all_probs[:, i]
    pred_path = save_path / "val_predictions_109_4.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save Metrics JSON
    metrics = {
        "model": "Model 109.4 (5-Class Multi-Prototype Deep Metric Clustering Net)",
        "num_classes": 5,
        "classes": CLASSES_5,
        "prototypes_per_class": 2,
        "total_prototypes": 10,
        "best_epoch": best_epoch,
        "accuracy": float(final_acc),
        "macro_f1": float(final_f1),
        "duration_minutes": float(total_time),
        "outlier_gate_rejection_rate": float(rejection_rate),
        "cluster_radii": outlier_gate.radii,
        "classification_report": report_dict,
        "history": history
    }
    metrics_path = save_path / "model109_4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved Metrics JSON: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_model109_4()
