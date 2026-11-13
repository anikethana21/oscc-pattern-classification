"""
train_model7_cv.py
------------------
Autonomous 5-Fold Stratified Cross-Validation and Deep Clustering for Model 7.
- Prototypical Phikon Cluster Network with Geometric Outlier Pruning.
- 5-Fold Stratified CV across all 1,450 tiles (Cohesive: 500, Dispersive: 450, Noncohesive: 500).
- Generates:
  1. Best model weights: weights/model7_best.pt & weights/model7_fold{k}_best.pt
  2. Overall Out-of-Fold Confusion Matrix: confusion_matrix_model7.png
  3. Overall Out-of-Fold ROC Curves: roc_curve_model7.png
  4. Overall Out-of-Fold 2D Cluster PCA Map: clusters_2d_model7.png
  5. Predictions CSV: val_predictions_sorted.csv & model7_oof_cv_predictions.csv
  6. CV Performance Summary: cv_metrics_summary.json
"""

import os
import sys
import time
import glob
import json
import shutil
import argparse
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
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


def run_5fold_cross_validation(
    data_dir=r"D:\Project\model7_phikon_cluster\dataset",
    out_dir=r"D:\Project\model7_phikon_cluster",
    desktop_weights_dir=r"C:\Users\anike\Desktop\generator with annotator\model4\weights",
    epochs_per_fold=5,
    batch_size=4,
    lr=1e-4,
    patience=2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=================================================================", flush=True)
    print("      MODEL 7: 5-FOLD STRATIFIED CROSS-VALIDATION & CLUSTERING", flush=True)
    print("=================================================================", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Classes: {CLASSES_3}", flush=True)
    print(f"Epochs per Fold: {epochs_per_fold} | Batch Size: {batch_size} | Patience: {patience}", flush=True)

    # 1. Gather all 1,450 dataset tiles
    records = []
    for cls in CLASSES_3:
        cls_folder = os.path.join(data_dir, cls)
        for f in glob.glob(os.path.join(cls_folder, "*.png")):
            records.append({"path": os.path.abspath(f), "label": cls})

    df_all = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    dataset_csv = os.path.join(out_dir, "dataset.csv")
    df_all.to_csv(dataset_csv, index=False)
    print(f"\n[+] Total Dataset: {len(df_all)} tiles saved to: {dataset_csv}", flush=True)
    print(df_all["label"].value_counts().to_string(), flush=True)

    # 2. Setup Stratified 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(df_all, df_all["label"]))

    oof_preds = [None] * len(df_all)
    oof_targets = [None] * len(df_all)
    oof_probs = [None] * len(df_all)
    oof_embeddings = [None] * len(df_all)
    oof_outlier_flags = [None] * len(df_all)
    oof_sims = [None] * len(df_all)
    oof_fold_ids = [None] * len(df_all)

    fold_metrics = []
    best_overall_f1 = 0.0
    best_overall_fold = 1
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n==================================================", flush=True)
        print(f"   FOLD [{fold_idx}/5] - Train: {len(train_idx)} | Val: {len(val_idx)}", flush=True)
        print(f"==================================================", flush=True)

        df_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_val = df_all.iloc[val_idx].reset_index(drop=True)

        train_ds = Model7Dataset(df_train, mode="train")
        val_ds = Model7Dataset(df_val, mode="val")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Fresh Model instantiation for this fold
        model = Model7PhikonCluster(num_classes=3, device=str(device)).to(device)

        phikon_params = [p for p in model.phikon.parameters() if p.requires_grad]
        head_params = list(model.cluster_head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": phikon_params, "lr": 1.5e-5, "weight_decay": 1e-2},
            {"params": head_params, "lr": lr, "weight_decay": 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_fold, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")
        criterion = OutlierPrunedPrototypicalLoss(outlier_threshold=0.20)

        fold_weights_path = os.path.join(out_dir, "weights", f"model7_fold{fold_idx}_best.pt")
        fold_best_f1 = 0.0
        fold_best_acc = 0.0
        patience_cnt = 0

        for epoch in range(1, epochs_per_fold + 1):
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
            elapsed = time.time() - t0

            print(f"Fold {fold_idx} | Ep [{epoch:02d}/{epochs_per_fold:02d}] ({elapsed:.0f}s) | Loss: {train_loss:.4f} | Outliers Pruned: {total_outliers:02d} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}", flush=True)

            if val_f1 > fold_best_f1:
                fold_best_f1 = val_f1
                fold_best_acc = val_acc
                patience_cnt = 0
                torch.save({
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_macro_f1": fold_best_f1,
                    "val_acc": fold_best_acc,
                    "classes": CLASSES_3,
                }, fold_weights_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Fold {fold_idx}] Early stopping triggered at epoch {epoch}.", flush=True)
                    break

        print(f"  [+] Fold {fold_idx} Best Macro-F1: {fold_best_f1:.4f} | Acc: {fold_best_acc*100:.2f}%", flush=True)
        fold_metrics.append({
            "fold": fold_idx,
            "best_macro_f1": fold_best_f1,
            "best_acc": fold_best_acc
        })

        if fold_best_f1 > best_overall_f1:
            best_overall_f1 = fold_best_f1
            best_overall_fold = fold_idx

        # Collect Out-of-Fold predictions with the best fold weights
        ckpt = torch.load(fold_weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        val_batch_idx = 0
        with torch.no_grad():
            for crops, labels, _ in val_loader:
                crops = crops.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    z, logits, prototypes = model(crops)
                    probs = F.softmax(logits, dim=1)

                preds = torch.argmax(probs, dim=1).cpu().numpy()
                probs_np = probs.cpu().numpy()
                z_np = z.cpu().numpy()
                labels_np = labels.numpy()

                # Cosine similarity to declared true prototype
                proto_norm = F.normalize(prototypes, p=2, dim=-1)
                true_proto = proto_norm[labels.to(device)]
                sims = torch.sum(z * true_proto, dim=-1).cpu().numpy()
                is_outlier = (sims < 0.20)

                for b in range(len(preds)):
                    global_idx = val_idx[val_batch_idx]
                    oof_preds[global_idx] = int(preds[b])
                    oof_targets[global_idx] = int(labels_np[b])
                    oof_probs[global_idx] = probs_np[b]
                    oof_embeddings[global_idx] = z_np[b]
                    oof_outlier_flags[global_idx] = bool(is_outlier[b])
                    oof_sims[global_idx] = float(sims[b])
                    oof_fold_ids[global_idx] = fold_idx
                    val_batch_idx += 1

    # 3. Overall Cross-Validation Performance Synthesis
    oof_targets = np.array(oof_targets)
    oof_preds = np.array(oof_preds)
    oof_probs = np.array(oof_probs)
    oof_embeddings = np.array(oof_embeddings)

    oof_acc = accuracy_score(oof_targets, oof_preds)
    oof_f1 = f1_score(oof_targets, oof_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(oof_targets, oof_preds, average=None, zero_division=0)

    accs = [m["best_acc"] for m in fold_metrics]
    f1s = [m["best_macro_f1"] for m in fold_metrics]

    print("\n=================================================================", flush=True)
    print("      5-FOLD STRATIFIED CROSS-VALIDATION FINAL RESULTS", flush=True)
    print("=================================================================", flush=True)
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: Acc = {m['best_acc']*100:.2f}% | Macro-F1 = {m['best_macro_f1']:.4f}", flush=True)
    print(f"\nMean CV Accuracy: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%", flush=True)
    print(f"Mean CV Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)
    print(f"Aggregated Out-of-Fold (OOF) Accuracy: {oof_acc*100:.2f}%", flush=True)
    print(f"Aggregated Out-of-Fold (OOF) Macro-F1: {oof_f1:.4f}", flush=True)
    print(f"Per-Class OOF F1: cohesive={per_class_f1[0]:.4f}, dispersive={per_class_f1[1]:.4f}, noncohesive={per_class_f1[2]:.4f}", flush=True)
    print("\nOverall OOF Classification Report:")
    print(classification_report(oof_targets, oof_preds, target_names=CLASSES_3, digits=4), flush=True)

    # 4. Save Champion Weights
    best_fold_weights = os.path.join(out_dir, "weights", f"model7_fold{best_overall_fold}_best.pt")
    master_weights = os.path.join(out_dir, "weights", "model7_best.pt")
    shutil.copy2(best_fold_weights, master_weights)
    print(f"[+] Champion Model 7 Weights saved from Fold {best_overall_fold} to: {master_weights}", flush=True)

    if os.path.exists(desktop_weights_dir):
        desktop_dst = os.path.join(desktop_weights_dir, "model7_best.pt")
        shutil.copy2(master_weights, desktop_dst)
        print(f"[+] Deployed weights to Desktop Annotator: {desktop_dst}", flush=True)

    # 5. Out-of-Fold Confusion Matrix Plot
    cm = confusion_matrix(oof_targets, oof_preds)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES_3, yticklabels=CLASSES_3, ax=ax,
                annot_kws={"size": 13, "fontweight": "bold"})
    ax.set_title(f"Model 7: 5-Fold OOF Confusion Matrix (N={len(df_all)})\nOverall CV Acc: {oof_acc*100:.2f}% | Macro-F1: {oof_f1:.4f}",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_model7.png")
    fig.savefig(cm_path, dpi=300)
    fig.savefig(os.path.join(out_dir, "confusion_matrix_model7_cv.png"), dpi=300)
    plt.close(fig)
    print(f"[+] Saved OOF Confusion Matrix: {cm_path}", flush=True)

    # 6. Multi-Class ROC Curves Plot
    y_true_bin = label_binarize(oof_targets, classes=[0, 1, 2])
    plt.figure(figsize=(7.5, 6.2))
    colors = ["#2ca02c", "#d62728", "#8c564b"]
    for i, (cls_name, color) in enumerate(zip(CLASSES_3, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], oof_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name.capitalize()} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("Model 7: 5-Fold Out-of-Fold Multi-Class ROC Curves", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", fontsize=10.5)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve_model7.png")
    plt.savefig(roc_path, dpi=300)
    plt.savefig(os.path.join(out_dir, "roc_curve_model7_cv.png"), dpi=300)
    plt.close()
    print(f"[+] Saved OOF ROC Curves: {roc_path}", flush=True)

    # 7. 2D Latent Prototypical Clusters Plot (PCA Projection)
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(oof_embeddings)
    plt.figure(figsize=(8.5, 7))
    class_markers = {"cohesive": "o", "dispersive": "^", "noncohesive": "s"}

    for idx, cls_name in enumerate(CLASSES_3):
        mask_regular = (oof_targets == idx) & (~np.array(oof_outlier_flags))
        plt.scatter(
            z_2d[mask_regular, 0], z_2d[mask_regular, 1],
            c=colors[idx], label=f"{cls_name.capitalize()} (Cluster)",
            alpha=0.75, edgecolors="none", s=55, marker=class_markers[cls_name]
        )

    # Highlight Outliers
    mask_outliers = np.array(oof_outlier_flags)
    if np.sum(mask_outliers) > 0:
        plt.scatter(
            z_2d[mask_outliers, 0], z_2d[mask_outliers, 1],
            facecolors="none", edgecolors="#ff7f0e", s=110, linewidths=1.8,
            marker="x", label=f"Pruned Outliers (N={np.sum(mask_outliers)})"
        )

    plt.title(f"Model 7 Latent Prototypical Clusters (N={len(df_all)})\nExplained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%",
              fontsize=12, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    cluster_path = os.path.join(out_dir, "clusters_2d_model7.png")
    plt.savefig(cluster_path, dpi=300)
    plt.savefig(os.path.join(out_dir, "clusters_2d_model7_cv.png"), dpi=300)
    plt.close()
    print(f"[+] Saved OOF 2D Cluster Map: {cluster_path}", flush=True)

    # 8. Export Full Out-of-Fold Predictions CSV
    records_out = []
    for i in range(len(df_all)):
        pred_c = CLASSES_3[oof_preds[i]]
        true_c = CLASSES_3[oof_targets[i]]
        conf = float(oof_probs[i][oof_preds[i]])
        records_out.append({
            "fold": oof_fold_ids[i],
            "path": df_all.iloc[i]["path"],
            "filename": os.path.basename(df_all.iloc[i]["path"]),
            "true_label": true_c,
            "pred_label": pred_c,
            "confidence": conf,
            "is_correct": (pred_c == true_c),
            "is_outlier": oof_outlier_flags[i],
            "cos_sim_to_prototype": oof_sims[i],
            "prob_cohesive": float(oof_probs[i][0]),
            "prob_dispersive": float(oof_probs[i][1]),
            "prob_noncohesive": float(oof_probs[i][2]),
        })

    df_out = pd.DataFrame(records_out).sort_values(by="confidence", ascending=True)
    csv_path = os.path.join(out_dir, "val_predictions_sorted.csv")
    csv_oof = os.path.join(out_dir, "model7_oof_cv_predictions.csv")
    df_out.to_csv(csv_path, index=False)
    df_out.to_csv(csv_oof, index=False)
    print(f"[+] Saved OOF Predictions CSV: {csv_path}", flush=True)

    # 9. Save Summary JSON
    summary_data = {
        "model": "Model 7 (Phikon Prototypical Clustering + Outlier Pruning)",
        "dataset_size": len(df_all),
        "classes": CLASSES_3,
        "n_folds": 5,
        "fold_metrics": fold_metrics,
        "mean_cv_accuracy": float(np.mean(accs)),
        "std_cv_accuracy": float(np.std(accs)),
        "mean_cv_macro_f1": float(np.mean(f1s)),
        "std_cv_macro_f1": float(np.std(f1s)),
        "aggregated_oof_accuracy": float(oof_acc),
        "aggregated_oof_macro_f1": float(oof_f1),
        "per_class_oof_f1": {
            "cohesive": float(per_class_f1[0]),
            "dispersive": float(per_class_f1[1]),
            "noncohesive": float(per_class_f1[2])
        },
        "total_pruned_outliers": int(np.sum(mask_outliers)),
        "best_fold": best_overall_fold,
        "best_fold_f1": float(best_overall_f1)
    }
    summary_json_path = os.path.join(out_dir, "cv_metrics_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved CV Summary: {summary_json_path}", flush=True)
    print("\n=================================================================", flush=True)
    print("      CROSS-VALIDATION EXECUTION COMPLETED SUCCESSFULLY!", flush=True)
    print("=================================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    run_5fold_cross_validation(
        epochs_per_fold=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )

