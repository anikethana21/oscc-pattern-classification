"""
evaluate_model4_3.py
--------------------
Comprehensive evaluation script for Model 4.3 on the validation split (831 tiles).
Generates:
  - Confusion matrix plot (D:/Project/model4_3_retrained/confusion_matrix_model4_3.png)
  - Detailed classification report (accuracy, precision, recall, F1 per class, macro F1, weighted F1)
  - Full scored CSV sorted with least confident tiles at top (D:/Project/model4_3_retrained/val_predictions_sorted.csv)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, "D:/Project/model4_cascade")
from m4_stage2_model import Stage2Specialist4Class
from m4_dataset import Stage2Dataset, CLASS_NAMES


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_csv = "D:/Project/model4_3_retrained/val.csv"
    best_weights = "D:/Project/model4_3_retrained/weights/model4_3_best.pt"
    out_dir = "D:/Project/model4_3_retrained"

    if not os.path.exists(best_weights):
        print(f"Weights file not ready yet at: {best_weights}")
        return

    print(f"Loading best weights from: {best_weights}")
    ckpt = torch.load(best_weights, map_location=device, weights_only=False)

    model = Stage2Specialist4Class(num_classes=4, device=str(device)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df_val = pd.read_csv(val_csv)
    val_ds = Stage2Dataset(df_val, mode="val")
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for crops, img_256, labels in val_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops, img_256)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

    print("\n" + "="*50)
    print("      MODEL 4.3 COMPREHENSIVE EVALUATION RESULTS")
    print("="*50)
    print(f"Validation Tiles: {len(all_targets)}")
    print(f"Overall Accuracy : {acc*100:.2f}% ({np.sum(all_targets == all_preds)}/{len(all_targets)})")
    print(f"Macro-F1 Score   : {macro_f1:.4f}")
    print(f"Weighted-F1 Score: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"Model 4.3 Validation Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Pattern", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Pathologist Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix_model4_3.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[+] Saved Confusion Matrix to: {cm_path}")

    # Scored CSV
    confidences = np.max(all_probs, axis=1)
    df_results = pd.DataFrame({
        "tile_path": df_val["path"],
        "true_label": [CLASS_NAMES[t] for t in all_targets],
        "predicted_class": [CLASS_NAMES[p] for p in all_preds],
        "confidence": np.round(confidences, 4),
        "is_correct": (all_targets == all_preds),
        "prob_cohesive": np.round(all_probs[:, 0], 4),
        "prob_dispersive": np.round(all_probs[:, 1], 4),
        "prob_no_feature": np.round(all_probs[:, 2], 4),
        "prob_noncohesive": np.round(all_probs[:, 3], 4),
    })

    # Sort ascending (least confident at top)
    df_sorted = df_results.sort_values(by="confidence", ascending=True).reset_index(drop=True)
    sorted_csv = os.path.join(out_dir, "val_predictions_sorted.csv")
    df_sorted.to_csv(sorted_csv, index=False)
    print(f"[+] Saved sorted predictions to: {sorted_csv}")


if __name__ == "__main__":
    evaluate()

