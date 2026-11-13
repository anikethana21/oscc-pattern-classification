"""
evaluate_model4_3_2.py
----------------------
Runs high-fidelity evaluation of Model 4.3.2 on the validation set.
Outputs:
  - Confusion Matrix PNG: D:/Project/model4_3_2_retrained/confusion_matrix_model4_3_2.png
  - Sorted Predictions CSV: D:/Project/model4_3_2_retrained/val_predictions_sorted.csv (lowest-confidence at top)
  - Detailed Precision, Recall, F1 report per class
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

sys.path.insert(0, "D:/Project/model4_cascade")
from m4_stage2_model import Stage2Specialist4Class
from m4_dataset import Stage2Dataset, CLASS_NAMES

VAL_CSV = r"D:\Project\model4_3_2_retrained\val.csv"
WEIGHTS_PATH = r"D:\Project\model4_3_2_retrained\weights\model4_3_2_best.pt"
OUTPUT_DIR = r"D:\Project\model4_3_2_retrained"

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"      EVALUATING MODEL 4.3.2 ON VALIDATION SET")
    print(f"==================================================")
    print(f"Device: {device}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Val CSV: {VAL_CSV}")

    df_val = pd.read_csv(VAL_CSV)
    val_ds = Stage2Dataset(df_val, mode="val")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = Stage2Specialist4Class(num_classes=4, unfreeze_last_n=0, device=str(device)).to(device)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for crops, img_256, labels in val_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(crops, img_256)
                probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print(f"\n[+] Validation Accuracy: {acc*100:.2f}%")
    print(f"[+] Validation Macro-F1: {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"Model 4.3.2 Confusion Matrix (Val Acc: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_model4_3_2.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[+] Confusion matrix saved to: {cm_path}")

    # Build predictions CSV sorted by confidence (lowest confidence at top)
    records = []
    for i, row in df_val.iterrows():
        pred_c = CLASS_NAMES[all_preds[i]]
        true_c = CLASS_NAMES[all_targets[i]]
        conf = float(all_probs[i][all_preds[i]])
        records.append({
            "path": row["path"],
            "filename": os.path.basename(row["path"]),
            "true_label": true_c,
            "pred_label": pred_c,
            "confidence": conf,
            "is_correct": (pred_c == true_c),
            "prob_cohesive": float(all_probs[i][0]),
            "prob_dispersive": float(all_probs[i][1]),
            "prob_no_feature": float(all_probs[i][2]),
            "prob_noncohesive": float(all_probs[i][3]),
        })

    df_out = pd.DataFrame(records)
    df_out = df_out.sort_values(by="confidence", ascending=True)
    csv_path = os.path.join(OUTPUT_DIR, "val_predictions_sorted.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[+] Predictions CSV saved to: {csv_path} (sorted with poorest predictions at top)")

if __name__ == "__main__":
    evaluate()

