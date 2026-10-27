import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_eval_cascade.py
------------------
Evaluate Model 4 on the locked test set.
Stage 1 screens cohesive (0) and no_feature (2).
Remaining/ambiguous tiles pass to Stage 2 (4-class specialist).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from m4_stage1_model import Stage1WithBackbone
from m4_stage2_model import Stage2Specialist4Class
from m4_dataset import Stage1Dataset, Stage2Dataset, CLASS_NAMES


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv",   default="hybridpath_model/split_test.csv")
    ap.add_argument("--s1-dir",     default="model4_cascade/weights_s1")
    ap.add_argument("--s2-dir",     default="model4_cascade/weights_s2")
    ap.add_argument("--threshold",  type=float, default=0.75)
    ap.add_argument("--batch-size", type=int,   default=4)
    ap.add_argument("--out-dir",    default="model4_cascade")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1
    s1_model = Stage1WithBackbone(device=device).to(device)
    s1_ckpt  = torch.load(Path(args.s1_dir) / "stage1_best.pt", map_location=device, weights_only=False)
    s1_model.load_state_dict(s1_ckpt["model_state"])
    s1_model.eval()

    nm = Path(args.s1_dir) / "named_mean.npy"
    ns = Path(args.s1_dir) / "named_std.npy"
    named_mean = np.load(nm) if nm.exists() else None
    named_std  = np.load(ns) if ns.exists() else None

    # Load Stage 2
    s2_model = Stage2Specialist4Class(num_classes=4, device=device).to(device)
    s2_ckpt  = torch.load(Path(args.s2_dir) / "stage2_best.pt", map_location=device, weights_only=False)
    s2_model.load_state_dict(s2_ckpt["model_state"])
    s2_model.eval()

    df_test = pd.read_csv(args.test_csv)
    print(f"Locked Test Set: {len(df_test)} tiles", flush=True)

    # 1. Run Stage 1 on all tiles
    s1_ds = Stage1Dataset(df_test, mode="test", named_mean=named_mean, named_std=named_std)
    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False)

    s1_probs = []
    with torch.no_grad():
        for crops, named, _ in s1_loader:
            crops, named = crops.to(device), named.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = s1_model(crops, named)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            s1_probs.append(probs)
    s1_probs = np.concatenate(s1_probs, axis=0)  # [N, 2] -> 0: cohesive, 1: no_feature

    # 2. Run Stage 2 on all tiles
    s2_ds = Stage2Dataset(df_test, mode="test")
    s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False)

    s2_probs = []
    true_labels = []
    with torch.no_grad():
        for crops, img_256, labels in s2_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = s2_model(crops, img_256)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            s2_probs.append(probs)
            true_labels.extend(labels.numpy())
    s2_probs = np.concatenate(s2_probs, axis=0)  # [N, 4]
    true_labels = np.array(true_labels)

    # 3. Decision Logic & Probability Extraction:
    # If Stage 1 max prob >= threshold, assign to cohesive (0) or no_feature (2)
    final_preds = []
    routed_to_s2 = []
    confidences = []
    decided_by = []

    for i in range(len(df_test)):
        s1_max_p = np.max(s1_probs[i])
        s1_pred  = np.argmax(s1_probs[i])

        if s1_max_p >= args.threshold:
            # High confidence screening
            final_pred = 0 if s1_pred == 0 else 2
            conf = float(s1_max_p)
            dec = f"Stage 1 ({CLASS_NAMES[final_pred]})"
            routed = False
        else:
            # Pass to Stage 2 specialist
            final_pred = int(np.argmax(s2_probs[i]))
            conf = float(s2_probs[i, final_pred])
            dec = "Stage 2 (Specialist)"
            routed = True

        final_preds.append(final_pred)
        routed_to_s2.append(routed)
        confidences.append(conf)
        decided_by.append(dec)

    final_preds = np.array(final_preds)
    routed_to_s2 = np.array(routed_to_s2)

    macro_f1 = f1_score(true_labels, final_preds, average="macro", zero_division=0)
    per_class = f1_score(true_labels, final_preds, average=None, zero_division=0)
    cm = confusion_matrix(true_labels, final_preds, labels=[0, 1, 2, 3])

    print(f"\n=======================================================", flush=True)
    print(f"Model 4 Evaluation (Threshold: {args.threshold})", flush=True)
    print(f"Tiles screened by Stage 1: {(~routed_to_s2).sum()}/{len(df_test)}", flush=True)
    print(f"Tiles evaluated by Stage 2: {routed_to_s2.sum()}/{len(df_test)}", flush=True)
    print(f"Macro-F1: {macro_f1:.4f}", flush=True)
    print(f"Per-class F1: {dict(zip(CLASS_NAMES, per_class.round(4)))}", flush=True)
    print(f"\nClassification Report:", flush=True)
    print(classification_report(true_labels, final_preds, target_names=CLASS_NAMES, zero_division=0), flush=True)
    print(f"=======================================================\n", flush=True)

    plot_confusion_matrix(cm, CLASS_NAMES, f"Model 4 Cascade\nMacro-F1: {macro_f1:.4f}", out_dir / "m4_confusion_matrix.png")
    df_test["predicted"] = [CLASS_NAMES[p] for p in final_preds]
    df_test["confidence"] = np.round(confidences, 6)
    df_test["decided_by"] = decided_by
    df_test["routed_to_s2"] = routed_to_s2
    df_test["s1_prob_cohesive"] = np.round(s1_probs[:, 0], 4)
    df_test["s1_prob_no_feature"] = np.round(s1_probs[:, 1], 4)
    df_test["s2_prob_cohesive"] = np.round(s2_probs[:, 0], 4)
    df_test["s2_prob_dispersive"] = np.round(s2_probs[:, 1], 4)
    df_test["s2_prob_no_feature"] = np.round(s2_probs[:, 2], 4)
    df_test["s2_prob_noncohesive"] = np.round(s2_probs[:, 3], 4)

    # Save full predictions with probabilities
    pred_path = out_dir / "m4_test_predictions.csv"
    df_test.to_csv(pred_path, index=False)
    print(f"Full predictions with confidence saved to: {pred_path}", flush=True)

    # Also save sorted predictions (grouped by class, then confidence descending)
    sorted_df = df_test.sort_values(by="confidence", ascending=True).reset_index(drop=True)
    sorted_path = out_dir / "m4_test_predictions_sorted.csv"
    sorted_df.to_csv(sorted_path, index=False)
    print(f"Sorted predictions (least confident first, 1.000 at bottom) saved to: {sorted_path}", flush=True)


if __name__ == "__main__":
    main()
