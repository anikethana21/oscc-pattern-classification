import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m5_eval.py
----------
Evaluate Model 5 on the locked test set and plot confusion matrix.
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

from m5_model import Model5LoopNetwork
from m5_dataset import OSCCLoopDataset, CLASS_NAMES


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
    ap.add_argument("--weights",    default="model5_loop/weights/model5_best.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir",    default="model5_loop")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Model5LoopNetwork(device=device).to(device)
    ckpt  = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df_test = pd.read_csv(args.test_csv)
    print(f"Locked Test Set: {len(df_test)} tiles", flush=True)

    test_ds = OSCCLoopDataset(df_test, mode="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for crops, img_256, labels in test_loader:
            crops, img_256 = crops.to(device), img_256.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                final_logits, _ = model(crops, img_256)
            preds = final_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    print(f"\n=======================================================", flush=True)
    print(f"Model 5 (Looping Network) Evaluation Results", flush=True)
    print(f"Macro-F1: {macro_f1:.4f}", flush=True)
    print(f"Per-class F1: {dict(zip(CLASS_NAMES, per_class.round(4)))}", flush=True)
    print(f"\nClassification Report:", flush=True)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0), flush=True)
    print(f"=======================================================\n", flush=True)

    plot_confusion_matrix(cm, CLASS_NAMES, f"Model 5 Looping Network\nMacro-F1: {macro_f1:.4f}", out_dir / "m5_confusion_matrix.png")
    df_test["predicted"] = [CLASS_NAMES[p] for p in all_preds]
    df_test.to_csv(out_dir / "m5_test_predictions.csv", index=False)
    print(f"Results and confusion matrix saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
