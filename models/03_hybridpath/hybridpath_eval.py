import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
hybridpath_eval.py
------------------
Evaluate best HybridPath model on the locked test set.
Prints full classification report + saves confusion matrix PNG.

Usage:
    python hybridpath_eval.py --test split_test.csv --model-dir weights
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score
)

from hybridpath_model import HybridPath
from hybridpath_dataset import OSCCDataset

CLASS_NAMES = ["cohesive", "dispersive", "no_feature", "noncohesive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",       default="split_test.csv")
    ap.add_argument("--model-dir",  default="weights")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers",    type=int, default=0)
    ap.add_argument("--device",     default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device    = args.device
    model_dir = Path(args.model_dir)

    # -- Load config ---------------------------------------------------------
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"Loaded Config: {cfg}", flush=True)
    else:
        cfg = {}

    # -- Load model ----------------------------------------------------------
    print(f"\nLoading HybridPath checkpoint from {model_dir}/hybridpath_best.pt ...", flush=True)
    model = HybridPath(num_classes=4, unfreeze_last_n=3, device=device).to(device)
    ckpt  = torch.load(model_dir / "hybridpath_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val macro-F1={ckpt.get('best_f1', 0):.4f})", flush=True)

    # -- Test data -----------------------------------------------------------
    df_test  = pd.read_csv(args.test)
    test_ds  = OSCCDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    print(f"\nTest tiles: {len(df_test)}", flush=True)
    print(f"Class distribution:\n{df_test['label'].value_counts().to_string()}", flush=True)

    # -- Inference with AMP --------------------------------------------------
    all_preds, all_labels, all_probs = [], [], []
    use_amp = (device == "cuda")

    with torch.no_grad():
        for crops, img_256, labels in test_loader:
            crops   = crops.to(device)
            img_256 = img_256.to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(crops, img_256)
            else:
                logits = model(crops, img_256)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # -- Metrics -------------------------------------------------------------
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)

    print(f"\n{'='*65}", flush=True)
    print(f"LOCKED TEST SET EVALUATION RESULTS", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"Macro-F1          : {macro_f1:.4f}", flush=True)
    print(f"Balanced Accuracy : {bal_acc:.4f}", flush=True)
    print(f"\nClassification Report:\n", flush=True)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES,
                                digits=4, zero_division=0), flush=True)

    # -- Confusion Matrix ----------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(f"HybridPath Test Confusion Matrix\n(Macro-F1 = {macro_f1:.4f}, Bal-Acc = {bal_acc:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = model_dir / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot: {cm_path}", flush=True)

    # -- Save predictions CSV ------------------------------------------------
    df_out = df_test.copy()
    df_out["predicted_class"] = [CLASS_NAMES[p] for p in all_preds]
    for i, c in enumerate(CLASS_NAMES):
        df_out[f"prob_{c}"] = all_probs[:, i].round(4)
    df_out["correct"] = (df_out["label"] == df_out["predicted_class"])

    preds_path = model_dir / "test_predictions.csv"
    df_out.to_csv(preds_path, index=False)
    print(f"Saved test predictions: {preds_path}", flush=True)
    print(f"\nAccuracy: {df_out['correct'].mean()*100:.2f}%\n", flush=True)


if __name__ == "__main__":
    main()
