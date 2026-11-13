import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "D:/Project/model4_cascade")
sys.path.insert(0, "D:/Project/model6_earlyexit")

from m4_dataset import Stage2Dataset, CLASS_NAMES, LABEL_ENC_4
from model6 import EarlyExitModel
from routing import route_batch

# Locked baseline metrics for Model 4 (Stage 2 cascade) on 267-tile test set
M4_METRICS = {
    "macro_f1": 0.8321,
    "accuracy": 0.8801,
    "f1": {
        "cohesive": 0.920,
        "dispersive": 0.770,
        "no_feature": 0.950,
        "noncohesive": 0.690,
    }
}


def evaluate(config_path="D:/Project/model6_earlyexit/config.yaml",
             weights_path=None,
             device="cuda" if torch.cuda.is_available() else "cpu",
             split="test"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if weights_path is None:
        weights_path = os.path.join(cfg["paths"]["save_dir"], "best_m6.pt")

    print(f"=== Model 6 Evaluation ===")
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Weights: {weights_path}")
    print(f"Split: {split}")

    csv_path = cfg["paths"].get(f"{split}_csv")
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV for split '{split}' not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} tiles from {csv_path}")

    # Dataset & Loader
    ds = Stage2Dataset(df, mode="val")
    loader = DataLoader(
        ds,
        batch_size=cfg["training"].get("batch_size", 6),
        shuffle=False,
        num_workers=cfg["dataloader"].get("num_workers", 2),
        pin_memory=cfg["dataloader"].get("pin_memory", True),
    )

    # Load Model
    model = EarlyExitModel(cfg, device=device).to(device)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"[Warning] Weights not found at {weights_path}. Evaluating uninitialized model.")

    model.eval()

    all_preds = []
    all_targets = []
    all_decisions = []

    with torch.no_grad():
        for crops_224, img_256, labels in loader:
            crops_224 = crops_224.to(device)
            img_256 = img_256.to(device)

            outputs = model(crops_224, img_256)
            decisions = route_batch(outputs, cfg["routing"])

            for dec, lbl in zip(decisions, labels):
                all_preds.append(dec["pred_class"])
                all_targets.append(CLASS_NAMES[lbl.item()])
                all_decisions.append(dec)

    # 1. Overall Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, labels=CLASS_NAMES, average=None, zero_division=0)
    f1_dict = {cls: per_class_f1[i] for i, cls in enumerate(CLASS_NAMES)}

    print("\n" + "=" * 65)
    print(f"{'METRIC':<18} | {'MODEL 4 (LOCKED)':<18} | {'MODEL 6 (EARLY-EXIT)':<18}")
    print("-" * 65)
    print(f"{'Macro-F1':<18} | {M4_METRICS['macro_f1']:<18.4f} | {macro_f1:<18.4f}")
    print(f"{'Accuracy':<18} | {M4_METRICS['accuracy']*100:<17.2f}% | {acc*100:<17.2f}%")
    print("-" * 65)
    for cls in CLASS_NAMES:
        m4_f = M4_METRICS["f1"].get(cls, 0.0)
        m6_f = f1_dict.get(cls, 0.0)
        print(f"{'F1 - ' + cls:<18} | {m4_f:<18.4f} | {m6_f:<18.4f}")
    print("=" * 65)

    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
    print("\nConfusion Matrix:")
    print(cm_df)

    # 2. Per-exit coverage and accuracy
    print("\n" + "=" * 65)
    print(f"{'EXIT / DECIDED_BY':<20} | {'COVERAGE (TILES)':<18} | {'ACCURACY WHEN FIRED':<18}")
    print("-" * 65)
    exits = sorted(list(set(d["decided_by"] for d in all_decisions)))
    for ex in exits:
        idxs = [i for i, d in enumerate(all_decisions) if d["decided_by"] == ex]
        cov = len(idxs)
        pct = (cov / len(all_decisions)) * 100
        corr = sum(1 for i in idxs if all_targets[i] == all_preds[i])
        exit_acc = (corr / cov) * 100 if cov > 0 else 0.0
        print(f"{ex:<20} | {cov:>4d} ({pct:>5.1f}%)       | {exit_acc:>6.2f}% ({corr}/{cov})")
    print("=" * 65)

    # 3. Noncohesive predicted as cohesive check
    nc_as_coh = []
    for i, (t, p, d) in enumerate(zip(all_targets, all_preds, all_decisions)):
        if t == "noncohesive" and p == "cohesive":
            nc_as_coh.append({
                "idx": i,
                "confidence": d["confidence"],
                "decided_by": d["decided_by"],
                "p_cohesive": d["p_cohesive"],
                "p_noncohesive": d["p_noncohesive"],
                "c_cohesive": d["c_cohesive"],
                "c_noncohesive": d["c_noncohesive"]
            })

    print(f"\nNoncohesive predicted as Cohesive check: {len(nc_as_coh)} tiles")
    if nc_as_coh:
        for item in nc_as_coh:
            print(f"  Tile #{item['idx']:<3d} | Conf: {item['confidence']:.3f} | Decided By: {item['decided_by']} | "
                  f"p_coh={item['p_cohesive']:.3f}, p_nc={item['p_noncohesive']:.3f}")
    else:
        print("  None! No noncohesive tiles were misclassified as cohesive.")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": f1_dict,
        "confusion_matrix": cm,
        "decisions": all_decisions,
        "nc_as_coh": nc_as_coh,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="D:/Project/model6_earlyexit/config.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    evaluate(args.config, args.weights, args.device, args.split)
