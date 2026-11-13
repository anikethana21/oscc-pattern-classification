import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "D:/Project/model4_cascade")

from m4_stage1_model import Stage1WithBackbone
from m4_stage2_model import Stage2Specialist4Class
from m4_dataset import get_val_transform, make_16_crops, CLASS_NAMES, NAMED_COLS


class M4InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, named_mean=None, named_std=None):
        self.df = df.reset_index(drop=True)
        self.val_transform = get_val_transform()
        self.named_mean = named_mean
        self.named_std = named_std
        self.avail_named = all(c in df.columns for c in NAMED_COLS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
        except Exception as e:
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)

        transformed = self.val_transform(image=img_np)
        img_512 = transformed["image"]
        crops_224 = make_16_crops(img_512)
        img_256 = F.interpolate(img_512.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).squeeze(0)

        if self.avail_named:
            named = np.array([row.get(c, 0.0) for c in NAMED_COLS], dtype=np.float32)
            if self.named_mean is not None and self.named_std is not None:
                named = (named - self.named_mean) / (self.named_std + 1e-8)
        else:
            named = np.zeros(len(NAMED_COLS), dtype=np.float32)

        return crops_224, img_256, torch.tensor(named, dtype=torch.float32), idx


def run_m4_confidence_scoring(
    input_csv="D:/Project/hybridpath_model/split_test.csv",
    s1_weights="D:/Project/model4_cascade/weights_s1/stage1_best.pt",
    s2_weights="D:/Project/model4_cascade/weights_s2/stage2_best.pt",
    output_csv="D:/Project/model4_cascade/m4_tile_scores.csv",
    threshold=0.75,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=4
):
    print(f"=== Model 4 Confidence Scoring (Zero Retraining / Untouched Weights) ===")
    print(f"Input: {input_csv}")
    print(f"Stage 1 Weights: {s1_weights}")
    print(f"Stage 2 Weights: {s2_weights}")
    print(f"Threshold: {threshold} | Device: {device}")

    # Load stats
    s1_dir = Path(s1_weights).parent
    nm_file = s1_dir / "named_mean.npy"
    ns_file = s1_dir / "named_std.npy"
    named_mean = np.load(nm_file) if nm_file.exists() else None
    named_std  = np.load(ns_file) if ns_file.exists() else None

    # Load Stage 1
    s1_model = Stage1WithBackbone(device=device).to(device)
    s1_ckpt  = torch.load(s1_weights, map_location=device, weights_only=False)
    s1_model.load_state_dict(s1_ckpt["model_state"])
    s1_model.eval()

    # Load Stage 2
    s2_model = Stage2Specialist4Class(num_classes=4, device=device).to(device)
    s2_ckpt  = torch.load(s2_weights, map_location=device, weights_only=False)
    s2_model.load_state_dict(s2_ckpt["model_state"])
    s2_model.eval()

    df = pd.read_csv(input_csv)
    dataset = M4InferenceDataset(df, named_mean=named_mean, named_std=named_std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    use_amp = (device == "cuda")
    all_s1_probs = []
    all_s2_probs = []

    print(f"Computing predictions on {len(df)} tiles...")
    with torch.no_grad():
        for crops_224, img_256, named_feats, idxs in loader:
            crops_224 = crops_224.to(device)
            img_256 = img_256.to(device)
            named_feats = named_feats.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                s1_logits = s1_model(crops_224, named_feats)
                s2_logits = s2_model(crops_224, img_256)

            p1 = torch.softmax(s1_logits, dim=1).cpu().numpy()
            p2 = torch.softmax(s2_logits, dim=1).cpu().numpy()
            all_s1_probs.append(p1)
            all_s2_probs.append(p2)

    s1_probs = np.concatenate(all_s1_probs, axis=0)  # [N, 2] -> 0: cohesive, 1: no_feature
    s2_probs = np.concatenate(all_s2_probs, axis=0)  # [N, 4] -> cohesive, dispersive, no_feature, noncohesive

    # Cascade Decision + Confidence
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        tile_path = row["path"]
        tile_id = os.path.splitext(os.path.basename(tile_path))[0]
        true_label = row.get("label", row.get("true_label", ""))

        s1_coh_prob = float(s1_probs[i, 0])
        s1_nf_prob  = float(s1_probs[i, 1])
        s1_max_prob = max(s1_coh_prob, s1_nf_prob)
        s1_choice   = "cohesive" if s1_coh_prob >= s1_nf_prob else "no_feature"

        s2_coh = float(s2_probs[i, 0])
        s2_dis = float(s2_probs[i, 1])
        s2_nf  = float(s2_probs[i, 2])
        s2_nc  = float(s2_probs[i, 3])
        s2_pred_idx = int(np.argmax(s2_probs[i]))
        s2_pred_cls = CLASS_NAMES[s2_pred_idx]
        s2_conf     = float(s2_probs[i, s2_pred_idx])

        if s1_max_prob >= threshold:
            pred_class = s1_choice
            confidence = s1_max_prob
            decided_by = f"Stage1 ({s1_choice})"
        else:
            pred_class = s2_pred_cls
            confidence = s2_conf
            decided_by = "Stage2 (Specialist)"

        results.append({
            "tile_id": tile_id,
            "tile_path": tile_path,
            "true_label": true_label,
            "pred_class": pred_class,
            "confidence": round(confidence, 6),
            "decided_by": decided_by,
            "is_correct": (pred_class == true_label) if true_label else "",
            "s1_prob_cohesive": round(s1_coh_prob, 4),
            "s1_prob_no_feature": round(s1_nf_prob, 4),
            "s2_prob_cohesive": round(s2_coh, 4),
            "s2_prob_dispersive": round(s2_dis, 4),
            "s2_prob_no_feature": round(s2_nf, 4),
            "s2_prob_noncohesive": round(s2_nc, 4),
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved confidence scores to: {output_csv}")

    # Create sorted file (by pred_class, then confidence descending)
    sorted_csv = output_csv.replace(".csv", "_sorted.csv")
    sorted_df = out_df.sort_values(by="confidence", ascending=True).reset_index(drop=True)
    sorted_df.to_csv(sorted_csv, index=False)
    print(f"Saved sorted tiles (least confident first, 1.000 at bottom) to: {sorted_csv}")

    # Summary
    print("\n=== Summary by Class (Model 4) ===")
    for cls, grp in sorted_df.groupby("pred_class"):
        corr = grp["is_correct"].sum() if "is_correct" in grp.columns and grp["is_correct"].iloc[0] != "" else "N/A"
        total = len(grp)
        mean_c = grp["confidence"].mean()
        min_c = grp["confidence"].min()
        max_c = grp["confidence"].max()
        print(f"  Class {cls:<15}: {total:>3d} tiles | Acc: {corr}/{total} | Conf: mean={mean_c:.3f}, min={min_c:.3f}, max={max_c:.3f}")

    return sorted_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="D:/Project/hybridpath_model/split_test.csv")
    parser.add_argument("--s1-weights", default="D:/Project/model4_cascade/weights_s1/stage1_best.pt")
    parser.add_argument("--s2-weights", default="D:/Project/model4_cascade/weights_s2/stage2_best.pt")
    parser.add_argument("--output", default="D:/Project/model4_cascade/m4_tile_scores.csv")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    run_m4_confidence_scoring(
        input_csv=args.input,
        s1_weights=args.s1_weights,
        s2_weights=args.s2_weights,
        output_csv=args.output,
        threshold=args.threshold,
        batch_size=args.batch_size
    )
