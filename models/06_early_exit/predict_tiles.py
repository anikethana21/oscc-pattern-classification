import os
import sys
import argparse
import glob
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "D:/Project/model4_cascade")
sys.path.insert(0, "D:/Project/model6_earlyexit")

from m4_dataset import get_val_transform, make_16_crops, CLASS_NAMES, LABEL_ENC_4
from model6 import EarlyExitModel
from routing import route_batch


class TileInferenceDataset(Dataset):
    """Dataset for inference on arbitrary tile paths or DataFrame."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.val_transform = get_val_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
        except Exception as e:
            print(f"[Warning] Error opening {img_path}: {e}")
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)

        transformed = self.val_transform(image=img_np)
        img_512 = transformed["image"]
        crops_224 = make_16_crops(img_512)
        img_256 = F.interpolate(img_512.unsqueeze(0), size=(256, 256),
                                mode="bilinear", align_corners=False).squeeze(0)
        return crops_224, img_256, idx


def predict(input_source,
            output_csv="D:/Project/model6_earlyexit/tile_scores.csv",
            config_path="D:/Project/model6_earlyexit/config.yaml",
            weights_path=None,
            split_label="unlabeled",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=8):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if weights_path is None:
        weights_path = os.path.join(cfg["paths"]["save_dir"], "best_m6.pt")

    print(f"=== Model 6 Tile Prediction ===")
    print(f"Input: {input_source}")
    print(f"Output: {output_csv}")
    print(f"Weights: {weights_path}")
    print(f"Device: {device}")

    # Build DataFrame from input
    if os.path.isdir(input_source):
        # Folder of images
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_source, "**", ext), recursive=True))
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images in {input_source}")
        df = pd.DataFrame({
            "tile_id": [os.path.splitext(os.path.basename(p))[0] for p in image_paths],
            "path": image_paths,
            "split": split_label,
            "true_label": ""
        })
    elif os.path.isfile(input_source):
        df = pd.read_csv(input_source)
        if "tile_id" not in df.columns:
            if "path" in df.columns:
                df["tile_id"] = df["path"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
            else:
                df["tile_id"] = [f"tile_{i}" for i in range(len(df))]
        if "split" not in df.columns:
            df["split"] = split_label
        if "true_label" not in df.columns:
            if "label" in df.columns:
                df["true_label"] = df["label"]
            else:
                df["true_label"] = ""
    else:
        raise ValueError(f"Input source not found: {input_source}")

    if len(df) == 0:
        print("No tiles to process.")
        return

    # Dataset & Loader
    dataset = TileInferenceDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = EarlyExitModel(cfg, device=device).to(device)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"[Warning] Weights not found at {weights_path}. Running with current initialization.")

    model.eval()

    results = []
    print(f"Running inference on {len(df)} tiles...")

    with torch.no_grad():
        for crops_224, img_256, batch_indices in loader:
            crops_224 = crops_224.to(device)
            img_256 = img_256.to(device)

            outputs = model(crops_224, img_256)
            decisions = route_batch(outputs, cfg["routing"])

            for dec, idx_t in zip(decisions, batch_indices):
                row = df.iloc[idx_t.item()]
                res_row = {
                    "tile_id": row["tile_id"],
                    "tile_path": row["path"],
                    "split": row["split"],
                    "true_label": row["true_label"],
                    "pred_class": dec["pred_class"],
                    "confidence": round(dec["confidence"], 6),
                    "decided_by": dec["decided_by"],
                    "p_no_feature": round(dec["p_no_feature"], 6),
                    "p_dispersive": round(dec["p_dispersive"], 6),
                    "p_cohesive": round(dec["p_cohesive"], 6),
                    "p_noncohesive": round(dec["p_noncohesive"], 6),
                    "c_cohesive": round(dec["c_cohesive"], 6),
                    "c_dispersive": round(dec["c_dispersive"], 6),
                    "c_no_feature": round(dec["c_no_feature"], 6),
                    "c_noncohesive": round(dec["c_noncohesive"], 6),
                }
                results.append(res_row)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(out_df)} predictions to {output_csv}")
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tile directory or CSV containing 'path' column")
    parser.add_argument("--output", default="D:/Project/model6_earlyexit/tile_scores.csv")
    parser.add_argument("--config", default="D:/Project/model6_earlyexit/config.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--split", default="unlabeled")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    predict(args.input, args.output, args.config, args.weights, args.split, args.device, args.batch_size)
