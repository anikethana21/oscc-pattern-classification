"""
Step 4: Predict class labels for new tiles using the trained model.

Usage:
    python predict.py --input "D:\NewSlides\tiles" --out predictions.csv
    python predict.py --input "D:\NewSlides\tiles\some_tile.png" --out predictions.csv

Loads the trained model from model/ and runs inference.
Applies confidence threshold for "needs human" flagging.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_crops(img_path, grid_size=4, crop_size=224):
    """Cut a tile into grid_size x grid_size crops, resize to crop_size."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    cw, ch = w // grid_size, h // grid_size
    crops = []
    for row in range(grid_size):
        for col in range(grid_size):
            box = (col * cw, row * ch, (col + 1) * cw, (row + 1) * ch)
            crop = img.crop(box).resize((crop_size, crop_size), Image.LANCZOS)
            crops.append(crop)
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to a single tile or a folder of tiles")
    ap.add_argument("--model-dir", default="model",
                    help="Path to saved model directory")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Confidence threshold for 'needs human' flag")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # ---- Load config and model ----
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    classes = np.array(config["classes"])
    num_classes = config["num_classes"]
    in_dim = config["in_dim"]

    head = LinearHead(in_dim, num_classes)
    head.load_state_dict(
        torch.load(model_dir / "linear_head.pt", map_location=args.device,
                    weights_only=True)
    )
    head = head.to(args.device)
    head.eval()

    # ---- Load Phikon backbone ----
    print("Loading Phikon backbone...")
    processor = AutoImageProcessor.from_pretrained(config["backbone"])
    backbone = ViTModel.from_pretrained(
        config["backbone"], add_pooling_layer=False
    )
    backbone = backbone.to(args.device)
    backbone.eval()
    print("Ready.")

    # ---- Find tiles ----
    input_path = Path(args.input)
    if input_path.is_file():
        tile_paths = [input_path]
    else:
        tile_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMG_EXT
        )
    print(f"Found {len(tile_paths)} tiles to predict.")

    # ---- Predict ----
    results = []
    with torch.no_grad():
        for p in tqdm(tile_paths, desc="Predicting"):
            try:
                crops = extract_crops(str(p))
                inputs = processor(images=crops, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

                outputs = backbone(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]

                mean_pool = cls_emb.mean(dim=0)
                max_pool = cls_emb.max(dim=0).values
                combined = torch.cat([mean_pool, max_pool]).unsqueeze(0)

                logits = head(combined)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = int(probs.argmax())

                row = {
                    "path": str(p),
                    "predicted": classes[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "needs_human": bool(probs[pred_idx] < args.threshold),
                }
                for i, cls_name in enumerate(classes):
                    row[f"prob_{cls_name}"] = float(probs[i])
                results.append(row)

            except Exception as e:
                print(f"  Error on {p.name}: {e}")
                results.append({
                    "path": str(p), "predicted": "ERROR",
                    "confidence": 0.0, "needs_human": True,
                })

    # ---- Save and report ----
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"Total tiles: {len(df)}")
    errors = (df["predicted"] == "ERROR").sum()
    if errors:
        print(f"Errors: {errors}")
    valid = df[df["predicted"] != "ERROR"]
    print(f"\nPredicted class distribution:")
    print(valid["predicted"].value_counts().to_string())
    print(f"\nConfidence stats:")
    print(f"  Mean: {valid['confidence'].mean():.3f}")
    print(f"  Min:  {valid['confidence'].min():.3f}")
    print(f"  Median: {valid['confidence'].median():.3f}")
    needs = valid["needs_human"].sum()
    print(f"\nNeeds human review (< {args.threshold:.0%}): "
          f"{needs} ({needs/len(valid):.1%})")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()