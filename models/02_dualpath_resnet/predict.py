"""
Step 4 (v2): Predict new tiles using trained model with named features.

Usage:
    cd D:\Project\second_model
    python predict.py --input "D:\Path\To\New\Tiles" --out predictions.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from scipy.spatial import KDTree
from scipy import ndimage
from tqdm import tqdm

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def extract_crops(img_path, grid_size=4, crop_size=224):
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


def extract_morphological_features(img_bgr):
    """Nuclei segmentation via color thresholding + morphological analysis."""
    h_img, w_img = img_bgr.shape[:2]
    total_pixels = h_img * w_img

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    inv = 255 - gray
    _, otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hue_mask = ((hsv[:, :, 0] >= 100) & (hsv[:, :, 0] <= 160)).astype(np.uint8) * 255
    sat_mask = (hsv[:, :, 1] >= 30).astype(np.uint8) * 255
    nuclei_mask = cv2.bitwise_and(otsu, cv2.bitwise_or(hue_mask, sat_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, kernel)
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_CLOSE, kernel)

    labeled, num_nuclei = ndimage.label(nuclei_mask)

    if num_nuclei == 0:
        return np.zeros(8, dtype=np.float32)

    areas, perimeters, centroids, eccentricities = [], [], [], []

    for i in range(1, num_nuclei + 1):
        component = (labeled == i).astype(np.uint8)
        area = component.sum()
        if area < 20 or area > 5000:
            continue
        areas.append(area)

        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cnt = contours[0]
            perimeters.append(cv2.arcLength(cnt, True))
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                centroids.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
            if len(cnt) >= 5:
                _, (ma, MA), _ = cv2.fitEllipse(cnt)
                if MA > 0:
                    eccentricities.append(
                        np.sqrt(1 - (min(ma, MA) / max(ma, MA)) ** 2)
                    )

    if len(areas) == 0:
        return np.zeros(8, dtype=np.float32)

    areas = np.array(areas, dtype=np.float32)
    perimeters = np.array(perimeters, dtype=np.float32)

    circularity = []
    for a, p in zip(areas, perimeters):
        if p > 0:
            circularity.append(4 * np.pi * a / (p * p))
    circularity = np.array(circularity) if circularity else np.array([0.0])

    if len(centroids) >= 2:
        tree = KDTree(np.array(centroids))
        dists, _ = tree.query(np.array(centroids), k=2)
        nn_dists = dists[:, 1]
        mean_nn, std_nn = nn_dists.mean(), nn_dists.std()
    else:
        mean_nn, std_nn = 0.0, 0.0

    return np.array([
        len(areas), len(areas) / total_pixels * 1e6,
        areas.mean(), areas.std() if len(areas) > 1 else 0.0,
        circularity.mean(),
        np.mean(eccentricities) if eccentricities else 0.0,
        mean_nn, std_nn,
    ], dtype=np.float32)


def extract_color_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    red1 = (hsv[:, :, 0] <= 10) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 50)
    red2 = (hsv[:, :, 0] >= 160) & (hsv[:, :, 1] >= 50) & (hsv[:, :, 2] >= 50)
    red_frac = (red1.sum() + red2.sum()) / total

    pink = ((hsv[:, :, 0] <= 20) & (hsv[:, :, 1] >= 20) &
            (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 150))
    pink_frac = pink.sum() / total

    return np.array([red_frac, pink_frac], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to a tile or folder of tiles")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    classes = np.array(config["classes"])
    head = LinearHead(config["in_dim"], config["num_classes"])
    head.load_state_dict(
        torch.load(model_dir / "linear_head.pt", map_location=args.device,
                    weights_only=True)
    )
    head = head.to(args.device)
    head.eval()

    # Load normalization stats for named features
    named_mean = np.load(model_dir.parent / "named_mean.npy")
    named_std = np.load(model_dir.parent / "named_std.npy")

    print("Loading Phikon backbone...")
    processor = AutoImageProcessor.from_pretrained(config["backbone"])
    backbone = ViTModel.from_pretrained(
        config["backbone"], add_pooling_layer=False
    )
    backbone = backbone.to(args.device)
    backbone.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        tile_paths = [input_path]
    else:
        tile_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMG_EXT
        )
    print(f"Found {len(tile_paths)} tiles to predict.")

    results = []
    with torch.no_grad():
        for p in tqdm(tile_paths, desc="Predicting"):
            try:
                # Deep features
                crops = extract_crops(str(p))
                inputs = processor(images=crops, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                outputs = backbone(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                mean_pool = cls_emb.mean(dim=0)
                max_pool = cls_emb.max(dim=0).values
                deep_vec = torch.cat([mean_pool, max_pool]).cpu().numpy()

                # Named features
                img_bgr = cv2.imread(str(p))
                morph = extract_morphological_features(img_bgr)
                color = extract_color_features(img_bgr)
                named_vec = np.concatenate([morph, color])
                named_normed = (named_vec - named_mean.flatten()) / named_std.flatten()

                combined = np.concatenate([deep_vec, named_normed])
                combined_t = torch.tensor(
                    combined, dtype=torch.float32
                ).unsqueeze(0).to(args.device)

                logits = head(combined_t)
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

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)

    print(f"\n=== PREDICTION SUMMARY (v2) ===")
    print(f"Total tiles: {len(df)}")
    valid = df[df["predicted"] != "ERROR"]
    print(f"\nPredicted class distribution:")
    print(valid["predicted"].value_counts().to_string())
    needs = valid["needs_human"].sum()
    print(f"\nNeeds human review (< {args.threshold:.0%}): "
          f"{needs} ({needs/len(valid):.1%})")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()