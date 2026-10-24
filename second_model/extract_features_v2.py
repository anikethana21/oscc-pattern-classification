"""
Step 2 (v2): Extract Phikon embeddings + named morphological and color features.

Usage:
    python extract_features_v2.py --manifest ..\manifest.csv --out-dir .

Pipeline per tile:
    1. Phikon (frozen ViT-B): 4x4 crops -> mean+max pool -> 1,536-dim deep features
       (Reuses pre-computed deep embeddings if --deep-embeddings exists)
    2. Fast OpenCV nuclei segmentation -> 8 morphological features
    3. OpenCV color analysis -> 2 color features
    4. Concatenate: 1,536 + 10 = 1,546-dim vector

Outputs:
    embeddings_v2.npy         (N x 1546 float32 array)
    embedding_labels_v2.csv   (matching metadata + named feature columns)
"""
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from scipy.spatial import KDTree
from tqdm import tqdm

NAMED_FEATURES = [
    "nuclei_count", "nuclei_density",
    "mean_nucleus_area", "std_nucleus_area",
    "mean_circularity", "mean_eccentricity",
    "mean_nn_distance", "std_nn_distance",
    "red_fraction", "pink_fraction",
]


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

    contours, _ = cv2.findContours(
        nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros(8, dtype=np.float32)

    areas = []
    perimeters = []
    centroids = []
    eccentricities = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 5000:
            continue

        areas.append(area)
        peri = cv2.arcLength(cnt, True)
        perimeters.append(peri)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append([cx, cy])

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
        centroids_arr = np.array(centroids)
        tree = KDTree(centroids_arr)
        dists, _ = tree.query(centroids_arr, k=2)
        nn_dists = dists[:, 1]
        mean_nn = nn_dists.mean()
        std_nn = nn_dists.std()
    else:
        mean_nn = 0.0
        std_nn = 0.0

    features = np.array([
        len(areas),
        len(areas) / total_pixels * 1e6,
        areas.mean(),
        areas.std() if len(areas) > 1 else 0.0,
        circularity.mean(),
        (np.mean(eccentricities) if eccentricities else 0.0),
        mean_nn,
        std_nn,
    ], dtype=np.float32)

    return features


def extract_color_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    red_mask1 = (
        (hsv[:, :, 0] <= 10) &
        (hsv[:, :, 1] >= 50) &
        (hsv[:, :, 2] >= 50)
    )
    red_mask2 = (
        (hsv[:, :, 0] >= 160) &
        (hsv[:, :, 1] >= 50) &
        (hsv[:, :, 2] >= 50)
    )
    red_fraction = (red_mask1.sum() + red_mask2.sum()) / total

    pink_mask = (
        (hsv[:, :, 0] <= 20) &
        (hsv[:, :, 1] >= 20) &
        (hsv[:, :, 1] <= 100) &
        (hsv[:, :, 2] >= 150)
    )
    pink_fraction = pink_mask.sum() / total

    return np.array([red_fraction, pink_fraction], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"..\manifest.csv",
                    help="Path to manifest.csv from build_manifest.py")
    ap.add_argument("--deep-embeddings", default=r"..\embeddings.npy",
                    help="Precomputed deep embeddings (from Model 1)")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to save outputs")
    ap.add_argument("--label-set", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    df = pd.read_csv(args.manifest)
    print(f"Manifest has {len(df)} rows")

    if "dup_flag" in df.columns:
        before = len(df)
        df = df[df["dup_flag"] == False].reset_index(drop=True)
        print(f"Removed {before - len(df)} duplicates -> {len(df)} remaining")

    if args.label_set:
        df = df[df["label_set"] == args.label_set].reset_index(drop=True)
        print(f"Filtered to {args.label_set}: {len(df)} tiles")

    compute_deep = True
    deep_arr = None
    if args.deep_embeddings and Path(args.deep_embeddings).exists():
        loaded_deep = np.load(args.deep_embeddings)
        if len(loaded_deep) == len(df):
            print(f"Loaded existing deep embeddings from {args.deep_embeddings}")
            deep_arr = loaded_deep
            compute_deep = False
        else:
            print(f"Warning: {args.deep_embeddings} length {len(loaded_deep)} does not match manifest length {len(df)}. Recomputing deep.")

    if compute_deep:
        print("Loading Phikon model (owkin/phikon)...")
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        model = model.to(args.device)
        model.eval()
        print("Phikon loaded.")

    all_deep = []
    all_named = []
    valid_indices = []
    errors = []

    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Extracting features"):
            row = df.iloc[idx]
            try:
                if compute_deep:
                    crops = extract_crops(row["path"])
                    inputs = processor(images=crops, return_tensors="pt")
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                    mean_pool = cls_emb.mean(dim=0)
                    max_pool = cls_emb.max(dim=0).values
                    deep_vec = torch.cat([mean_pool, max_pool]).cpu().numpy()
                    all_deep.append(deep_vec)

                img_bgr = cv2.imread(row["path"])
                if img_bgr is None:
                    raise ValueError(f"cv2 could not read {row['path']}")

                morph_feats = extract_morphological_features(img_bgr)
                color_feats = extract_color_features(img_bgr)
                named_vec = np.concatenate([morph_feats, color_feats])

                all_named.append(named_vec)
                valid_indices.append(idx)

            except Exception as e:
                errors.append((row["path"], str(e)))
                continue

            if (idx + 1) % 500 == 0:
                print(f"  [{idx+1}/{len(df)}] processed, {len(errors)} errors so far")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    if compute_deep:
        deep_arr = np.stack(all_deep).astype(np.float32)
    else:
        deep_arr = deep_arr[valid_indices].astype(np.float32)

    named_arr = np.stack(all_named).astype(np.float32)

    named_mean = named_arr.mean(axis=0, keepdims=True)
    named_std = named_arr.std(axis=0, keepdims=True) + 1e-8
    named_normed = (named_arr - named_mean) / named_std

    np.save(out_dir / "named_mean.npy", named_mean)
    np.save(out_dir / "named_std.npy", named_std)

    combined = np.concatenate([deep_arr, named_normed], axis=1).astype(np.float32)
    np.save(out_dir / "embeddings_v2.npy", combined)

    labels_df = df.iloc[valid_indices].reset_index(drop=True)
    for i, name in enumerate(NAMED_FEATURES):
        labels_df[name] = named_arr[:, i]
    labels_df.to_csv(out_dir / "embedding_labels_v2.csv", index=False)

    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Tiles processed: {len(combined)}/{len(df)}")
    print(f"Deep features shape:  {deep_arr.shape}")
    print(f"Named features shape: {named_arr.shape}")
    print(f"Combined shape:       {combined.shape}")

    print(f"\nNamed feature statistics (raw, before normalization):")
    for i, name in enumerate(NAMED_FEATURES):
        vals = named_arr[:, i]
        print(f"  {name:>25}: mean={vals.mean():.2f}  std={vals.std():.2f}  min={vals.min():.2f}  max={vals.max():.2f}")

    print(f"\nSaved to: {out_dir / 'embeddings_v2.npy'}")
    print(f"Labels to: {out_dir / 'embedding_labels_v2.csv'}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
    print(f"\nNext step: python train.py")

if __name__ == "__main__":
    main()
