"""
Step 2: Extract Phikon embeddings for all tiles.

Usage:
    python extract_features.py --manifest manifest.csv --out-dir .

Each 2048x2048 tile -> 4x4 grid -> 16 crops of 224x224
-> frozen Phikon (ViT-B) -> 768-dim CLS token per crop
-> mean + max pool -> 1,536-dim vector per tile

Outputs:
    embeddings.npy         (N x 1536 float32 array)
    embedding_labels.csv   (matching metadata for each row)
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from tqdm import tqdm


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
    ap.add_argument("--manifest", default="manifest.csv",
                    help="Path to manifest.csv from build_manifest.py")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to save embeddings.npy and embedding_labels.csv")
    ap.add_argument("--label-set", default=None,
                    help="Filter to a specific label_set (e.g. set0). Default: use all.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Number of crops per forward pass (16 = one full tile)")
    ap.add_argument("--device", default=None,
                    help="Device (auto-detected if not set)")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # ---- Load manifest ----
    df = pd.read_csv(args.manifest)
    print(f"Manifest has {len(df)} rows")

    # Remove flagged duplicates
    if "dup_flag" in df.columns:
        before = len(df)
        df = df[df["dup_flag"] == False].reset_index(drop=True)
        print(f"Removed {before - len(df)} duplicate-flagged rows -> {len(df)} remaining")

    # Optional: filter to one label set
    if args.label_set:
        df = df[df["label_set"] == args.label_set].reset_index(drop=True)
        print(f"Filtered to {args.label_set}: {len(df)} tiles")

    # ---- Load Phikon ----
    print("Loading Phikon model (owkin/phikon)...")
    processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = model.to(args.device)
    model.eval()
    print("Phikon loaded and ready.")

    # ---- Extract embeddings ----
    embeddings = []
    valid_indices = []
    errors = []

    with torch.no_grad():
        for idx in tqdm(range(len(df)), desc="Extracting embeddings"):
            row = df.iloc[idx]
            try:
                crops = extract_crops(row["path"])

                # Process all 16 crops in one forward pass
                inputs = processor(images=crops, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                # CLS token for each crop: shape [16, 768]
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                # Mean and max pooling across crops
                mean_pool = cls_embeddings.mean(dim=0)       # [768]
                max_pool = cls_embeddings.max(dim=0).values   # [768]
                combined = torch.cat([mean_pool, max_pool])   # [1536]

                embeddings.append(combined.cpu().numpy())
                valid_indices.append(idx)

            except Exception as e:
                errors.append((row["path"], str(e)))
                continue

            # Progress checkpoint every 500 tiles
            if (idx + 1) % 500 == 0:
                print(f"  [{idx+1}/{len(df)}] processed, {len(errors)} errors so far")

    # ---- Save ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    embeddings_arr = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "embeddings.npy", embeddings_arr)

    labels_df = df.iloc[valid_indices].reset_index(drop=True)
    labels_df.to_csv(out_dir / "embedding_labels.csv", index=False)

    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Tiles processed: {len(embeddings)}/{len(df)}")
    print(f"Embedding shape: {embeddings_arr.shape}")
    print(f"Saved to: {out_dir / 'embeddings.npy'}")
    print(f"Labels to: {out_dir / 'embedding_labels.csv'}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
    print(f"\nDone. Next step: python train.py")


if __name__ == "__main__":
    main()