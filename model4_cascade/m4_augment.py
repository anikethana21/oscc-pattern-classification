import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
m4_augment.py
-------------
Scales each class in split_train.csv to ~2500 tiles by saving
augmented image copies to disk, then writes a new CSV for Model 4.

Run ONCE before training Model 4:
  cd D:\Project
  venv\Scripts\python model4_cascade\m4_augment.py
"""

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

TARGET_PER_CLASS = 2500
AUG_DIR = Path("D:/Project/model4_cascade/augmented_tiles")

LABEL_ENC = {
    "cohesive":    0,
    "dispersive":  1,
    "no_feature":  2,
    "noncohesive": 3,
}
MINORITY_CLASSES = {"dispersive", "noncohesive"}


def _heavy_aug():
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.Rotate(limit=20, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(size=(512, 512), scale=(0.70, 1.0),
                            ratio=(0.85, 1.15), p=0.5),
        A.ElasticTransform(alpha=80, sigma=8, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25,
                             val_shift_limit=18, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.18,
                                   contrast_limit=0.15, p=0.6),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Resize(height=512, width=512),
    ])


def _light_aug():
    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                             val_shift_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.10,
                                   contrast_limit=0.08, p=0.4),
        A.Resize(height=512, width=512),
    ])


def _get_aug(label: str):
    return _heavy_aug() if label in MINORITY_CLASSES else _light_aug()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",  default="D:/Project/hybridpath_model/split_train.csv")
    ap.add_argument("--out-csv",    default="D:/Project/model4_cascade/m4_train.csv")
    ap.add_argument("--aug-dir",    default=str(AUG_DIR))
    ap.add_argument("--target",     type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    rng     = np.random.default_rng(args.seed)
    aug_dir = Path(args.aug_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    print(f"Original training set: {len(df)} tiles")
    print(df["label"].value_counts().to_string())
    print()

    out_rows = []

    for label in LABEL_ENC:
        cls_df = df[df["label"] == label].reset_index(drop=True)
        n_orig  = len(cls_df)
        n_need  = args.target - n_orig

        if n_need < 0:
            cls_df = cls_df.sample(args.target, random_state=args.seed)
            n_need = 0

        for _, row in cls_df.iterrows():
            out_rows.append({"path": row["path"], "label": label,
                             "slide": row.get("slide", ""),
                             "patient": row.get("patient", ""),
                             "aug": False})

        if n_need == 0:
            print(f"  {label}: {len(cls_df)} tiles (already >= target)")
            continue

        print(f"  {label}: {n_orig} originals  generating {n_need} augmented copies...")
        cls_aug_dir = aug_dir / label
        cls_aug_dir.mkdir(exist_ok=True)

        aug_fn  = _get_aug(label)
        paths   = cls_df["path"].tolist()
        slides  = cls_df["slide"].tolist() if "slide" in cls_df.columns else [""] * n_orig
        pats    = cls_df["patient"].tolist() if "patient" in cls_df.columns else [""] * n_orig

        for i in tqdm(range(n_need), desc=f"  aug {label}", ncols=80):
            src_idx  = int(rng.integers(0, n_orig))
            src_path = paths[src_idx]
            slide    = slides[src_idx]
            patient  = pats[src_idx]

            try:
                img_np = np.array(Image.open(src_path).convert("RGB"))
            except Exception as e:
                print(f"    [WARN] Cannot open {src_path}: {e}")
                continue

            result  = aug_fn(image=img_np)
            aug_img = Image.fromarray(result["image"])

            stem     = Path(src_path).stem
            out_name = f"{stem}_aug{i:05d}.png"
            out_path = cls_aug_dir / out_name
            aug_img.save(str(out_path))

            out_rows.append({"path": str(out_path), "label": label,
                             "slide": slide, "patient": patient, "aug": True})

        total_cls = n_orig + n_need
        print(f"    -> {total_cls} tiles for {label}")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\nFinal augmented dataset: {len(out_df)} tiles")
    print(out_df["label"].value_counts().to_string())
    print(f"\nSaved to: {args.out_csv}")
    print(f"Augmented images in: {args.aug_dir}")


if __name__ == "__main__":
    main()
