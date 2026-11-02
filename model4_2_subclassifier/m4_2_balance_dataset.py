"""
m4_2_balance_dataset.py
-----------------------
Utility to prepare balanced dataset from Seniors' folder.
Implements the 15% Controlled Augmentation Rule:
  1. Finds N_max (the count of the largest majority class).
  2. For any minority class with count N_c < 0.85 * N_max,
     augments ONLY until N_target = round(0.85 * N_max).
  3. No blind runaway multiplication. Keeps classes tight and balanced.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A


def augment_image(img_np, transform):
    return transform(image=img_np)["image"]


def get_aug_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.4),
    ])


def prepare_balanced_seniors_data(input_dir: str, output_csv: str = "D:/Project/model4_2_subclassifier/seniors_balanced_train.csv"):
    print(f"\n=== Scanning Seniors' Dataset: {input_dir} ===")
    
    # 1. Discover subfolders (the 4 classes)
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    classes.sort()
    print(f"Found {len(classes)} classes: {classes}")

    class_files = {}
    for cls in classes:
        cls_folder = os.path.join(input_dir, cls)
        # Search for png, jpg, tif
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(cls_folder, "**", ext), recursive=True))
        class_files[cls] = files
        print(f"  Class '{cls}': {len(files)} raw images")

    # 2. Compute majority class and 15% rule target
    counts = {cls: len(files) for cls, files in class_files.items()}
    n_max = max(counts.values())
    majority_class = [cls for cls, cnt in counts.items() if cnt == n_max][0]
    target_count = int(round(0.85 * n_max))

    print(f"\n--- 15% Controlled Augmentation Calculation ---")
    print(f"Majority Class: '{majority_class}' with {n_max} images.")
    print(f"15% Tolerance Target for minority classes: {target_count} images (85% of {n_max}).")

    rows = []
    aug_pipe = get_aug_pipeline()
    aug_out_dir = "D:/Project/model4_2_subclassifier/augmented_samples"
    os.makedirs(aug_out_dir, exist_ok=True)

    for cls, files in class_files.items():
        n_orig = len(files)
        # Add original files
        for f in files:
            rows.append({"path": os.path.abspath(f), "label": cls, "is_augmented": False})

        # Augment if needed to reach target_count
        if n_orig < target_count:
            needed = target_count - n_orig
            print(f"  Class '{cls}': {n_orig} original -> adding {needed} augmented images to reach {target_count}...")
            
            aug_idx = 0
            while aug_idx < needed:
                src_path = files[aug_idx % n_orig]
                try:
                    img = Image.open(src_path).convert("RGB")
                    img_np = np.array(img)
                    aug_np = augment_image(img_np, aug_pipe)
                    
                    base_name = os.path.splitext(os.path.basename(src_path))[0]
                    aug_filename = f"aug_{cls}_{base_name}_{aug_idx}.png"
                    aug_path = os.path.join(aug_out_dir, aug_filename)
                    Image.fromarray(aug_np).save(aug_path)

                    rows.append({"path": os.path.abspath(aug_path), "label": cls, "is_augmented": True})
                    aug_idx += 1
                except Exception as e:
                    print(f"Error augmenting {src_path}: {e}")
                    break
        else:
            print(f"  Class '{cls}': {n_orig} images (already within 15% of majority, no augmentation needed).")

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[+] Successfully saved balanced dataset to: {output_csv}")
    print(f"Total tiles in dataset: {len(df)}")
    print("Class distribution:\n", df["label"].value_counts().to_string())

    # Save classes JSON
    classes_json = "D:/Project/model4_2_subclassifier/classes.json"
    with open(classes_json, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"[+] Saved classes list to: {classes_json}")

    return df, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to Seniors dataset directory")
    parser.add_argument("--output-csv", default="D:/Project/model4_2_subclassifier/seniors_balanced_train.csv")
    args = parser.parse_args()

    prepare_balanced_seniors_data(args.input_dir, args.output_csv)

