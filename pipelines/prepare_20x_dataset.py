import os
import sys
import zipfile
import glob
import io
import random
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# Seed for reproducibility
random.seed(42)

OUTPUT_DIR = r"D:\Project\dataset_20x_master"
IMG_SIZE = (512, 512)

# Target counts requested by user
TARGETS = {
    "cohesive": 600,
    "dysplasia": 500,
    "noncohesive": 400,
    "dispersive": 400,
    "normal": 350,
    "no_feature": 500,
    "blank": 100
}

# D4 transformations (7 non-identity operations)
def apply_d4_transform(img, op_idx):
    if op_idx == 0:
        return img.transpose(Image.Transpose.ROTATE_90)
    elif op_idx == 1:
        return img.transpose(Image.Transpose.ROTATE_180)
    elif op_idx == 2:
        return img.transpose(Image.Transpose.ROTATE_270)
    elif op_idx == 3:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif op_idx == 4:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif op_idx == 5:
        # Transpose (rot90 + flip)
        return img.transpose(Image.Transpose.TRANSPOSE)
    elif op_idx == 6:
        # Transverse (rot270 + flip)
        return img.transpose(Image.Transpose.TRANSVERSE)
    return img

OP_NAMES = ["rot90", "rot180", "rot270", "hflip", "vflip", "transpose", "transverse"]

def get_class_from_path(p):
    parts = p.replace("\\", "/").split("/")
    for part in parts:
        plow = part.lower()
        if "dysplasia" in plow:
            return "dysplasia"
        elif "normal" in plow:
            return "normal"
        elif "cohesive" in plow and "non" not in plow:
            return "cohesive"
        elif "non_cohesive" in plow or "noncohesive" in plow:
            return "noncohesive"
        elif "dispersive" in plow:
            return "dispersive"
        elif "no_feature" in plow or "nofeature" in plow:
            return "no_feature"
        elif "blank" in plow:
            return "blank"
    return None

def main():
    print("=== OSCC 20X DATASET IN-MEMORY STREAMING & D4 STAGING ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for c in TARGETS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

    zips = sorted(glob.glob(r"C:\Users\anike\Downloads\*20261111*.zip"))
    print(f"Found {len(zips)} zip archives in Downloads:")
    for z in zips:
        print(f"  - {os.path.basename(z)}")

    # Index all files in all zips
    class_to_files = {c: [] for c in TARGETS.keys()}
    for z in zips:
        with zipfile.ZipFile(z, "r") as zf:
            for n in zf.namelist():
                if n.endswith("/") or n.startswith("__MACOSX"):
                    continue
                c = get_class_from_path(n)
                if c in class_to_files:
                    class_to_files[c].append((z, n))

    print("\nRaw files cataloged per class:")
    for c, files in class_to_files.items():
        print(f"  {c:<12}: {len(files)} raw files (target: {TARGETS[c]})")

    all_staged_records = []

    # Process each class
    for c, target in TARGETS.items():
        files = class_to_files[c]
        random.shuffle(files)
        raw_count = len(files)
        target_count = TARGETS[c]
        print(f"\nProcessing class: {c} (Raw: {raw_count} -> Target: {target_count})...")

        # Open zip cache map
        zip_handles = {}

        def get_image_bytes(zpath, fname):
            if zpath not in zip_handles:
                zip_handles[zpath] = zipfile.ZipFile(zpath, "r")
            return zip_handles[zpath].read(fname)

        if raw_count >= target_count:
            # Subsample raw files
            selected_files = files[:target_count]
            for idx, (zpath, fname) in enumerate(selected_files):
                data = get_image_bytes(zpath, fname)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                
                out_name = f"{c}_{idx:04d}_raw.jpg"
                out_path = os.path.join(OUTPUT_DIR, c, out_name)
                img.save(out_path, format="JPEG", quality=95)

                all_staged_records.append({
                    "tile_id": f"{c}_{idx:04d}",
                    "file_path": out_path,
                    "relative_path": os.path.relpath(out_path, OUTPUT_DIR),
                    "class_name": c,
                    "source_archive": os.path.basename(zpath),
                    "source_file": fname,
                    "augmentation": "none"
                })
                if (idx + 1) % 100 == 0 or (idx + 1) == target_count:
                    print(f"  Saved {idx + 1}/{target_count} raw tiles...")
        else:
            # Keep all raw tiles first
            staged_count = 0
            # Preload and resize raw tiles
            raw_cache = []
            for idx, (zpath, fname) in enumerate(files):
                data = get_image_bytes(zpath, fname)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                
                out_name = f"{c}_{staged_count:04d}_raw.jpg"
                out_path = os.path.join(OUTPUT_DIR, c, out_name)
                img.save(out_path, format="JPEG", quality=95)

                all_staged_records.append({
                    "tile_id": f"{c}_{staged_count:04d}",
                    "file_path": out_path,
                    "relative_path": os.path.relpath(out_path, OUTPUT_DIR),
                    "class_name": c,
                    "source_archive": os.path.basename(zpath),
                    "source_file": fname,
                    "augmentation": "none"
                })
                raw_cache.append((img, zpath, fname, idx))
                staged_count += 1

            needed_aug = target_count - staged_count
            print(f"  Extracted {staged_count} raw tiles. Generating {needed_aug} D4 dihedral rotations/flips...")

            # Generate D4 augmentations in round-robin fashion over raw tiles
            aug_idx = 0
            while staged_count < target_count:
                base_img, zpath, fname, base_orig_idx = raw_cache[aug_idx % len(raw_cache)]
                op_idx = (aug_idx // len(raw_cache)) % 7
                op_name = OP_NAMES[op_idx]

                aug_img = apply_d4_transform(base_img, op_idx)
                out_name = f"{c}_{staged_count:04d}_d4_{op_name}_src{base_orig_idx}.jpg"
                out_path = os.path.join(OUTPUT_DIR, c, out_name)
                aug_img.save(out_path, format="JPEG", quality=95)

                all_staged_records.append({
                    "tile_id": f"{c}_{staged_count:04d}",
                    "file_path": out_path,
                    "relative_path": os.path.relpath(out_path, OUTPUT_DIR),
                    "class_name": c,
                    "source_archive": os.path.basename(zpath),
                    "source_file": fname,
                    "augmentation": f"d4_{op_name}"
                })
                staged_count += 1
                aug_idx += 1

            print(f"  Completed {staged_count}/{target_count} total tiles for {c}.")

        # Close any open zip handles for this class
        for zh in zip_handles.values():
            zh.close()

    df_all = pd.DataFrame(all_staged_records)
    print("\n=== TOTAL DATASET SUMMARY ===")
    print(df_all["class_name"].value_counts())
    print(f"Total Staged Tiles: {len(df_all)}")

    # 1. Manifest for Model 104.4 (7 classes)
    # Stratified 80/20 train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    splits = list(sss.split(df_all, df_all["class_name"]))
    train_idx, val_idx = splits[0]

    df_104 = df_all.copy()
    df_104["split"] = "train"
    df_104.loc[val_idx, "split"] = "val"

    # Assign integer class labels for 104.4
    CLASS_MAP_7 = {
        "cohesive": 0,
        "dysplasia": 1,
        "noncohesive": 2,
        "dispersive": 3,
        "normal": 4,
        "no_feature": 5,
        "blank": 6
    }
    df_104["label"] = df_104["class_name"].map(CLASS_MAP_7)
    manifest_104_path = os.path.join(OUTPUT_DIR, "manifest_104_4.csv")
    df_104.to_csv(manifest_104_path, index=False)
    print(f"\nSaved Model 104.4 Manifest: {manifest_104_path}")
    print("Model 104.4 Split Distribution:")
    print(pd.crosstab(df_104["class_name"], df_104["split"]))

    # 2. Manifest for Model 109.4 (5 biological classes)
    # Exclude no_feature and blank from training, keep them as outlier_test
    bio_classes = ["cohesive", "dysplasia", "noncohesive", "dispersive", "normal"]
    df_bio = df_all[df_all["class_name"].isin(bio_classes)].copy().reset_index(drop=True)
    df_outliers = df_all[~df_all["class_name"].isin(bio_classes)].copy().reset_index(drop=True)

    sss_bio = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    splits_bio = list(sss_bio.split(df_bio, df_bio["class_name"]))
    t_idx, v_idx = splits_bio[0]

    df_bio["split"] = "train"
    df_bio.loc[v_idx, "split"] = "val"

    CLASS_MAP_5 = {
        "cohesive": 0,
        "dysplasia": 1,
        "noncohesive": 2,
        "dispersive": 3,
        "normal": 4
    }
    df_bio["label"] = df_bio["class_name"].map(CLASS_MAP_5)

    df_outliers["split"] = "outlier_eval"
    df_outliers["label"] = -1  # Outlier label

    df_109 = pd.concat([df_bio, df_outliers], ignore_index=True)
    manifest_109_path = os.path.join(OUTPUT_DIR, "manifest_109_4.csv")
    df_109.to_csv(manifest_109_path, index=False)
    print(f"\nSaved Model 109.4 Manifest: {manifest_109_path}")
    print("Model 109.4 Split Distribution:")
    print(pd.crosstab(df_109["class_name"], df_109["split"]))

    print("\nDataset preparation and D4 dihedral augmentation completed successfully!")

if __name__ == "__main__":
    main()

