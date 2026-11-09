import os
import glob
import random
import shutil
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold

SOURCE_DIR = r"D:\new_whip_432"
DEST_DIR = r"D:\Project\model7_phikon_cluster\dataset"
OUT_DIR = r"D:\Project\model7_phikon_cluster"

TARGET_CLASSES = ["cohesive", "dispersive", "noncohesive"]
for c in TARGET_CLASSES:
    os.makedirs(os.path.join(DEST_DIR, c), exist_ok=True)

def get_durgesh_raw(cls):
    files = glob.glob(os.path.join(SOURCE_DIR, cls, "*durgesh*.png"))
    raw = [f for f in files if not os.path.basename(f).startswith("aug_")]
    return raw

# 1. Cohesive: Sample 500 raw Durgesh tiles
raw_coh = get_durgesh_raw("cohesive")
random.seed(42)
random.shuffle(raw_coh)
chosen_coh = raw_coh[:500]
print(f"[+] Cohesive: Sampled {len(chosen_coh)} raw Durgesh tiles (from {len(raw_coh)} total).")
for f in chosen_coh:
    shutil.copy2(f, os.path.join(DEST_DIR, "cohesive", os.path.basename(f)))

# 2. Noncohesive: All 403 raw Durgesh tiles + augment to 500
raw_nc = get_durgesh_raw("noncohesive")
print(f"[+] Noncohesive: Copying {len(raw_nc)} raw Durgesh tiles.")
for f in raw_nc:
    shutil.copy2(f, os.path.join(DEST_DIR, "noncohesive", os.path.basename(f)))

# 3. Dispersive: All 185 raw Durgesh tiles + augment to 450
raw_dis = get_durgesh_raw("dispersive")
print(f"[+] Dispersive: Copying {len(raw_dis)} raw Durgesh tiles.")
for f in raw_dis:
    shutil.copy2(f, os.path.join(DEST_DIR, "dispersive", os.path.basename(f)))

# D4 rotations for balancing
def apply_d4(img_np, transform_idx):
    if transform_idx == 0:
        return cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    elif transform_idx == 1:
        return cv2.rotate(img_np, cv2.ROTATE_180)
    elif transform_idx == 2:
        return cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transform_idx == 3:
        return cv2.flip(img_np, 1)
    elif transform_idx == 4:
        return cv2.flip(img_np, 0)
    elif transform_idx == 5:
        return cv2.flip(cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE), 1)
    elif transform_idx == 6:
        return cv2.flip(cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE), 0)
    else:
        angle = np.random.choice([-15, -10, 10, 15, -20, 20])
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def process_one(args):
    src_path, transform_idx, out_path = args
    img = cv2.imread(src_path)
    if img is None:
        return False
    aug = apply_d4(img, transform_idx)
    cv2.imwrite(out_path, aug, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True

# Augment Noncohesive from 403 to 500 (+97)
nc_needed = 500 - len(raw_nc)
nc_tasks = []
for i in range(nc_needed):
    src = raw_nc[i % len(raw_nc)]
    t_idx = (i // len(raw_nc)) % 8
    base = os.path.splitext(os.path.basename(src))[0]
    out_p = os.path.join(DEST_DIR, "noncohesive", f"aug_rot_{base}_v{i:04d}.png")
    nc_tasks.append((src, t_idx, out_p))

# Augment Dispersive from 185 to 450 (+265)
dis_needed = 450 - len(raw_dis)
dis_tasks = []
for i in range(dis_needed):
    src = raw_dis[i % len(raw_dis)]
    t_idx = (i // len(raw_dis)) % 8
    base = os.path.splitext(os.path.basename(src))[0]
    out_p = os.path.join(DEST_DIR, "dispersive", f"aug_rot_{base}_v{i:04d}.png")
    dis_tasks.append((src, t_idx, out_p))

print(f"[+] Augmenting Noncohesive (+{nc_needed}) and Dispersive (+{dis_needed}) on 8 threads...")
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(process_one, nc_tasks + dis_tasks))

# Build dataset manifest
rows = []
for c in TARGET_CLASSES:
    folder = os.path.join(DEST_DIR, c)
    files = glob.glob(os.path.join(folder, "*.png"))
    print(f"  Final {c:<12}: {len(files)} tiles")
    for f in files:
        rows.append({"path": os.path.abspath(f), "label": c})

df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"\n[+] Total Model 7 Dataset: {len(df)} tiles")

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(df, df["label"]))

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

train_csv = os.path.join(OUT_DIR, "train.csv")
val_csv = os.path.join(OUT_DIR, "val.csv")

df_train.to_csv(train_csv, index=False)
df_val.to_csv(val_csv, index=False)

print(f"[+] Saved train split ({len(df_train)} tiles) to: {train_csv}")
print(f"[+] Saved val split   ({len(df_val)} tiles) to: {val_csv}")
print("\nValidation Distribution:")
print(df_val["label"].value_counts().to_string())

