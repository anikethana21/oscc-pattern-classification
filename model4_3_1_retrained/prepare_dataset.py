"""
prepare_dataset.py (Model 4.3.1)
-------------------------------
Creates stratified train and validation splits from D:/new_whip_431.
Total tiles: 5491 (1580 cohesive, 1000 dispersive, 1711 no_feature, 1200 noncohesive).
Generates:
  - D:/Project/model4_3_1_retrained/train.csv (85%)
  - D:/Project/model4_3_1_retrained/val.csv (15%)
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_DIR = r"D:\new_whip_431"
OUT_DIR = r"D:\Project\model4_3_1_retrained"
os.makedirs(OUT_DIR, exist_ok=True)

classes = ["cohesive", "dispersive", "no_feature", "noncohesive"]
rows = []

for c in classes:
    folder = os.path.join(DATA_DIR, c)
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    for f in files:
        rows.append({"path": os.path.abspath(f), "label": c})

df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"Total tiles in dataset: {len(df)}")
print(df["label"].value_counts().to_string())

# 85% train, 15% validation stratified split (6 folds: ~83.3% / 16.7%)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(df, df["label"]))

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

train_csv = os.path.join(OUT_DIR, "train.csv")
val_csv = os.path.join(OUT_DIR, "val.csv")

df_train.to_csv(train_csv, index=False)
df_val.to_csv(val_csv, index=False)

print(f"\n[+] Saved train split ({len(df_train)} tiles) to: {train_csv}")
print(f"[+] Saved val split   ({len(df_val)} tiles) to: {val_csv}")
print("\nValidation Class Distribution:")
print(df_val["label"].value_counts().to_string())

