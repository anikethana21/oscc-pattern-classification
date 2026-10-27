import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
hybridpath_split.py
-------------------
Reads manifest.csv, derives slide/patient IDs, and creates
patient-grouped train / val / test splits.

Test  : 15% of slides  -- ORIGINAL UNAUGMENTED, locked away
Val   : 10% of remaining -- ORIGINAL UNAUGMENTED, used during training
Train : rest -- augmented on-the-fly during training

Usage:
    python hybridpath_split.py --manifest ../manifest.csv
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../manifest.csv",
                    help="Path to manifest.csv (from build_manifest.py)")
    ap.add_argument("--out-dir", default=".",
                    help="Where to save split CSVs")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size",  type=float, default=0.12,
                    help="Fraction of trainval to use as val (~10% of total)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # -- Load manifest -------------------------------------------------------
    df = pd.read_csv(args.manifest)
    print(f"Manifest loaded: {len(df)} rows")

    if "dup_flag" in df.columns:
        before = len(df)
        df = df[df["dup_flag"] == False].reset_index(drop=True)
        print(f"Removed {before - len(df)} duplicates -> {len(df)} remaining")

    # -- Derive slide / patient IDs ------------------------------------------
    if "slide" not in df.columns:
        df["slide"] = df["path"].apply(
            lambda p: re.sub(r"_x\d+_y\d+(\s*\(\d+\))?$", "", Path(p).stem)
        )
    if "patient" not in df.columns:
        df["patient"] = df["slide"].str.extract(r"^(\d+)")[0].fillna(df["slide"])

    print(f"\nUnique slides  : {df['slide'].nunique()}")
    print(f"Unique patients: {df['patient'].nunique()}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts().to_string())

    # -- Split 1: carve off test (by slide) ----------------------------------
    gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                  random_state=args.seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=df["slide"]))

    df_test     = df.iloc[test_idx].reset_index(drop=True)
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)

    # -- Split 2: val from trainval ------------------------------------------
    gss_val = GroupShuffleSplit(n_splits=1, test_size=args.val_size,
                                 random_state=args.seed)
    train_idx, val_idx = next(gss_val.split(df_trainval,
                                             groups=df_trainval["slide"]))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    # -- Save ----------------------------------------------------------------
    df_train.to_csv(out_dir / "split_train.csv", index=False)
    df_val.to_csv(out_dir   / "split_val.csv",   index=False)
    df_test.to_csv(out_dir  / "split_test.csv",  index=False)

    print(f"\n{'='*50}")
    print(f"SPLIT COMPLETE")
    print(f"{'='*50}")
    print(f"Train : {len(df_train):>5} tiles")
    print(f"Val   : {len(df_val):>5} tiles  (original only)")
    print(f"Test  : {len(df_test):>5} tiles  (original only -- LOCKED)")
    print(f"\nTrain class breakdown:")
    print(df_train["label"].value_counts().to_string())
    print(f"\nVal class breakdown:")
    print(df_val["label"].value_counts().to_string())
    print(f"\nTest class breakdown:")
    print(df_test["label"].value_counts().to_string())
    print(f"\nSaved: split_train.csv, split_val.csv, split_test.csv")


if __name__ == "__main__":
    main()
