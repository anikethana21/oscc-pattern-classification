"""
Step 1: build a manifest of all labeled tiles.

Usage (Windows):
    python build_manifest.py --root "D:\Project" --out manifest.csv
    python build_manifest.py --root "D:\Project" --out manifest.csv --hash   # also detect byte-identical files

Expected layout:
    <root>/<batch folder>/40x/<class folder>/<slide>_x<X>_y<Y>.png
Class folders: cohesive, noncohesive, dispersive, no_feature
A " (1)" suffix on a class folder (e.g. "cohesive (1)") is treated as a second label set.
    set0 = Durgesh. B (no suffix)
    set1 = Amrutha S G (suffix "(1)")
"""
import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_ALIASES = {
    "cohesive": "cohesive",
    "noncohesive": "noncohesive",
    "non_cohesive": "noncohesive",
    "non-cohesive": "noncohesive",
    "dispersive": "dispersive",
    "no_feature": "no_feature",
    "nofeature": "no_feature",
    "no-feature": "no_feature",
}
FOLDER_RE = re.compile(r"^(?P<name>.+?)(?:\s*\((?P<n>\d+)\))?$")
FILE_RE = re.compile(r"^(?P<slide>.+)_x(?P<x>\d+)_y(?P<y>\d+)(?:\s*\(\d+\))?$")
MAG_RE = re.compile(r"^\d+x$", re.IGNORECASE)
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--hash", action="store_true",
                    help="md5 every file (slower, finds identical copies)")
    args = ap.parse_args()

    root = Path(args.root)
    rows, skipped_folders, bad_names, bad_images = [], Counter(), [], []

    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(files)} image files under {root}")

    for p in files:
        rel = p.relative_to(root).parts
        batch = rel[0] if len(rel) > 1 else ""
        m = FOLDER_RE.match(p.parent.name)
        cls_raw = m.group("name").strip().lower()
        cls = CLASS_ALIASES.get(cls_raw)
        if cls is None:
            skipped_folders[p.parent.name] += 1
            continue
        label_set = f"set{m.group('n')}" if m.group("n") else "set0"
        mag = (p.parent.parent.name
               if MAG_RE.match(p.parent.parent.name) else "unknown")

        fm = FILE_RE.match(p.stem)
        if not fm:
            bad_names.append(str(p))
            continue
        slide = fm.group("slide")
        pm = re.match(r"^(\d+)", slide)
        patient = pm.group(1) if pm else slide

        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            bad_images.append((str(p), str(e)))
            continue

        rows.append(dict(
            path=str(p), batch=batch, mag=mag, label=cls,
            label_set=label_set, slide=slide, patient=patient,
            x=int(fm.group("x")), y=int(fm.group("y")),
            width=w, height=h, bytes=p.stat().st_size,
            md5=md5(p) if args.hash else "",
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No usable tiles found. Check --root and folder names.")
        return

    # Exact repeats
    key = ["slide", "x", "y", "label", "label_set"]
    df["dup_flag"] = df.duplicated(subset=key, keep="first")
    # Byte-identical files
    df["identical_file_flag"] = (
        df.duplicated(subset=["md5"], keep="first") if args.hash else False
    )
    # Label conflicts
    uniq = df[~df["dup_flag"]]
    n_labels = uniq.groupby(["slide", "x", "y"])["label"].nunique()
    conflict_idx = n_labels[n_labels > 1].index
    df["conflict_flag"] = (
        df.set_index(["slide", "x", "y"]).index.isin(conflict_idx)
    )

    df.to_csv(args.out, index=False)

    # ---------- report ----------
    print("\n=== SUMMARY ===")
    print(
        f"Usable tiles: {len(df)}   "
        f"exact repeats flagged: {int(df.dup_flag.sum())}"
        + (f"   byte-identical files: {int(df.identical_file_flag.sum())}"
           if args.hash else "")
    )
    print("\nTiles per class x label_set (repeats removed):")
    print(pd.crosstab(uniq["label"], uniq["label_set"], margins=True))
    print("\nTiles per batch folder:")
    print(uniq["batch"].value_counts().sort_index().to_string())
    print(
        f"\nDistinct slides: {uniq.slide.nunique()}   "
        f"distinct patients (leading ID): {uniq.patient.nunique()}"
    )
    print(
        "Tiles per patient (min / median / max): "
        f"{uniq.groupby('patient').size().min()} / "
        f"{int(uniq.groupby('patient').size().median())} / "
        f"{uniq.groupby('patient').size().max()}"
    )
    print("\nTile pixel sizes found:")
    print(uniq.groupby(["mag", "width", "height"]).size().to_string())

    # Inter-annotator agreement
    both = uniq.pivot_table(
        index=["slide", "x", "y"], columns="label_set",
        values="label", aggfunc="first"
    )
    if {"set0", "set1"}.issubset(both.columns):
        pair = both.dropna(subset=["set0", "set1"])
        if len(pair):
            agree = (pair["set0"] == pair["set1"]).mean()
            print(
                f"\nTiles labeled in BOTH set0 and set1: {len(pair)}   "
                f"agreement: {agree:.1%}"
            )
            try:
                from sklearn.metrics import cohen_kappa_score
                print(
                    f"Cohen's kappa: "
                    f"{cohen_kappa_score(pair['set0'], pair['set1']):.3f}"
                )
            except ImportError:
                print("(install scikit-learn to get Cohen's kappa)")
        else:
            print(
                "\nNo tile appears in both set0 and set1 "
                "(the two sets label different tiles)."
            )

    print(f"\nLabel conflicts (same tile in different classes): "
          f"{len(conflict_idx)}")
    if skipped_folders:
        print("\nSkipped folders (name not recognized as a class):",
              dict(skipped_folders))
    if bad_names:
        print(f"\n{len(bad_names)} files did not match "
              f"<slide>_x<X>_y<Y> naming, e.g. {bad_names[:3]}")
    if bad_images:
        print(f"\n{len(bad_images)} unreadable images, "
              f"e.g. {bad_images[:3]}")
    print(f"\nManifest written to {args.out}")


if __name__ == "__main__":
    main()