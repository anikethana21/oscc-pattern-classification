import os
import sys
import argparse
import pandas as pd


def sort_tiles(input_csv="D:/Project/model6_earlyexit/tile_scores.csv",
               output_csv=None,
               ascending=False,
               per_class=False):
    """
    Reads tile_scores.csv and writes tile_scores_sorted.csv.
    Groups by pred_class, then sorts by confidence (descending by default, or ascending for human review).
    Optionally outputs one CSV per class.
    Does NOT move or copy any tile images.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input scores CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    if "pred_class" not in df.columns or "confidence" not in df.columns:
        raise ValueError("Input CSV must contain 'pred_class' and 'confidence' columns.")

    if output_csv is None:
        base_dir = os.path.dirname(os.path.abspath(input_csv))
        direction_tag = "_least_confident" if ascending else ""
        output_csv = os.path.join(base_dir, f"tile_scores_sorted{direction_tag}.csv")

    # Sort: group by pred_class, then by confidence
    # (stable sorting ensures consistent grouping)
    sorted_df = df.sort_values(by=["pred_class", "confidence"], ascending=[True, ascending]).reset_index(drop=True)

    sorted_df.to_csv(output_csv, index=False)
    direction_str = "ascending (least confident first)" if ascending else "descending (most confident first)"
    print(f"Saved sorted tiles ({direction_str}) to: {output_csv}")

    # Summary
    print("\n=== Summary by Class ===")
    for cls, group in sorted_df.groupby("pred_class"):
        mean_conf = group["confidence"].mean()
        min_conf = group["confidence"].min()
        max_conf = group["confidence"].max()
        print(f"  Class {cls:<15}: {len(group):>4d} tiles | Conf: mean={mean_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")

    # Optional per-class files
    if per_class:
        base_dir = os.path.dirname(os.path.abspath(output_csv))
        for cls, group in sorted_df.groupby("pred_class"):
            class_csv = os.path.join(base_dir, f"tiles_{cls}_sorted.csv")
            group.to_csv(class_csv, index=False)
            print(f"  [Per-class] Saved {len(group)} tiles to {class_csv}")

    return sorted_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort tile predictions by class and confidence score.")
    parser.add_argument("--input", default="D:/Project/model6_earlyexit/tile_scores.csv", help="Path to tile_scores.csv")
    parser.add_argument("--output", default=None, help="Path for sorted output CSV")
    parser.add_argument("--ascending", action="store_true", help="Sort least confident first (useful for active learning/human review)")
    parser.add_argument("--per-class", action="store_true", help="Also generate separate CSV file for each predicted class")
    args = parser.parse_args()

    sort_tiles(args.input, args.output, args.ascending, args.per_class)
