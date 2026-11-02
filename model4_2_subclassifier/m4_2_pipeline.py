"""
m4_2_pipeline.py
----------------
Full Unified Two-Tier Inference Pipeline:
  1. Runs Model 4.0 Cascade:
     - Cohesive, Noncohesive, Dispersive are finalized and scored.
     - Ambiguous/no_feature tiles are captured.
  2. Runs Model 4.2 Sub-classifier:
     - Classifies all no_feature tiles into the 4 Seniors' sub-classes.
     - Logs confidence and full probabilities.
  3. Cache Cleanup:
     - Automatically wipes out intermediate no_feature cache.
     - Produces master CSV with 100% of tiles classified into their final distinct categories.
     - Output sorted with least confident / borderline tiles at top for rapid human review.
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, "D:/Project")
sys.path.insert(0, "D:/Project/model4_2_subclassifier")

from model4_export import Model4Predictor
from m4_2_model import Model4_2_Classifier
from m4_2_dataset import make_16_crops


class UnifiedPipeline:
    def __init__(
        self,
        m4_2_weights: str = "D:/Project/model4_2_subclassifier/weights/model4_2_best.pt",
        classes_json: str = "D:/Project/model4_2_subclassifier/classes.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        print(f"\n[Unified Pipeline] Initializing Model 4.0 Primary Classifier...")
        self.m4_predictor = Model4Predictor(device=str(self.device))

        print(f"[Unified Pipeline] Initializing Model 4.2 Sub-Classifier...")
        with open(classes_json, "r") as f:
            self.sub_classes = json.load(f)

        self.m4_2_model = Model4_2_Classifier(num_classes=len(self.sub_classes), device=str(self.device)).to(self.device)
        ckpt = torch.load(m4_2_weights, map_location=self.device, weights_only=False)
        self.m4_2_model.load_state_dict(ckpt["model_state"])
        self.m4_2_model.eval()
        print(f"[Unified Pipeline] Sub-classes: {self.sub_classes}")

    def run_pipeline(
        self,
        input_tiles_dir: str,
        output_csv: str = "D:/Project/model4_2_subclassifier/pipeline_final_predictions.csv",
        cleanup_cache: bool = True
    ):
        print(f"\n=== Processing Slide / Folder: {input_tiles_dir} ===")
        # Discover tiles
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        tile_paths = []
        for ext in exts:
            tile_paths.extend(Path(input_tiles_dir).glob(ext))
        print(f"Total tiles found: {len(tile_paths)}")

        final_records = []
        no_feature_tiles = []

        # Tier 1: Model 4.0
        print("\n--- Running Tier 1 (Model 4.0 Primary Screening) ---")
        for i, tp in enumerate(tile_paths, 1):
            res = self.m4_predictor.predict(str(tp))
            pred = res["pred_class"]
            conf = res["confidence"]

            if pred == "no_feature":
                no_feature_tiles.append((tp, res))
            else:
                final_records.append({
                    "tile_id": tp.stem,
                    "tile_path": str(tp),
                    "tier1_class": pred,
                    "final_class": pred,
                    "confidence": conf,
                    "decided_by": f"Model 4.0 ({res['decided_by']})",
                    "sub_probabilities": "N/A (Tumor Pattern)",
                })
            if i % 50 == 0 or i == len(tile_paths):
                print(f"  Processed {i}/{len(tile_paths)} tiles...")

        print(f"Primary Tumor Patterns: {len(final_records)} | Routed to Model 4.2: {len(no_feature_tiles)}")

        # Tier 2: Model 4.2 for no_feature tiles
        if no_feature_tiles:
            print("\n--- Running Tier 2 (Model 4.2 Sub-Classifier) ---")
            for tp, t1_res in no_feature_tiles:
                # Preprocess
                crops_224, img_256, _ = self.m4_predictor.preprocess_image(str(tp))
                with torch.no_grad():
                    with torch.amp.autocast(self.device.type, dtype=torch.float16):
                        logits = self.m4_2_model(crops_224, img_256)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                best_idx = int(np.argmax(probs))
                sub_cls = self.sub_classes[best_idx]
                sub_conf = float(probs[best_idx])
                sub_probs_dict = {self.sub_classes[j]: round(float(probs[j]), 4) for j in range(len(self.sub_classes))}

                final_records.append({
                    "tile_id": tp.stem,
                    "tile_path": str(tp),
                    "tier1_class": "no_feature",
                    "final_class": sub_cls,
                    "confidence": round(sub_conf, 4),
                    "decided_by": "Model 4.2 (Sub-classifier)",
                    "sub_probabilities": str(sub_probs_dict),
                })

        df = pd.DataFrame(final_records)
        # Sort ascending: least confident / borderline tiles at top
        df_sorted = df.sort_values(by="confidence", ascending=True).reset_index(drop=True)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df_sorted.to_csv(output_csv, index=False)
        print(f"\n[+] Saved complete final classifications to: {output_csv}")

        # Cache cleanup
        if cleanup_cache:
            cache_folder = Path(input_tiles_dir) / "no_feature_cache"
            if cache_folder.exists() and cache_folder.is_dir():
                shutil.rmtree(cache_folder)
                print(f"[+] Cleaned up temporary cache: {cache_folder}")

        print("\n=== Final Class Distribution Summary ===")
        print(df_sorted["final_class"].value_counts().to_string())
        return df_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tile directory")
    parser.add_argument("--output", default="D:/Project/model4_2_subclassifier/pipeline_final_predictions.csv")
    args = parser.parse_args()

    pipeline = UnifiedPipeline()
    pipeline.run_pipeline(args.input, args.output)

