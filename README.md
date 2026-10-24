# 🔬 OSCC Invasion Pattern Classification in Histopathology

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Foundation Model](https://img.shields.io/badge/Backbone-Phikon%20(ViT--B)-purple.svg)](https://huggingface.co/owkin/phikon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated deep learning and computational pathology pipeline for classifying **Pattern of Invasion (POI)** in **Oral Squamous Cell Carcinoma (OSCC)** whole-slide image (WSI) tiles.

This repository implements two models and benchmarks them side-by-side:
1. **Model 1 (Baseline):** Foundation model deep embeddings (**Phikon ViT-B**, 1,536-dim).
2. **Model 2 (V2 — State-of-the-Art):** Hybrid feature fusion combining **Phikon deep embeddings** with **10 named morphological and color biomarkers** (1,546-dim).

---

## 📊 Benchmark Results (5-Fold Stratified Cross-Validation)

Both models were evaluated across **2,557 annotated OSCC tiles** across all 4 histopathological classes: `cohesive`, `dispersive`, `noncohesive`, and `no_feature`.

| Metric | Model 1 (Baseline: Phikon Only) | Model 2 (V2: Phikon + 10 Named Features) | Delta |
| :--- | :---: | :---: | :---: |
| **Input Feature Dimension** | 1,536 deep | 1,546 (1,536 deep + 10 named) | +10 features |
| **Overall Accuracy** | **86.0%** | **87.0%** | **+1.0%** |
| **Mean Macro-F1** | **0.8062 ± 0.0337** | **0.8145 ± 0.0248** | **+0.0083 (Higher stability)** |
| **Dispersive Precision** | **0.66** | **0.73** | **+7.0%** |
| **Dispersive F1-Score** | **0.70** | **0.72** | **+2.0%** |
| **Cohesive F1-Score** | **0.90** | **0.91** | **+1.0%** |
| **No Feature F1-Score** | **0.89** | **0.90** | **+1.0%** |

> 📌 See the detailed [Full Benchmark Report](MODEL_COMPARISON.md) for fold-by-fold breakdowns and confusion matrices.

---

## 🧬 Biological & Morphological Feature Importance

Model 2 extracts 10 clinically interpretable biomarkers using OpenCV color thresholding, morphology, and nearest-neighbor spatial clustering.

The classifier assigns **33% higher average weight** to the named biomarkers (`0.0198`) than to the raw deep features (`0.0149`):

| Rank | Biomarker Feature | Mean Weight | Clinical Relevance |
| :---: | :--- | :---: | :--- |
| **1** | `nuclei_density` | **0.0322** | **Dispersive (42.8) & Noncohesive (42.1)** have nearly 2x the cellular density of **Cohesive (22.6)**. |
| **2** | `red_fraction` | **0.0272** | **Noncohesive** exhibits high stromal hemorrhage/erythrocyte fraction (0.32 vs 0.12–0.16). |
| **3** | `nuclei_count` | **0.0247** | Absolute cellular count per 2048px field. |
| **4** | `std_nn_distance` | **0.0210** | Standard deviation of nearest-neighbor distance (quantifies spatial clustering vs random infiltration). |
| **5** | `mean_nucleus_area` | **0.0184** | Stromal tissue / no feature has larger background objects (773px) vs tumor nuclei (562–653px). |
| **6** | `mean_eccentricity` | **0.0175** | Nuclear elongation index (spindle vs circular). |
| **7** | `mean_circularity` | **0.0174** | Roundness metric ($4\pi \cdot \text{area} / \text{perimeter}^2$). |
| **8** | `pink_fraction` | **0.0152** | Keratin pearls and eosinophilic cytoplasmic content. |
| **9** | `std_nucleus_area` | **0.0148** | Nuclear pleomorphism (variation in size). |
| **10** | `mean_nn_distance` | **0.0096** | Mean cell-to-cell separation distance. |

---

## 📁 Repository Structure

```
├── README.md                           # Documentation & benchmark
├── MODEL_COMPARISON.md                 # In-depth statistical analysis & report
├── build_manifest.py                   # Data ingestion, regex parser & duplicate detector
├── manifest.csv                        # Registered tile dataset (2,557 rows)
├── requirements.txt                    # Project dependencies
│
├── model/                              # Model 1 (Baseline) weights & artifacts
│   ├── linear_head.pt                  # Trained PyTorch weights (26 KB)
│   ├── config.json                     # Architecture metadata & hyperparameters
│   └── label_classes.npy               # Class mapping
├── extract_features.py                 # Baseline feature extraction (Phikon ViT-B)
├── train.py                            # Baseline training (5-fold Stratified K-Fold)
├── predict.py                          # Baseline inference script
├── predictions_cv.csv                  # Out-of-fold validation predictions
│
├── second_model/                       # Model 2 (V2 with Named Features)
│   ├── model/
│   │   ├── linear_head.pt              # Trained V2 weights (26 KB)
│   │   ├── config.json                 # V2 architecture & metrics
│   │   └── label_classes.npy           # Class mapping
│   ├── named_mean.npy                  # Z-score normalization means (10 features)
│   ├── named_std.npy                   # Z-score normalization standard deviations
│   ├── extract_features_v2.py          # Fast hybrid extraction (Phikon + OpenCV morphology)
│   ├── train.py                        # V2 training & feature importance analyzer
│   ├── predict.py                      # V2 inference script
│   ├── embedding_labels_v2.csv         # Computed biomarker values for all tiles
│   └── predictions_cv_v2.csv           # Out-of-fold validation predictions
│
└── exports/                            # Standalone ready-to-share ZIP packages
    ├── OSCC_Model1_Baseline.zip        # ~193 KB
    └── OSCC_Model2_V2_NamedFeatures.zip # ~316 KB
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/anikethana21/oscc-pattern-classification.git
cd oscc-pattern-classification
```

### 2. Install Dependencies
```bash
# PyTorch with CUDA 12.6 (or your system's CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Remaining requirements
pip install transformers scikit-learn pandas opencv-python scipy pillow tqdm
```

---

## 🔍 Running Inference on New Tiles

### Using Model 2 (V2 — Recommended):
```bash
cd second_model
python predict.py --input "path/to/tile_or_folder" --out "predictions.csv"
```

The script outputs `predictions.csv` containing:
* Predicted Class (`cohesive`, `dispersive`, `noncohesive`, `no_feature`)
* Confidence Score (Softmax probability)
* Flag for human review (`True` if confidence < 70%)

---

## 🏋️ Training from Scratch

If you have tile images and want to reproduce training:

```bash
# Step 1: Scan tile directories and build manifest
python build_manifest.py --root "path/to/tiles" --out manifest.csv

# Step 2: Extract features (Model 1 & 2)
python extract_features.py --manifest manifest.csv
cd second_model && python extract_features_v2.py --manifest ../manifest.csv

# Step 3: Train 5-fold cross-validation
python train.py
cd second_model && python train.py
```

---

## 📜 Citation & Credits

* **Backbone Foundation Model:** [Phikon](https://huggingface.co/owkin/phikon) (*Owkin, Nature Medicine*).
* **Research Team:** Anikethana, Durgesh, and Amrutha.
