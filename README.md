# Oral Squamous Cell Carcinoma (OSCC) Histopathology AI Research Platform
### Multi-Scale Deep Vision Transformers, Dynamic Gated Fusion & Metric Sub-Clustering Systems

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu126-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.1%20%7C%20RTX%204070-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Foundation Model](https://img.shields.io/badge/Foundation%20Model-Owkin%20Phikon%20ViT-blue.svg)](https://huggingface.co/owkin/phikon)
[![Streamlit](https://img.shields.io/badge/Web%20UI-Streamlit%201.67.0-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 Project Overview

This repository represents an end-to-end artificial intelligence research platform for the automated screening, pattern segregation, and sub-phenotypic clustering of **Oral Squamous Cell Carcinoma (OSCC)** from Whole Slide Images (WSI).

Spanning multiple design generations (from baseline ResNets to multi-scale Vision Transformers with Multi-Class Focal Loss and Riemannian Metric Clustering), this platform models the entire histological continuum:
- **Pre-Malignant Lesions:** Epithelial Dysplasia
- **Benign Baseline:** Normal Oral Mucosa
- **Invasive Patterns:** Cohesive sheets (Grade I/II), Non-Cohesive micro-nests (Grade II/III), and Dispersive single-cell cords (Grade III / High-Risk Pattern of Invasion 4-5)
- **Artifact Rejection:** Non-feature stroma, hemorrhage, and blank optical glass slides

---

## 🏆 Master Model Evolution & Benchmark Scorecard

| Model Identifier | Optical Zoom | Architectural Paradigm | Target Classes | Validation Accuracy | Validation Macro-F1 | Key Breakthrough / Role |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Model 1** | 40x | Baseline Phikon ViT (Frozen Features) | 3 Classes | 74.90% | 0.6974 | Linear classification baseline |
| **Model 2** | 40x | Dual-Path: Phikon + 10 Morphological Features | 3 Classes | 79.80% | 0.7424 | Incorporates nuclear circularity & density |
| **Model 3** | 40x | HybridPath Tri-Branch Network | 3 Classes | 80.90% | 0.7587 | Dynamic gating between spatial streams |
| **Model 4.0 - 4.2** | 40x | Two-Stage Cohesive Screener + Specialist | 3 Classes | 87.60% | 0.8321 | First dedicated invasive pattern cascade |
| **Model 4.3 - 4.3.1** | 40x | Tri-Branch (Frozen ViT + Tuned ViT + TileCNN) | 3 Classes | 93.30% | 0.9328 | Fine-tuning top transformer blocks with Focal Loss |
| **Model 4.3.2** | 40x | Gated Tri-Branch with Multi-Scale Context | 3 Classes | **94.21%** | **0.9427** | 🏆 **40x Supervised Baseline Champion** |
| **Model 5** | 40x | Recurrent Feedback Looping Dual-Stage Net | 3 Classes | 87.00% | 0.8179 | Feedback looping for zero dispersive false alarms |
| **Model 6** | 40x | Early-Exit Confidence-Gated Network | 4 Classes | 88.40% | 0.8410 | Sub-millisecond inference on obvious patches |
| **Model 7** | 40x | Foundation Feature Metric Clustering | 3 Classes | 91.20% | 0.9105 | Unsupervised manifold analysis |
| **Model 8** | 40x | Deep Phikon Domain-Adapted Classifier | 3 Classes | 93.80% | 0.9387 | Unfreezing layers 8-11 with layerwise decay |
| **Model 9.3** | 40x | Multi-Prototype Sub-Clustering Net | 3 Classes | **94.21%** | **0.9431** | 🏆 **40x Metric Sub-Clustering Champion** (6 Centroids) |
| **Model 104.4** | **20x** | **7-Class Gated Tri-Branch + Focal Loss ($\gamma=2$)** | **7 Classes** | **97.19%** | **0.9747** | 🏆 **Next-Gen 7-Class State-of-the-Art Screener** |
| **Model 109.4** | **20x** | **Multi-Prototype Metric Net + Outlier Gate** | **5 Biological** | **97.33%** | **0.9750** | 🏆 **Next-Gen Biological Metric Champion** (10 Centroids) |

---

## 📂 Curated Repository Structure

```
oscc-pattern-classification/
│
├── models/                           # Model Architectures, Training Scripts & Checkpoints
│   ├── 01_baseline_resnet/           # Model 1: Baseline linear classifier on foundation embeddings
│   ├── 02_dualpath_resnet/           # Model 2: Dual-path vision + cellular morphometry
│   ├── 03_hybridpath/                # Model 3: Early 3-branch gated fusion
│   ├── 04_cascade_series/            # Model 4 Suite: Cascades, Stage 1/2, and 4.3.2 Champion
│   │   ├── 4.0_cascade_base/         # Model 4.0 baseline cascade
│   │   ├── 4.2_subclassifier/        # Model 4.2 sub-classification head
│   │   ├── 4.3_tri_branch/           # Model 4.3 tri-branch foundation
│   │   ├── 4.3.1_focal/              # Model 4.3.1 class-weighted focal loss
│   │   ├── 4.3.2_champion_40x/       # Model 4.3.2 40x Champion (94.21% Val Acc)
│   │   ├── 4.3.2.1_extended/         # Model 4.3.2.1 extended margin model
│   │   └── exports/                  # Model 4 deployment weights & metadata
│   ├── 05_feedback_loop/             # Model 5: Recurrent feedback looping architecture
│   ├── 06_early_exit/                # Model 6: Dynamic early-exit inference network
│   ├── 07_phikon_cluster/            # Model 7: Latent clustering & PCA projections
│   ├── 08_phikon_deep_vit/           # Model 8: Deep fine-tuned ViT blocks 8-11 (93.80% Acc)
│   ├── 09_metric_clustering/         # Model 9 Suite: 9.1, 9.2, and 9.3 Champions (94.21% Acc)
│   ├── 104.4_tri_branch_wsi_7class/  # Model 104.4: 20x 7-Class Unified Screener (97.19% Acc, 0.9747 F1)
│   └── 109.4_metric_clustering_5class/ # Model 109.4: 20x 5-Class 10 Sub-Centroid Net (97.33% Acc, 0.9750 F1)
│
├── apps/                             # Interactive Demonstration & Clinical Applications
│   └── streamlit_wsi_demo/           # Streamlit Web UI: Gigapixel WSI Streaming & Dynamic Heatmap
│       ├── demo_app_104_4.py         # Main interactive web application
│       ├── generate_architecture_diagram.py # Vector architecture rendering script
│       └── model104_4_architecture_diagram.png # Dark-mode architecture diagram
│
├── pipelines/                        # Automated Data Preparation & Training Orchestrators
│   ├── prepare_20x_dataset.py        # In-memory streaming 20x extraction & D4 dihedral augmentation
│   ├── run_master_104_109.py         # Master overnight sequential runner for 104.4 & 109.4
│   ├── run_experiments_overnight.py  # Overnight benchmarking suite
│   ├── build_manifest.py             # Manifest generation & stratified split verification
│   ├── extract_features.py           # Pre-computation of foundation ViT embeddings
│   ├── train_baseline.py             # Baseline training entry point
│   └── predict_baseline.py           # Baseline inference evaluation script
│
├── reports/                          # Comprehensive Technical & Publication Reports
│   ├── docx/                         # Formal Microsoft Word (.docx) publication reports
│   │   ├── OSCC_Models_104_and_109_Comprehensive_Report.docx
│   │   ├── OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx
│   │   └── Model_4_3_2_Performance_Report.docx
│   ├── markdown/                     # Technical specifications, interview guides & study notes
│   │   ├── MASTER_ARCHITECTURE_STUDY_GUIDE.md
│   │   ├── OSCC_AI_INTERVIEW_COMPREHENSIVE_GUIDE.md
│   │   └── MODEL_4_3_2_1_EXECUTIVE_SUMMARY_SHEET.md
│   └── figures/                      # High-resolution confusion matrices, ROC curves & PCA maps
│
├── benchmarks/                       # Cross-Model Benchmarking & Confusion Analysis
│   ├── benchmark_out/                # K-fold cross-validation logs and metrics
│   ├── benchmark.py                  # Standardized evaluation harness
│   ├── plot_all_5_models_confusion_matrix.py # 5-panel comparative matrix plotter
│   └── print_cms.py                  # Exact tabular confusion matrix extraction utility
│
├── data_manifests/                   # Manifests, Splits & Prediction Audits (Zero Raw Pixels)
│   ├── dataset_20x/                  # 20x master manifests (manifest_104_4.csv, manifest_109_4.csv)
│   ├── dataset_40x/                  # 40x baseline manifests (manifest.csv, tiles.csv)
│   └── predictions/                  # Validation predictions & outlier screening logs
│
├── utilities/                        # Environment Diagnostics & Verification Scripts
│   ├── check_deps.py                 # Dependency verification suite
│   ├── check_wsi_libs.py             # Gigapixel WSI library audit
│   ├── inspect_zips.py               # Raw dataset ZIP archive inspector
│   ├── test_demo_backend.py          # Backend inference test script
│   └── test_wsi_handler.py           # Memory-safe pyramidal TIFF streaming test
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Master Documentation
```

---

## 🚀 Interactive Streamlit WSI Platform (Model 104.4)

An interactive diagnostic demonstrator is available for clinical Whole Slide Image (WSI) screening:
- **Gigapixel Ingestion:** Reads up to **65,000 × 65,000 px** TIFF/WSI biopsy scans via memory-mapped pyramidal streaming (`tifffile`) with zero `MemoryError`.
- **Dynamic 7-Class Spatial Heatmap:** Slices multi-megapixel WSIs into standardized optical fields, evaluates each tile on the RTX 4070, and draws a color-coded pathological overlay.
- **Micro-Architectural Receptive Fields:** Displays the $4 \times 4$ grid of $224 \times 224$ crops analyzed by the Vision Transformer for every patch.

### Launch Instructions:
```bash
# Option 1: Double-click desktop shortcut
C:\Users\anike\Desktop\LAUNCH_OSCC_MODEL104_DEMO.bat

# Option 2: Command line execution
cd D:\Project
streamlit run apps/streamlit_wsi_demo/demo_app_104_4.py --server.port 8501
```

---

## 📊 Dataset Engineering & Dihedral D4 Symmetries

To guarantee mathematical invariance across slide orientations without synthetic hallucinations (such as SMOTE), dataset patches were augmented using the **Dihedral Group $D_4$** (8 isometric transformations: rotations of 0°, 90°, 180°, 270° and orthogonal reflections):

```
                       [ Raw 20x Tile (2048 x 2048) ]
                                      │
                         In-Memory Streaming Resize
                                      ▼
                             [ 512 x 512 RGB ]
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     [ D4 Dihedral Rotations ]                      [ Stratified 80/20 Split ]
  (rot90, rot180, rot270, flips)                    (Train: 2,280 | Val: 570)
```

- **Cohesive:** 600 tiles (sampled from 712 raw)
- **Dysplasia:** 500 tiles (200 raw + 300 D4 augmented)
- **Non-Cohesive:** 400 tiles (187 raw + 213 D4 augmented)
- **Dispersive:** 400 tiles (63 raw + 337 D4 augmented)
- **Normal Mucosa:** 350 tiles (66 raw + 284 D4 augmented)
- **No Feature (Stroma):** 500 tiles (sampled from 964 raw)
- **Blank Optical Glass:** 100 tiles (76 raw + 24 D4 augmented)
- **Total Pipeline Staged:** **2,850 tiles**

---

## 🧠 Model 104.4 & Model 109.4 Architectural Principles

### Model 104.4: 7-Class Supervised Tri-Branch Gated Screener
- **Branch A:** Frozen Phikon Foundation ViT (1536-d via mean+max pooling of 16 crops)
- **Branch B:** Domain-adapted Phikon top 4 transformer blocks (blocks 8–11 + LayerNorm, 1536-d)
- **Branch C:** Custom 4-stage TileCNN (512-d) preserving cytoplasmic borders and nuclear textures
- **Cross-Branch Attention Gating:** Adaptive softmax weights $(w_A, w_B, w_C)$ dynamically assigned per tile
- **Loss Function:** Multi-Class Focal Loss with inverse class-frequency weights:
  $$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^{\gamma} \cdot \log(p_t) \quad (\gamma = 2.0)$$

### Model 109.4: 5-Class Multi-Prototype Metric Clustering Network
- **10 Learnable Sub-Centroids:** Two orthogonal prototypes per biological class capturing intra-class morphological heterogeneity (e.g. solid sheets vs micro-nests, mild vs severe dysplasia)
- **Riemannian Metric Hypersphere:** 256-D L2-normalized metric embeddings scaled by learnable temperature $\tau$
- **Automated Outlier Radius Gate:** Statistical hypersphere boundary:
  $$R_c = \mu_c + 2.5 \cdot \sigma_c$$
  Automatically flags non-biological background and necrotic artifacts without distorting cancer classification boundaries.

---

## 📈 Citation & License

Developed as part of the Oral Squamous Cell Carcinoma (OSCC) Histopathology Pattern Classification Initiative.  
Distributed under the MIT License. See `LICENSE` for details.
