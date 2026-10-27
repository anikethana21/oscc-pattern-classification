# Oral Squamous Cell Carcinoma (OSCC) Histopathology Classification

Comprehensive AI research system evaluating 5 generations of deep neural architectures on high-resolution 40x WSI patches.

---

## Master Architecture & Study Guide
For the full architectural deep dive, layer-by-layer tensor flows, activation functions, mathematical equations, and Model 6 blueprint, see:
- **[MASTER_ARCHITECTURE_STUDY_GUIDE.md](file:///D:/Project/MASTER_ARCHITECTURE_STUDY_GUIDE.md)**

---

## Model Evolution Summary (Evaluated on Locked Test Set, N=267 Tiles)

| Model | Architecture | Locked Test Macro-F1 | Overall Accuracy | Dispersive F1 | Noncohesive F1 | Key Strength |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | Baseline Phikon Deep Features | 0.6974 | 74.9% | 0.645 | 0.495 | Fast linear baseline |
| **Model 2** | Phikon + 10 Named Morphological Features | 0.7424 | 79.8% | 0.737 | 0.541 | Exploits nuclei density & circularity |
| **Model 3** | HybridPath (3-Branch Dynamic Gating) | 0.7587 | 80.9% | 0.800 | 0.529 | High dispersive recall (88.9%) |
| **Model 4** | **Augmented Specialist Network (~2.5k Scaled)** | **0.8321** | **87.6%** | **0.849** | **0.696** | **Top overall accuracy & macro-F1** |
| **Model 5** | **Recurrent Feedback Looping Dual-Stage Network** | **0.8179** | **87.0%** | **0.839** | **0.654** | **100% precision on dispersive (0 false alarms)** |

---

## Repository Structure & Model Folders

- [`MASTER_ARCHITECTURE_STUDY_GUIDE.md`](file:///D:/Project/MASTER_ARCHITECTURE_STUDY_GUIDE.md): Complete architectural trace, tensor dimensions, activations, and study notes.
- [`model4_cascade/`](file:///D:/Project/model4_cascade/): Model 4 codebase (dataset scaling, Stage 1 screener, Stage 2 specialist, evaluations).
- [`model5_loop/`](file:///D:/Project/model5_loop/): Model 5 codebase (recurrent feedback loop network, multi-task training, evaluations).
- [`hybridpath_model/`](file:///D:/Project/hybridpath_model/): Model 3 codebase (HybridPath 3-branch gated network).
- [`second_model/`](file:///D:/Project/second_model/): Model 2 codebase (Phikon + tabular morphological features).
- [`all_5_models_confusion_matrix.png`](file:///D:/Project/all_5_models_confusion_matrix.png): Complete 5-panel side-by-side confusion matrix figure.
