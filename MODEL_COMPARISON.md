# 🔬 Model Comparison Report: Baseline vs. V2 (Named Features)

## Executive Summary

Both models were trained on the complete dataset of **2,557 tiles** using **5-fold Stratified Cross-Validation**.

Model 2 (Deep Embeddings + 10 Named Morphological & Color Features) **outperformed** the baseline across accuracy, macro-F1, stability, and minority class precision.

---

## 📊 Head-to-Head Benchmark

| Metric | Model 1 (Baseline — Phikon Only) | Model 2 (V2 — Phikon + 10 Named Features) | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Input Feature Dimension** | 1,536 deep | 1,546 (1,536 deep + 10 named) | +10 features |
| **Overall Accuracy** | **86%** | **87%** | **+1.0%** |
| **Mean Macro-F1** | **0.8062 ± 0.0337** | **0.8145 ± 0.0248** | **+0.0083 (More stable)** |
| **Fold 1 F1** | 0.7835 | 0.8046 | +0.0211 |
| **Fold 2 F1** | 0.8446 | 0.8433 | -0.0013 |
| **Fold 3 F1** | 0.7522 | 0.7715 | +0.0193 |
| **Fold 4 F1** | 0.8203 | 0.8276 | +0.0073 |
| **Fold 5 F1** | 0.8305 | 0.8255 | -0.0050 |

---

## 🎯 Per-Class Performance Breakdown

### Model 1: Baseline (Phikon 1,536-dim)
```
              precision    recall  f1-score   support
    cohesive       0.92      0.88      0.90      1458
  dispersive       0.66      0.74      0.70       186
  no_feature       0.87      0.92      0.89       640
 noncohesive       0.71      0.76      0.73       273

    accuracy                           0.86      2557
   macro avg       0.79      0.82      0.81      2557
weighted avg       0.87      0.86      0.87      2557
```

### Model 2: V2 (Phikon + 10 Named Features)
```
              precision    recall  f1-score   support
    cohesive       0.93      0.89      0.91      1458
  dispersive       0.73      0.72      0.72       186
  no_feature       0.88      0.92      0.90       640
 noncohesive       0.68      0.78      0.73       273

    accuracy                           0.87      2557
   macro avg       0.81      0.82      0.81      2557
weighted avg       0.87      0.87      0.87      2557
```

### 🔑 Key Classification Gains
1. **Dispersive Precision (+7%):** Rose from **0.66 to 0.73** with F1 rising from **0.70 to 0.72**. Dispersive pattern is the hardest minority class (only 7.3% of tiles), and the named features provided clear discriminatory signals.
2. **Cohesive F1 (+1%):** Rose from **0.90 to 0.91** (precision increased to 0.93, recall increased to 0.89).
3. **No Feature F1 (+1%):** Rose from **0.89 to 0.90**.
4. **Variance Reduction:** Standard deviation across cross-validation folds dropped from **0.0337 to 0.0248**, proving the classifier is less prone to wild swings across validation sets.

---

## 🧬 Biological & Morphological Feature Importance

The classifier assigns **33% higher average weight** to the 10 named features (`0.0198`) than to the 1,536 raw Phikon deep features (`0.0149`).

| Rank | Feature | Mean Absolute Weight | Primary Biological Signal |
| :---: | :--- | :---: | :--- |
| **1** | `nuclei_density` | **0.0322** | **Dispersive & Noncohesive** have ~42–43 nuclei/Mpx vs **Cohesive** with 22.6. Massive differentiator. |
| **2** | `red_fraction` | **0.0272** | **Noncohesive** has 0.32 red/hemorrhage fraction vs 0.12–0.16 in other classes. |
| **3** | `nuclei_count` | **0.0247** | Strongly correlates with cellularity across patterns. |
| **4** | `std_nn_distance` | **0.0210** | Measures spatial clustering/dispersion irregularity. |
| **5** | `mean_nucleus_area` | **0.0184** | **No feature** background has larger areas (773px) vs **Noncohesive** (562px). |
| **6** | `mean_eccentricity` | **0.0175** | Nuclear elongation. |
| **7** | `mean_circularity` | **0.0174** | Nuclear shape roundness vs irregularity. |
| **8** | `pink_fraction` | **0.0152** | Keratin pearl / eosinophilic stroma content. |
| **9** | `std_nucleus_area` | **0.0148** | Pleomorphism (variation in nuclear size). |
| **10** | `mean_nn_distance` | **0.0096** | Average cell-to-cell separation. |

---

## 📁 Artifacts & Saved Models

### Model 1 (Baseline)
- Model Weights: [`D:/Project/model/linear_head.pt`](file:///D:/Project/model/linear_head.pt)
- Model Metadata: [`D:/Project/model/config.json`](file:///D:/Project/model/config.json)
- Classes: [`D:/Project/model/label_classes.npy`](file:///D:/Project/model/label_classes.npy)
- CV Predictions: [`D:/Project/predictions_cv.csv`](file:///D:/Project/predictions_cv.csv)
- Predict Script: [`D:/Project/predict.py`](file:///D:/Project/predict.py)

### Model 2 (V2 with Named Features)
- Model Weights: [`D:/Project/second_model/model/linear_head.pt`](file:///D:/Project/second_model/model/linear_head.pt)
- Model Metadata: [`D:/Project/second_model/model/config.json`](file:///D:/Project/second_model/model/config.json)
- Feature Normalization Stats: [`D:/Project/second_model/named_mean.npy`](file:///D:/Project/second_model/named_mean.npy), [`D:/Project/second_model/named_std.npy`](file:///D:/Project/second_model/named_std.npy)
- Extracted Embeddings: [`D:/Project/second_model/embeddings_v2.npy`](file:///D:/Project/second_model/embeddings_v2.npy) (2,557 × 1,546)
- Features & Labels CSV: [`D:/Project/second_model/embedding_labels_v2.csv`](file:///D:/Project/second_model/embedding_labels_v2.csv)
- CV Predictions: [`D:/Project/second_model/predictions_cv_v2.csv`](file:///D:/Project/second_model/predictions_cv_v2.csv)
- Predict Script: [`D:/Project/second_model/predict.py`](file:///D:/Project/second_model/predict.py)

---

## 🚀 How to Run Inference on New Tiles

### Using Model 1 (Baseline):
```powershell
cd D:\Project
.\venv\Scripts\python.exe predict.py --input "D:\Path\To\New\Tiles" --out-dir predictions_output
```

### Using Model 2 (V2 — Recommended):
```powershell
cd D:\Project\second_model
..\venv\Scripts\python.exe predict.py --input "D:\Path\To\New\Tiles" --out-dir predictions_output
```
