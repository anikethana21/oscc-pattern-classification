# Model 4.3.2.1: Executive Performance & Pipeline Integration Sheet

---

## 1. Release & Repository Overview

| Property | Value |
| :--- | :--- |
| **Model Version** | **Model 4.3.2.1 (3-Class Cancer Sub-Specialist)** |
| **Primary Target** | Refined classification across OSCC cancer patterns (`cohesive`, `dispersive`, `noncohesive`) |
| **GitHub Repository** | `https://github.com/anikethana21/oscc-pattern-classification.git` |
| **Branch** | `main` |
| **Commit Hashes** | `c45d0d8` (Weights & Matrix) / `ef297fc` (Pipeline code & splits) |
| **Git LFS Tracked** | `model4_3_2_1_retrained/weights/model4_3_2_1_best.pt` (707 MB) |
| **Working Tree** | `clean` (100% synced with remote) |

---

## 2. Benchmark Accuracy & Performance Metrics

Evaluated on the independent validation split (**206 unseen validation tiles**):

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Validation Accuracy** | **97.57%** | 201 correct out of 206 validation tiles |
| **Macro-F1 Score** | **0.9778** | Near-perfect harmonic mean across all 3 classes |
| **Macro-Precision** | **97.93%** | Minimal false alarms across any tumor subtype |
| **Macro-Recall** | **97.71%** | Negligible missed pathology across all 3 classes |

---

## 3. Confusion Matrix (True vs Predicted)

| True Pattern \ Predicted | `cohesive` | `dispersive` | `noncohesive` | **Total True** | **Class Recall** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`cohesive`** | **40** | 0 | 1 | 41 | **97.56%** |
| **`dispersive`** | 0 | **75** | 0 | 75 | **100.00%** |
| **`noncohesive`** | 0 | 4 | **86** | 90 | **95.56%** |
| **Total Predicted** | **40** | **79** | **87** | **206** | — |
| **Class Precision** | **100.00%** | **94.94%** | **98.85%** | — | **Overall: 97.57%** |

### Key Diagnostic Takeaways:
1. **Zero Missed Dispersive Infiltrates:** Dispersive achieved **100.00% Recall (75 out of 75 correct)** with **0 false negatives**.
2. **Cohesive Precision:** 100.00% Precision (0 non-cohesive or dispersive tiles were falsely called cohesive).
3. **Noncohesive Performance:** 86 out of 90 correct (95.56% Recall, 98.85% Precision).

---

## 4. Dataset Composition (`D:\new_whip_4321\`)

Constructed strictly from verified raw tiles without `no_feature` noise:

- **`cohesive`:** **250** raw tiles *(stratified random sample)*
- **`dispersive`:** **450** tiles *(265 raw tiles + 185 D4 rotations)*
- **`noncohesive`:** **536** raw tiles *(100% original raw tiles)*
- **Train Split (83.3%):** 1,030 tiles (`train.csv`)
- **Validation Split (16.7%):** 206 tiles (`val.csv`)

---

## 5. Local On-Disk Paths

- **Trained Model Weights:**
  - `D:\Project\model4_3_2_1_retrained\weights\model4_3_2_1_best.pt`
  - `C:\Users\anike\Desktop\generator with annotator\model4\weights\model4_3_2_1_best.pt`
  - `D:\generator with annotator\model4\weights\model4_3_2_1_best.pt`
- **Confusion Matrix Graphic:**
  - `D:\Project\model4_3_2_1_retrained\confusion_matrix_model4_3_2_1.png`
- **Sorted Predictions Audit CSV:**
  - `D:\Project\model4_3_2_1_retrained\val_predictions_sorted.csv`

---

## 6. How Your Other Machine / Pipeline Can Pull This

From your other machine or downstream pipeline:

```bash
git pull origin main
git lfs pull
```

The weights will download directly to `model4_3_2_1_retrained/weights/model4_3_2_1_best.pt`.
