# Model 4: Hierarchical Cascaded Classification Network

## Overview & Architecture Concept
Model 4 combines the strengths of **Model 2** and **Model 3 (HybridPath)** into a high-performance two-stage hierarchical cascade:

```
                         Input Image Tile (512x512)
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │       STAGE 1 CLASSIFIER          │
                   │   Phikon (Frozen) + Named Feats   │
                   └─────────────────┬─────────────────┘
                                     │
                    P(cohesive) + P(no_feature) >= 0.60?
                                    ╱ ╲
                             YES   ╱   ╲   NO (Ambiguous / Infiltrative)
                                  ╱     ╲
                                 ▼       ▼
               ┌─────────────────────┐  ┌───────────────────────────────────┐
               │ Confident Screening │  │        STAGE 2 SPECIALIST         │
               │ cohesive / no_feature│  │      HybridPath Architecture:     │
               └─────────────────────┘  │  - Branch A: Frozen Phikon        │
                                        │  - Branch B: Unfrozen Phikon (9-11)│
                                        │  - Branch C: Custom Tile CNN      │
                                        │  - Dynamic Gating Network         │
                                        │  => Binary: dispersive / noncohesive
                                        └─────────────────┬─────────────────┘
                                                          │
                                                          ▼
                                            High-Sensitivity Scoring for
                                              dispersive & noncohesive
```

---

## Dataset Scaling (~2,500 Tiles Per Class)
Using `m4_augment.py`, the training split (`split_train.csv`) is expanded to **~2,500 tiles per class** (~10,000 total tiles):
- **Dispersive & Noncohesive** (minority classes): Receive heavy targeted augmentation (rotation, elastic deformation, grid distortion, color jitter, blur, zoom/crop).
- **Cohesive & No Feature**: Receive balanced light geometric/photometric augmentation.

---

## Execution Guide

### Step 1: Generate Augmented Dataset (~2,500 Tiles / Class)
```powershell
cd D:\Project
python model4_cascade\m4_augment.py --target 2500
```
This generates `D:\Project\model4_cascade\m4_train.csv` and augmented tiles in `D:\Project\model4_cascade\augmented_tiles\`.

---

### Step 2: Train Stage 1 (Screening Classifier)
```powershell
cd D:\Project
python model4_cascade\m4_train_stage1.py --epochs 40 --batch-size 8 --lr 3e-4 --patience 10
```
- Saved weights: `D:\Project\model4_cascade\weights_s1\stage1_best.pt`
- Training history: `D:\Project\model4_cascade\weights_s1\training_log.csv`

---

### Step 3: Train Stage 2 (Dispersive vs Noncohesive Specialist)
```powershell
cd D:\Project
python model4_cascade\m4_train_stage2.py --epochs 30 --batch-size 4 --unfreeze 3 --patience 8
```
- Saved weights: `D:\Project\model4_cascade\weights_s2\stage2_best.pt`
- Training history: `D:\Project\model4_cascade\weights_s2\training_log.csv`

---

### Step 4: Evaluate Locked Test Set & Generate Confusion Matrix
```powershell
cd D:\Project
python model4_cascade\m4_eval_cascade.py --threshold 0.60
```
Outputs generated in `D:\Project\model4_cascade\`:
- `m4_confusion_matrix.png`: Full 4-class confusion matrix of the cascade.
- `m4_test_predictions.csv`: Per-tile predictions, true labels, and routing flags.
- Comprehensive classification report printed to console.
