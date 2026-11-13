# OSCC Histopathology AI Master Architecture Guide

> **Interactive Visual Document**: Open [`MASTER_STUDY_GUIDE.html`](file:///D:/Project/MASTER_STUDY_GUIDE.html) in your browser for the full interactive visual flowcharts, styled cards, and embedded high-resolution figures.

---

## 1. Dataset Provenance & Patient-Stratified Split Strategy

### Data Sources
- **Optical Magnification**: 40x
- **Acquisition Batches**: 6 source archives (`40x-001` through `40x-006`)
- **Dual-Annotator Strategy**: Two independent student annotators/pathologists working on distinct, non-overlapping tissue fields.
- **Master Manifest**: 2,661 unique, non-duplicate, conflict-free tiles.

### Honest Patient-Level Split Breakdown (No Patient Leakage)

| Class Name | Clinical Phenotype | Full Dataset | Train Split (80%) | Val Split (10%) | Locked Test Split (10%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`cohesive`** | Large solid tumor sheets (>50 cells) | 1,550 | 1,240 | 155 | 155 |
| **`no_feature`** | Acellular stroma, collagen, background | 646 | 516 | 65 | 65 |
| **`noncohesive`**| Intermediate detached clusters (15–50 cells) | 278 | 222 | 27 | 29 |
| **`dispersive`**  | Solitary cells & cords (<15 cells) — High Risk | 187 | 150 | 19 | 18 |
| **TOTAL** | Non-Duplicate Curated Tiles | **2,661** | **2,128** | **266** | **267** |

### Dataset Scaling to ~2,500 Tiles / Class (~10,000 Total Tiles)
- Used in **Model 4** and **Model 5**.
- Generated via `m4_augment.py` on `split_train.csv`.
- Scaled each class to exactly 2,500 tiles.
- Applied targeted geometric transformations (RandomRotate90, ElasticTransform, GridDistortion) and color perturbations (HueSaturation, GaussianNoise, ScaleJitter) heavily to `dispersive` and `noncohesive`.
- Validation and Locked Test splits were **never augmented**.

---

## 2. Complete Model Benchmark Comparison (Locked Test Set, N=267 Tiles)

| Model | Architecture Description | Macro-F1 | Overall Accuracy | Cohesive F1 | Dispersive F1 | No Feature F1 | Noncohesive F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | Baseline Phikon Deep Features (Linear Probe) | 0.6974 | 74.9% | 0.8654 | 0.6452 | 0.7846 | 0.4946 |
| **Model 2** | Phikon Deep (1536) + 10 Named Morphological (10) | 0.7424 | 79.8% | 0.8800 | 0.7368 | 0.8125 | 0.5405 |
| **Model 3** | HybridPath (3-Branch End-to-End Dynamic Gating) | 0.7587 | 80.9% | 0.8874 | 0.8000 | 0.8182 | 0.5294 |
| **Model 4** | **Specialist Gated Network (~2.5k Scaled Augmentation)** | **0.8321** | **87.6%** | 0.9024 | **0.8485** | **0.8819** | **0.6957** |
| **Model 5** | **Recurrent Feedback Looping Dual-Stage Network** | **0.8179** | **87.0%** | **0.9040** | **0.8387** *(100% Prec)* | 0.8750 | 0.6538 |

![All Models Confusion Matrix](all_5_models_confusion_matrix.png)

---

## 3. Step-by-Step Tensor Flow & Activation Functions

### A. Pre-Processing & Feature Extractors (Shared Across Models 3, 4, 5)

1. **Input Tile**: Shape `[B, 3, 512, 512]`
2. **Branch A (Phikon Frozen)**:
   - Input: 16 crops of `[B, 16, 3, 224, 224]`
   - Backbone: 12-layer Vision Transformer (`owkin/phikon`), 100% frozen.
   - Output CLS token per crop: `[B, 16, 768]`
   - Dual Pooling: `mean(dim=1)` concat with `max(dim=1)` $\rightarrow$ `[B, 1536]`
   - Linear Projection: `Linear(1536 -> 512)` with `GELU` activation $\rightarrow$ $P_a \in [B, 512]$
3. **Branch B (Phikon Adaptive)**:
   - Input: 16 crops of `[B, 16, 3, 224, 224]`
   - Backbone: Blocks 0–8 frozen; Blocks 9, 10, 11 + LayerNorm fine-tuned ($LR = 10^{-5}$)
   - Dual Pooling: `mean(dim=1)` concat with `max(dim=1)` $\rightarrow$ `[B, 1536]`
   - Linear Projection: `Linear(1536 -> 512)` with `GELU` activation $\rightarrow$ $P_b \in [B, 512]$
4. **Branch C (TileCNN from Scratch)**:
   - Input: Resized tile `[B, 3, 256, 256]`
   - Stage 1: `Conv2D(3->32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `Conv2D(32->32)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `MaxPool(2x2)`
   - Stage 2: `Conv2D(32->64)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `Conv2D(64->64)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `MaxPool(2x2)`
   - Stage 3: `Conv2D(64->128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `Conv2D(128->128)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `MaxPool(2x2)`
   - Stage 4: `Conv2D(128->256)` $\rightarrow$ `BatchNorm` $\rightarrow$ `GELU` $\rightarrow$ `MaxPool(2x2)`
   - Spatial Pooling: `AdaptiveAvgPool2D((4, 4))` $\rightarrow$ `[B, 256, 4, 4]`
   - Projection: `Flatten()` $\rightarrow$ `Linear(4096 -> 512)` with `GELU` and `Dropout(0.3)` $\rightarrow$ $P_c \in [B, 512]$

---

### B. Dynamic Gating Network (Model 3 & Model 4)

- **Input**: Concatenation of raw branch outputs `[F_a (1536), F_b (1536), F_c (512)]` $\rightarrow$ Shape `[B, 3584]`
- **Layers**:
  1. `Linear(3584 -> 128)`
  2. Activation: `GELU`
  3. Regularization: `Dropout(0.2)`
  4. `Linear(128 -> 3)`
  5. Activation: `Softmax(dim=-1)`
- **Output**: Branch weights $[w_a, w_b, w_c]$, where $w_a + w_b + w_c = 1.0$
- **Fused Feature**:
  $$F_{\text{fused}} = w_a \cdot P_a + w_b \cdot P_b + w_c \cdot P_c \quad \in [B, 512]$$
- **Classification Head**:
  1. `Linear(512 -> 256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)`
  2. `Linear(256 -> 128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.2)`
  3. `Linear(128 -> 4)` $\rightarrow$ Final Logits `[B, 4]`
- **Loss Function**: Focal Loss ($\gamma = 2.0$) with class balance weights.

---

### C. Recurrent Feedback Looping Network (Model 5)

Model 5 introduces a **bidirectional feedback loop** between two modules:

1. **Module 1 (Screener)**:
   - Input: Frozen Phikon `[B, 1536]`
   - Transformation: `Linear(1536 -> 512)` $\rightarrow$ `LayerNorm(512)` $\rightarrow$ `GELU`
   - Yields: Initial screener state $h_1 \in [B, 512]$ and auxiliary binary logits (cohesive vs no_feature).
2. **Module 2 (Specialist Network)**:
   - Input: Concat `[F_frozen, F_adapt, F_cnn, h1]` $\rightarrow$ `[B, 4096]`
   - Gating: Gating Network computes branch weights conditioned on $h_1$.
   - Fused Specialist State: $h_2 \in [B, 512]$
3. **Recurrent Feedback Connection**:
   - Feedback Transform: $fb = \text{GELU}(\text{LayerNorm}(\text{Linear}(h_2))) \in [B, 512]$
   - Recurrent Gate: $g = \text{Sigmoid}(\text{Linear}(\text{Concat}[h_1, fb])) \in [B, 512]$
   - Looped State:
     $$h_{\text{loop}} = g \odot h_1 + (1 - g) \odot fb \quad \in [B, 512]$$
4. **Final Forcing Classifier**:
   - `Linear(512 -> 256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.3)`
   - `Linear(256 -> 128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.2)`
   - `Linear(128 -> 4)` $\rightarrow$ Final 4-Class Logits `[B, 4]`
5. **Multi-Task Joint Loss**:
   $$\mathcal{L} = \text{FocalLoss}(\text{Final Logits}, Y; \gamma=2.0) + 0.3 \cdot \text{CrossEntropy}(\text{Auxiliary Logits}, Y_{\text{binary}})$$

---

## 4. Class-by-Class Failure Analysis

### The Easy Classes (>88% F1):
- **`cohesive` (90.4% F1)**: Massive solid tumor islands (>50 cells). ViT CLS tokens detect glandular sheets with 95.5% recall.
- **`no_feature` (88.2% F1)**: Acellular fibrous stroma, keratin pearls, necrosis. Low nuclei count and high pink stroma make this cleanly separable.

### The High-Precision Class:
- **`dispersive` (84.9% F1)**: Single malignant cells & small cords (<15 cells). In Model 5, the recurrent loop achieved **100.0% precision** (zero false alarms). Recall sits at ~72–78% because tiny single cells in corner crops occasionally get averaged out.

### The Bottleneck Class (65–69% F1):
- **`noncohesive` (69.6% F1)**: Small detached cell clusters (15 to 50 cells).
- **Why it Fails**: Out of 29 true noncohesive test tiles, **12 were misclassified as cohesive**.
- **Root Cause**: All models used `mean()` and `max()` spatial pooling over crops. This pooling averages out the empty space between clusters, making 15-cell clusters look like a continuous solid sheet.

---

---

---

## 5. Architectural Blueprint for Model 6: Step-by-Step Waterfall Early-Exit Cascade

Model 6 is organized as a sequential **step-by-step waterfall cascade** of specialized subnetworks. Unlike parallel heads that fire simultaneously, each stage acts as a dedicated sieve that filters out confident classifications before passing ambiguous features deeper into the network.

### Step-by-Step Waterfall Flow:

1. **Upper Trunk & Fusion**:
   - 16 Crops + Whole Tile $\\to$ 3 Encoders (Phikon A, Phikon B, TileCNN) $\\to$ Dynamic Gating $\\to$ **Fused Feature Vector (512 neurons)**.

2. **Stage 1 (Subnetwork 1 - 
o_feature)**:
   - Dense layers extract empty stroma/background characteristics.
   - Output: 1 Sigmoid neuron ({nf}$).
   - **Decision Gate**: If {nf} \\ge 0.70$ $\\implies$ **EXIT 1: 
o_feature**.
   - **Else ({nf} < 0.70$)**: Tile data flows down into **Network 2**.

3. **Stage 2 (Subnetwork 2 - dispersive)**:
   - Specialized dense layers evaluate isolated cell clusters and nuclear dispersion.
   - Output: 1 Sigmoid neuron ({dis}$).
   - **Decision Gate**: If {dis} \\ge 0.70$ $\\implies$ **EXIT 2: dispersive**.
   - **Else ({dis} < 0.70$)**: Tile data flows down into **Network 3**.

4. **Stage 3 (Subnetwork 3 - cohesive)**:
   - Evaluates solid continuous sheet architecture and tight cell-to-cell adhesion.
   - Output: 1 Sigmoid neuron ({coh}$).
   - **Decision Gate**: If {coh} \\ge 0.70$ $\\implies$ **EXIT 3: cohesive**.
   - **Else ({coh} < 0.70$)**: Tile data flows down into **Network 4**.

5. **Stage 4 (Subnetwork 4 - 
oncohesive)**:
   - Evaluates irregular jagged cords and detached infiltrative clusters.
   - Output: 1 Sigmoid neuron ({nc}$).
   - **Decision Gate**: If {nc} \\ge 0.70$ $\\implies$ **EXIT 4: 
oncohesive**.
   - **Else ({nc} < 0.70$)**: Tile data flows down into the **Final Concluder**.

6. **Stage 5 (Final Concluder Network)**:
   - Evaluates all tiles where no individual stage cleared the 0.70 confidence threshold.
   - Output: 4 neurons with **Softmax** activation over all 4 classes.
   - Final label = $\\operatorname{argmax}(c_{coh}, c_{dis}, c_{nf}, c_{nc})$.
