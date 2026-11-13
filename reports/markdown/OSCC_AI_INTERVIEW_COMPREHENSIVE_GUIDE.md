# OSCC Histopathology AI: Comprehensive Technical Interview Master Guide

> **Prepared for Technical Defense & Pathology AI Interview**  
> **Project**: Multi-Scale Deep Learning for Invasive Pattern Classification in Oral Squamous Cell Carcinoma (OSCC)  
> **Flagship Systems**: Model 4 (Cascaded Specialist), Model 4.2 (Sub-phenotype Classifier), Model 4.3 (Retrained 90.97% Champion)

---

## Executive Summary & Clinical Problem Statement

In Oral Squamous Cell Carcinoma (OSCC), the **Worst Pattern of Invasion (WPOI)** at the invasive tumor front is one of the strongest independent prognosticators of cervical lymph node metastasis, recurrence, and overall survival:
- **`cohesive` (WPOI 1–3)**: Broad pushing borders or large solid invasive tumor islands (>50 cells). Associated with better surgical margins and lower metastatic risk.
- **`noncohesive` (WPOI 4)**: Small detached tumor clusters (15–50 cells) infiltrating stroma. Represents active epithelial-to-mesenchymal transition (EMT) and loss of E-cadherin adhesion.
- **`dispersive` (WPOI 5)**: Solitary infiltrating malignant cells or tiny linear cords (<15 cells) advancing $\ge 1\text{ mm}$ ahead of the tumor mass. Represents highly aggressive, high-risk disease.
- **`no_feature`**: Non-tumorous desmoplastic stroma, keratin pearls, acellular collagen, or necrotic tissue.

**The Core Machine Learning Challenge**:  
Single-cell infiltrates (`dispersive`) and small detached clusters (`noncohesive`) are minuscule relative to whole-slide images (WSIs). Standard single-scale CNNs or global ViTs fail because downsampling 512×512 tiles destroys single-cell nuclei, while naive pooling averages out detached borders into solid sheets.

---

## 1. Evolution of the Architectures: Why Some Failed & Others Succeeded

```
  ┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────────────┐
  │   Model 1    │     │   Model 2    │     │    Model 3     │     │   Model 4 / 4.3       │
  │ Baseline ViT │ ──> │ ViT + Morpho │ ──> │   HybridPath   │ ──> │ Multi-Scale Gated     │
  │ Linear Probe │     │ Feature Conc │     │ End-to-End Gate│     │ Specialist (Flagship) │
  │ F1: 0.6974   │     │ F1: 0.7424   │     │ F1: 0.7587     │     │ F1: 0.8321 -> 0.9000  │
  └──────────────┘     └──────────────┘     └────────────────┘     └───────────────────────┘
```

### Complete Comparative Matrix

| Model Identifier | Core Architectural Concept | Macro-F1 | Accuracy | Key Shortcoming / Bottleneck | Why It Was Replaced |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Model 1** | Baseline Frozen Phikon ViT + Linear Probe | 0.6974 | 74.9% | Weak on minority classes (`noncohesive` F1: 0.4946). Generic TCGA embeddings lack fine-grained OSCC specificity. | Linear head cannot model complex non-linear cell detachment boundaries. |
| **Model 2** | Frozen Phikon (1,536) + 10 Nuclear Morphometrics (10) | 0.7424 | 79.8% | Extracted features were static; no gradients backpropagated into the visual representation. | Handcrafted nuclear heuristics could not generalize across variable staining/illumination. |
| **Model 3 (HybridPath)** | 3-Branch Gated Network (Frozen ViT + Adaptive ViT + TileCNN) | 0.7587 | 80.9% | Severe minority class starvation (`dispersive`: 181 tiles, `noncohesive`: 278 tiles). `noncohesive` misclassified as `cohesive` (F1: 0.5294). | Network architecture was solid, but dataset imbalance and crop resolution starved the specialist branches. |
| **Model 4** | 2-Stage Hierarchical Cascade Network (Stage 1 Screener + Stage 2 Specialist) | 0.8321 | 87.6% | Stage 1 binary screener saturated on missing nuclear features, causing cascade lockout during naive inference. | Split into two distinct checkpoints; required unified inference refactoring. |
| **Model 5** | Recurrent Feedback Looping Network (State Persistence) | 0.8179 | 87.0% | Reached 100% precision on `dispersive`, but high architectural complexity increased latency (2.4x) without beating Model 4 F1. | Excessive recurrent parameter overhead with diminishing returns over feed-forward gating. |
| **Model 6** | 5-Stage Waterfall Early-Exit Cascade | 0.8250 | 86.8% | Sequential thresholds (0.70 per class) caused cascading threshold sensitivity and error propagation. | Multi-threshold calibration is fragile across unseen hospital slide scanners. |
| **Model 4.2** | 3-Branch Gated Sub-classifier for Non-Cancerous Tissues | N/A | High | Built specifically to sub-classify the seniors' normal tissue subset (`no_feature` decomposition into 4 sub-phenotypes). | Specialized for benign/normal profiling rather than invasive tumor front grading. |
| **Model 4.3 (Champion)** | Retrained 3-Branch Gated Specialist + D4 Dihedral Augmentation (4,986 tiles) | **0.9000** | **90.97%** | **Flagship Model**: Fixed patch geometry (128x128 -> 224x224), eliminated Stage 1 lockout, D4 dihedral balanced. | **Current Production Standard.** |

---

## 2. Foundations: Vision Transformers (ViT), Foundation Models & Transformers vs CNNs

### A. Vision Transformers (ViT) Mechanics
Standard Vision Transformers (Dosovitskiy et al.) process 2D images as sequences of 1D patch tokens:
1. **Patch Extraction & Projection**: An input crop $X \in \mathbb{R}^{H \times W \times C}$ (e.g., $224 \times 224 \times 3$) is divided into non-overlapping patches of size $P \times P$ ($16 \times 16$).
   - Number of patches $N = \frac{HW}{P^2} = \frac{224 \times 224}{16 \times 16} = 14 \times 14 = 196$ tokens.
   - Each patch is flattened into $\mathbb{R}^{P^2 C} = \mathbb{R}^{768}$ and linearly projected via trainable weight matrix $E \in \mathbb{R}^{(P^2 C) \times D}$ ($D = 768$).
2. **Prepend `[CLS]` Token & Add Position Embeddings**:
   - A learnable classification token $x_{\text{class}} \in \mathbb{R}^D$ is prepended: $Z_0 = [x_{\text{class}}; x_1 E; x_2 E; \dots; x_N E] \in \mathbb{R}^{197 \times 768}$.
   - 1D Learnable Position Embeddings $E_{\text{pos}} \in \mathbb{R}^{197 \times 768}$ are added to retain 2D spatial arrangement: $Z_0 = Z_0 + E_{\text{pos}}$.
3. **Multi-Head Self-Attention (MSA)**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   For 12 heads ($d_k = 64$), each head computes pairwise correlations between all 196 image patches simultaneously, creating a **global receptive field from layer 1**.

### B. What is Phikon (`owkin/phikon`)?
- **Architecture**: ViT-Base/16 (12 Transformer blocks, 768 hidden dimension, 12 attention heads, ~86M parameters).
- **Pre-training Paradigm**: Self-supervised learning using **iBOT** (masked image modeling with student-teacher knowledge distillation) on **40+ million pan-cancer histopathology tiles** from The Cancer Genome Atlas (TCGA).
- **Why Phikon over ImageNet ViT?**: ImageNet models learn natural scene priors (dogs, cars, landscapes, edges) with strong bias toward color and high-frequency silhouettes. Phikon learns **chromatin coarseness, nuclear pleomorphism, glandular architectures, and hematoxylin-eosin (H&E) staining invariants**.
- **What is C-TransPath?**: A competing pathology foundation model using a hybrid CNN-Swin Transformer pre-trained with semantically-relevant contrastive learning (SRCL). While C-TransPath is powerful, Phikon's uncompressed 16x16 standard ViT patches allow direct CLS extraction across high-magnification multi-crop grids without window-shifting artifacts.

### C. Why CNN vs RNN for Histopathology?

```
Feature                  CNN (Convolutional Neural Net)        RNN / LSTM / GRU
--------------------------------------------------------------------------------------------------
Spatial Bias             2D Inductive Bias (translation        1D Sequential Bias (temporal order)
                         equivariance, local locality)
Histological Fit         Matches tissue biology: cells cluster  Imposes arbitrary 1D raster scan
                         in 2D space without temporal order    (left-to-right / top-to-bottom)
Receptive Field          Hierarchical: edges -> nuclei ->      Linear sequential accumulation
                         glands -> tumor sheets
Parallelization          High: O(1) time across spatial grid   Low: O(T) sequential recurrence
Vanishing Gradients      Controlled via BatchNorm & ResNet     Severe over 262,144 pixel sequences
```
- **Interview Key Point**: "Histopathology images possess no temporal dimension or native reading order. Treating a 512x512 tile as an RNN sequence forces an artificial sequential bias and breaks rotational invariance. CNNs, by contrast, exploit **translation equivariance** ($f(g * x) = g * f(x)$) and **local weight sharing**, which directly models localized cell clusters regardless of where they appear on the glass slide."

---

## 3. Flagship Architecture: Model 4 / 4.3 Deep Technical Blueprint

Model 4 / 4.3 uses a **Tri-Branch Multi-Scale Gated Network** that unifies microscopic nuclear detail with macroscopic global tissue architecture:

```
                                Input Tile (512 x 512 x 3)
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
        16 Uniform Crops                           Resized Whole Tile
      (128x128 -> 224x224)                            (256 x 256)
                 │                                         │
       ┌─────────┴─────────┐                               │
       │                   │                               │
       ▼                   ▼                               ▼
 ┌───────────┐       ┌───────────┐                  ┌─────────────┐
 │ Branch A  │       │ Branch B  │                  │  Branch C   │
 │  Phikon   │       │  Phikon   │                  │   TileCNN   │
 │ (Frozen)  │       │(Unfrozen) │                  │(From Scratch│
 └─────┬─────┘       └─────┬─────┘                  └──────┬──────┘
       │                   │                               │
 [16, 768] CLS       [16, 768] CLS                   7 Conv Layers
       │                   │                               │
  Dual Pooling        Dual Pooling                 AdaptiveAvgPool
 (Mean + Max)        (Mean + Max)                          │
       │                   │                               ▼
   [B, 1536]           [B, 1536]                        [B, 512]
       │                   │                               │
   Linear/GELU         Linear/GELU                         │
       │                   │                               │
       ▼                   ▼                               │
    P_a [512]           P_b [512]                       P_c [512]
       │                   │                               │
       └─────────┬─────────┴───────────────┬───────────────┘
                 │                         │
                 ▼                         ▼
         Dynamic Gating               Gated Fusion
        Linear(3584->128)      F_fused = w_a*P_a + w_b*P_b + w_c*P_c
        Softmax([w_a,w_b,w_c])                 │
                 │                             ▼
                 └──────────────────> Classification Head
                                     Linear(512->256) -> GELU -> Dropout
                                     Linear(256->128) -> GELU -> Dropout
                                     Linear(128->4)
                                               │
                                               ▼
                                      4-Class Predictions
```

### A. Multi-Crop High-Resolution Geometry
- **Input Tile**: $512 \times 512 \times 3$ at $40\times$ optical magnification (approx. $0.25\ \mu\text{m/pixel}$).
- **The Problem**: Resizing a $512 \times 512$ tile directly down to $224 \times 224$ reduces pixel area by 81%, blurring solitary malignant cell nuclei (which are only 10–15 $\mu\text{m}$ in diameter).
- **The Solution (`make_16_crops_correct`)**:
  - The tile is divided into a $4 \times 4$ grid of 16 sub-patches, each $128 \times 128$ pixels.
  - Each $128 \times 128$ patch is upsampled via **bilinear interpolation** to $224 \times 224$.
  - Output tensor shape: `[B, 16, 3, 224, 224]`.
  - **Effect**: Vision Transformer patches receive an effective optical zoom, enabling the self-attention heads to resolve individual chromatin borders and nuclear membranes.

### B. Branch A: Frozen Foundation Anchor
- **Model**: `ViTModel.from_pretrained("owkin/phikon")`, 100% frozen (`param.requires_grad = False`).
- **Input**: 16 crops `[B * 16, 3, 224, 224]`.
- **Feature Extraction**:
  - Extracts the `[CLS]` token from the 12th layer: `[B * 16, 768]`.
  - Reshaped to `[B, 16, 768]`.
- **Dual-Pooling Aggregation**:
  $$F_{\text{mean}} = \frac{1}{16} \sum_{i=1}^{16} \text{CLS}_i \in [B, 768], \quad F_{\text{max}} = \max_{i=1}^{16}(\text{CLS}_i) \in [B, 768]$$
  $$F_a = [F_{\text{mean}} \parallel F_{\text{max}}] \in [B, 1536]$$
- **Why Dual-Pooling?**:
  - $\text{Mean}$ captures the prevailing tissue background (stroma density, tumor cellularity).
  - $\text{Max}$ preserves the salient outlier (e.g., a single isolated malignant cell in crop #14 that would be diluted by pure mean pooling).
- **Projection**: `Linear(1536 -> 512) -> GELU()` $\rightarrow P_a \in [B, 512]$.
- **Role**: Invariant foundation anchor. Prevents catastrophic forgetting and maintains broad histopathological generalization.

### C. Branch B: Partially Unfrozen Adaptive ViT
- **Model**: `ViTModel.from_pretrained("owkin/phikon")`.
- **Unfreezing Strategy**:
  - Transformer Blocks 0 to 8: **100% Frozen**.
  - Transformer Blocks 9, 10, 11 + Final LayerNorm: **Unfrozen (`requires_grad = True`)**.
- **Learning Rate**: Fine-tuned at $1 \times 10^{-5}$ (10x smaller than head LR).
- **Dual-Pooling**: Identical mean + max aggregation yielding $F_b \in [B, 1536] \rightarrow$ Projected to $P_b \in [B, 512]$.
- **Why Only the Last 3 Blocks?**:
  - Early layers (0–4) capture basic edges, textures, and color gradients.
  - Middle layers (5–8) capture general cellular motifs.
  - Deep layers (9–11) represent high-level semantic abstractions. Unfreezing only the top 3 blocks allows the self-attention heads to adapt specifically to **OSCC infiltrative discohesion** without destabilizing the foundation representations.

### D. Branch C: Custom TileCNN (Architectural Breakdown)
Processes the entire downsampled $256 \times 256 \times 3$ tile from scratch to capture macro-architecture (tumor nests, pushing borders, invasive margins):

```
Layer                  Input Shape       Output Shape      Kernel / Stride / Pad   Parameters / Notes
--------------------------------------------------------------------------------------------------------
Input                  -                 [B, 3, 256, 256]  -                       Raw RGB Tile (256x256)
Conv2d_1               [B, 3, 256, 256]  [B, 32, 256, 256] 3x3, stride=1, pad=1   BatchNorm2d(32) + GELU()
Conv2d_2               [B, 32, 256, 256] [B, 32, 256, 256] 3x3, stride=1, pad=1   BatchNorm2d(32) + GELU()
MaxPool2d_1            [B, 32, 256, 256] [B, 32, 128, 128] 2x2, stride=2          Downsample 2x
Conv2d_3               [B, 32, 128, 128] [B, 64, 128, 128] 3x3, stride=1, pad=1   BatchNorm2d(64) + GELU()
Conv2d_4               [B, 64, 128, 128] [B, 64, 128, 128] 3x3, stride=1, pad=1   BatchNorm2d(64) + GELU()
MaxPool2d_2            [B, 64, 128, 128] [B, 64, 64, 64]   2x2, stride=2          Downsample 2x
Conv2d_5 (Large)       [B, 64, 64, 64]   [B, 128, 64, 64]  5x5, stride=1, pad=2   5x5 Kernel captures broader stroma
Conv2d_6               [B, 128, 64, 64]  [B, 128, 64, 64]  3x3, stride=1, pad=1   BatchNorm2d(128) + GELU()
MaxPool2d_3            [B, 128, 64, 64]  [B, 128, 32, 32]  2x2, stride=2          Downsample 2x
Conv2d_7               [B, 128, 32, 32]  [B, 256, 32, 32]  3x3, stride=1, pad=1   BatchNorm2d(256) + GELU()
MaxPool2d_4            [B, 256, 32, 32]  [B, 256, 16, 16]  2x2, stride=2          Downsample 2x
AdaptiveAvgPool2d      [B, 256, 16, 16]  [B, 256, 4, 4]    Output size (4, 4)     Spatial condensation
Flatten                [B, 256, 4, 4]    [B, 4096]         -                      256 * 4 * 4 = 4096 features
Linear_Proj            [B, 4096]         [B, 512]          -                      Linear + GELU() + Dropout(0.3)
```

- **Why a 5x5 Kernel in Block 3?**: Most CNNs use strictly $3 \times 3$ stacks. In histopathology, desmoplastic stroma and keratin pearls span larger spatial fields. A $5 \times 5$ kernel (effective receptive field $5 \times 5$) accelerates spatial context aggregation before downsampling.

### E. Dynamic Gating Network & Fusion
Instead of static concatenation or equal-weight averaging, the network dynamically computes branch reliability per tile:
1. **Gate Input**: Concatenation of raw feature vectors:
   $$X_{\text{gate}} = [F_a (1536) \parallel F_b (1536) \parallel F_c (512)] \in \mathbb{R}^{B \times 3584}$$
2. **Gating Mechanism**:
   $$\text{Gate}(X_{\text{gate}}) = \text{Softmax}\Big(\mathbf{W}_2 \cdot \text{Dropout}\big(\text{GELU}(\mathbf{W}_1 X_{\text{gate}} + \mathbf{b}_1)\big) + \mathbf{b}_2\Big)$$
   - $\mathbf{W}_1 \in \mathbb{R}^{3584 \times 128}$, $\mathbf{W}_2 \in \mathbb{R}^{128 \times 3}$.
   - Output: Normalized weights $[w_a, w_b, w_c]$ such that $w_a + w_b + w_c = 1.0$.
3. **Convex Combination Fusion**:
   $$F_{\text{fused}} = w_a \cdot P_a + w_b \cdot P_b + w_c \cdot P_c \quad \in \mathbb{R}^{B \times 512}$$
4. **Behavior**:
   - For **large continuous cohesive sheets**: $w_c$ (TileCNN) receives high weight because macro-scale architecture is sufficient.
   - For **solitary dispersive cells**: $w_a$ and $w_b$ (Phikon ViT crops) dominate because single-cell nuclear details govern the diagnosis.

### F. Classification Head & Regularization
$$F_{\text{fused}} \in \mathbb{R}^{B \times 512} \xrightarrow{\text{Linear}} [B, 256] \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.3)} [B, 256] \xrightarrow{\text{Linear}} [B, 128] \xrightarrow{\text{GELU}} \xrightarrow{\text{Dropout}(0.2)} [B, 128] \xrightarrow{\text{Linear}} [B, 4]$$
- **Loss Function**: Weighted Cross-Entropy Loss with inverse class frequency weights:
  $$\mathcal{L} = -\sum_{c=1}^{4} \alpha_c y_c \log(\hat{y}_c), \quad \alpha_c = \frac{N_{\text{total}}}{4 \cdot N_c}$$

---

## 4. Model 4.2 vs Model 4.3: Purpose, Data & Design

### Model 4.2: The Sub-Phenotype Normal Tissue Classifier
- **Target Context**: Developed to address the `no_feature` class. While cancer tiles exhibit invasion, normal patient tissues (annotated by senior pathologists) contained rich non-cancerous histological sub-types.
- **Role**: Operates downstream of Model 4. When a tile is classified as `no_feature`, Model 4.2 sub-classifies it into 4 distinct histological categories (e.g., normal epithelium, deep muscle, salivary glands, dense fibrous connective tissue).
- **15% Augmentation Rule**: Regulates imbalance so no class exceeds $0.85 \times N_{\text{max}}$, preventing synthetic artifacts in clean normal histology.

### Model 4.3: The Retrained Champion Classifier
- **Target Context**: The primary 4-class cancer invasion model. Retrained directly on the balanced 4,986 tiles (`D:\new whip`) after resolving the crop geometry and cascade bugs.
- **Validation Metrics (831 Tiles)**:
  - **Accuracy**: **90.97%** (756 / 831 correct)
  - **Macro-F1**: **0.9000**
  - **Per-Class Breakdown**:
    - `no_feature`: **0.9596 F1** (Precision: 96.1%, Recall: 95.8%)
    - `cohesive`: **0.8948 F1** (Precision: 84.2%, Recall: 95.4%)
    - `dispersive`: **0.8915 F1** (Precision: 92.0%, Recall: 86.5%)
    - `noncohesive`: **0.8540 F1** (Precision: 94.4%, Recall: 78.0%)
- **Early Stopping**: Strict `patience = 3` on Validation Macro-F1. Reached peak at Epoch 3 and halted at Epoch 6.

---

## 5. Critical Engineering Debugging: What Failed & How It Was Fixed

### Bug 1: The Crop Geometry Coordinate Bug
- **The Defect**: In the original predictor, `make_16_crops` extracted coordinate slices:
  `[0, 96, 192, 288]` directly as $224 \times 224$ slices from a $512 \times 512$ tile.
- **Why It Failed**: Overlapping raw slices distorted the physical pixel spacing. In Vision Transformers, the receptive field depends on scale: a $128 \times 128$ patch resized to $224 \times 224$ provides an optical magnification effect. Raw slicing forced the model to evaluate detached cells at the wrong magnification, degrading `noncohesive` and `dispersive` detection.
- **The Fix**: Implemented `make_16_crops_correct` with strict $4 \times 4$ grid tiling ($128 \times 128$ per patch) followed by bilinear interpolation to $224 \times 224$.

### Bug 2: The Stage 1 Binary Cascade Lockout
- **The Defect**: Model 4 was originally structured as a 2-stage cascade where Stage 1 screened `cohesive` vs `no_feature`. When inference was run without nuclear morphometric tabular features, dummy zero vectors (`named_zero = torch.zeros(...)`) were passed.
- **Why It Failed**: The binary classifier saturated ($P(\text{cohesive}) \ge 0.75$ or $P(\text{no\_feature}) \ge 0.75$). The cascade threshold logic intercepted **100% of tiles** at Stage 1, completely bypassing Stage 2 (the only stage capable of recognizing `dispersive` and `noncohesive`).
- **The Fix**: Refactored `Model4Predictor` so that inference routes directly through the 4-class specialist network (`Stage2Specialist4Class`), evaluating all 4 classes simultaneously and eliminating the lockout.

### Bug 3: Why D4 Dihedral Rotations Beat SMOTE
- **Why SMOTE Failed**: Synthetic Minority Over-sampling Technique (SMOTE) performs linear interpolation between feature vectors:
  $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \in [0, 1]$$
  In histopathology image space or deep feature space, interpolating between two cell images creates unnatural, blurry "ghost cells" and non-physical staining artifacts that violate biological reality.
- **Why D4 Dihedral Rotations Succeeded**: Histopathology glass slides possess **zero native orientation** (a tissue section looks identical under $90^\circ, 180^\circ, 270^\circ$ rotation or vertical/horizontal reflection). D4 transformations generate 8 genuine, physically valid views per tile, preserving true nuclear membranes and chromatin textures.

---

## 6. Interview Defense: Anticipated Technical Questions & Model Answers

### Q1: "Why did you use a hybrid CNN + ViT architecture instead of an end-to-end ViT alone?"
> **Answer**: "Vision Transformers excel at capturing long-range global dependencies through self-attention, but they lack the intrinsic 2D inductive bias (translation equivariance and localized receptive fields) of CNNs. In histopathology, classification requires both **cellular-level morphology** (chromatin texture, nuclear pleomorphism) and **tissue-level topology** (pushing margins, stromal reaction). Our Tri-Branch architecture assigns high-resolution multi-crop ViT branches to capture individual nuclear detail, while the TileCNN extracts holistic tissue architecture. Our Dynamic Gating Network then computes a data-dependent convex combination of these representations for each individual tile."

### Q2: "Why freeze Branch A completely while fine-tuning only the top 3 layers of Branch B?"
> **Answer**: "Completely fine-tuning an 86-million parameter ViT on a clinical dataset of ~5,000 tiles carries a severe risk of catastrophic forgetting and overfitting to our hospital's specific slide scanners and staining protocols. By keeping Branch A 100% frozen, it acts as an immutable anchor grounded in 40+ million pan-cancer TCGA tiles. In Branch B, by unfreezing only Transformer blocks 9–11 and LayerNorm, we preserve general low-level and mid-level cellular features while allowing the top self-attention heads to specialize in OSCC-specific discohesive infiltrates."

### Q3: "Why did you use dual pooling (Mean + Max) on the 16 crop embeddings instead of simple Mean pooling?"
> **Answer**: "Mean pooling computes the centroid of the crop tokens, which captures the general tissue background. However, the most clinically dangerous pattern—`dispersive` (WPOI 5)—often manifests as just one or two solitary infiltrating cancer cells located in a single corner crop. Mean pooling across 16 crops dilutes that solitary cell signal by a factor of 16. Max pooling, by contrast, captures the strongest activation across all crops. Concatenating Mean and Max pooling ($1536$ dimensions) ensures the classifier receives both the contextual stromal background and the isolated infiltrative outlier."

### Q4: "How did you prevent data leakage during your validation and testing?"
> **Answer**: "Histopathology tiles extracted from the same whole-slide image share identical tissue preparation, microtome sectioning thickness, and H&E staining nuances. Randomly splitting tiles at the tile level causes severe data leakage and artificially inflates accuracy. We implemented **strict patient-level stratified splitting**: all tiles from any individual patient slide resided exclusively in either the training split or the validation split. Furthermore, all rotation augmentations were applied strictly to the training split after partitioning."

### Q5: "What is the clinical significance of your model distinguishing `dispersive` vs `noncohesive`?"
> **Answer**: "In the Brandwein-Gensler and WHO histological grading systems, distinguishing between cohesive sheets (WPOI 1–3) and noncohesive clusters (WPOI 4) vs solitary dispersive cells (WPOI 5) dictates whether a patient undergoes elective neck dissection (END). Dispersive single-cell invasion indicates complete loss of E-cadherin and high lymphovascular invasion propensity, with occult metastasis rates exceeding 30%. Reaching an **F1 score of 0.8915 for dispersive** and **0.8540 for noncohesive** allows pathologists to reliably identify micro-invasive patterns that are easily missed during manual microscopic screening."

---

## 7. Quick Technical Cheat Sheet (Formula & Parameter Summary)

- **Tile Dimensions**: $512 \times 512 \times 3$ at $40\times$ magnification ($0.25\ \mu\text{m/px}$).
- **Crop Geometry**: $4 \times 4$ uniform grid $\rightarrow 16$ crops of $128 \times 128 \rightarrow$ resized to $224 \times 224$.
- **Phikon Backbone**: ViT-B/16 (12 layers, 768 hidden dim, 12 heads, iBOT self-supervised on TCGA).
- **Branch A Dimension**: $768 \times 2 (\text{mean} \parallel \text{max}) = 1536 \rightarrow$ Projected to $512$ with GELU.
- **Branch B Dimension**: $768 \times 2 (\text{mean} \parallel \text{max}) = 1536 \rightarrow$ Projected to $512$ with GELU (Layers 9–11 fine-tuned).
- **Branch C (TileCNN)**: 7 Conv layers ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$), $5\times 5$ kernel in Stage 3, AdaptiveAvgPool(4,4) $\rightarrow$ Projected to $512$.
- **Gating Network Input**: $1536 + 1536 + 512 = 3584$ dimensions $\rightarrow$ Linear(128) $\rightarrow$ Softmax(3) $\rightarrow$ Weights $[w_a, w_b, w_c]$.
- **Final Classification Head**: $512 \rightarrow 256 \rightarrow 128 \rightarrow 4$ logits with Dropout(0.3, 0.2) and GELU.
- **Champion Checkpoint**: [`model4_3_best.pt`](file:///D:/Project/model4_3_retrained/weights/model4_3_best.pt) (**90.97% Accuracy, 0.9000 Macro-F1**).

