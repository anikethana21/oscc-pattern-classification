# Model 5: Recurrent Feedback / Looping Dual-Stage Network

## Overview
Model 5 implements an iterative loop connection between two network modules:
1. **Module 1 (First Two Classes)**: Evaluates input tiles for `cohesive` and `no_feature`.
2. **Module 2 (Specialist Network)**: Focuses heavily on the remaining two classes (`dispersive` and `noncohesive`) using multi-branch deep gating.
3. **Loop Connection**: The latent state of Module 2 is fed back into Module 1 via a recurrent gating mechanism.
4. **Final Forcing Classifier**: A unified head takes the looped representation and forces classification of any tile into one of the 4 classes.

## Scripts & Usage

### 1. Training Model 5
```powershell
cd D:\Project
python model5_loop\m5_train.py --epochs 35 --batch-size 4 --unfreeze 3 --patience 8
```
- Saved weights: `D:\Project\model5_loop\weights\model5_best.pt`

### 2. Evaluation on Locked Test Set
```powershell
python model5_loop\m5_eval.py
```
- Output confusion matrix: `D:\Project\model5_loop\m5_confusion_matrix.png`
- Output predictions: `D:\Project\model5_loop\m5_test_predictions.csv`
