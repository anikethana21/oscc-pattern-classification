# Model 4.3.2 (Dual Phikon ViT + TileCNN Specialist)

## Data

| Class | Training data | Val / Test | Total |
| :--- | :---: | :---: | :---: |
| **Cohesive** | 2414 | 483 | 2897 |
| **Non cohesive** | 1379 | 276 | 1655 |
| **Dispersive** | 1226 | 245 | 1471 |
| **No feature** | 2622 | 525 | 3147 |
| **Total** | **7641** | **1529** | **9170** |

---

## Classification Report

```text
              precision    recall  f1-score   support

    cohesive     0.9399    0.9710    0.9552       483
  dispersive     0.9390    0.9429    0.9409       245
  no_feature     0.9794    0.9943    0.9868       525
non_cohesive     0.9602    0.8732    0.9146       276

    accuracy                         0.9568      1529
   macro avg     0.9546    0.9453    0.9494      1529
weighted avg     0.9570    0.9568    0.9564      1529
```

---

## Confusion Matrix

![Confusion Matrix - Validation (Model 4.3.2)](C:/Users/anike/.gemini/antigravity/brain/40d4b308-7194-4358-b8e7-0f6a076836ae/confusion_matrix_styled.png)

---

## ROC Curve

![ROC Curve (Model 4.3.2)](C:/Users/anike/.gemini/antigravity/brain/40d4b308-7194-4358-b8e7-0f6a076836ae/roc_curve_model4_3_2.png)

### Area Under the Curve (AUC) Summary:
- **`no_feature`**: **AUC = 0.999**
- **`dispersive`**: **AUC = 0.997**
- **`cohesive`**: **AUC = 0.996**
- **`non_cohesive`**: **AUC = 0.995**
