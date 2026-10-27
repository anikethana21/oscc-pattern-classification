# Model 4 Export (Champion Dual-Stage Cascade)

This module provides a self-contained, high-performance inference engine for **Model 4 (Dual-Stage Cascade)**.

## Quickstart (2 lines of code)
```python
from model4_export import Model4Predictor

# Automatically loads Stage 1 and Stage 2 weights on GPU
predictor = Model4Predictor()

# Predict any tile path, PIL image, or numpy array
result = predictor.predict("path/to/tile.png")

print("Prediction:", result["pred_class"])
print("Confidence:", result["confidence"])
print("Decided by:", result["decided_by"])
print("Probabilities:", result["probabilities"])
```

## Performance on Locked Test Set (N=267)
- **Overall Accuracy**: 87.64%
- **Macro-F1**: 0.8321
- **Cohesive Precision**: 88.55%
- **Noncohesive Precision**: 94.12%
- **No-Feature Precision**: 88.89%
