import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score

CLASSES = ["cohesive", "dispersive", "no_feature", "noncohesive"]

# Load test predictions
df1 = pd.read_csv(r"D:\Project\predictions_cv.csv")
df2 = pd.read_csv(r"D:\Project\second_model\predictions_cv_v2.csv")
df3 = pd.read_csv(r"D:\Project\hybridpath_model\weights\test_predictions.csv")
df4 = pd.read_csv(r"D:\Project\model4_cascade\m4_test_predictions.csv")
df5 = pd.read_csv(r"D:\Project\model5_loop\m5_test_predictions.csv")

if "dup_flag" in df1.columns: df1 = df1[df1["dup_flag"] == False]
if "dup_flag" in df2.columns: df2 = df2[df2["dup_flag"] == False]

test_paths = set(df3["path"])

df1_test = df1[df1["path"].isin(test_paths)].sort_values("path").reset_index(drop=True)
df2_test = df2[df2["path"].isin(test_paths)].sort_values("path").reset_index(drop=True)
df3_test = df3.sort_values("path").reset_index(drop=True)
df4_test = df4.sort_values("path").reset_index(drop=True)
df5_test = df5.sort_values("path").reset_index(drop=True)

y_true = df3_test["label"].values

# Predictions
y_preds = [
    df1_test["predicted"].values,
    df2_test["predicted"].values,
    df3_test["predicted_class"].values,
    df4_test["pred_label"].values,
    df5_test["predicted"].values,
]

model_names = [
    "Model 1: Baseline (Phikon)",
    "Model 2: V2 (Phikon + Named)",
    "Model 3: HybridPath (Deep Gated)",
    "Model 4: Specialist (Scaled ~2.5k)",
    "Model 5: Looping (Recurrent Dual-Stage)",
]

cmaps = ["Blues", "Purples", "Greens", "Oranges", "YlGnBu"]

# Compute metrics
metrics = []
cms = []
for name, y_p in zip(model_names, y_preds):
    cm = confusion_matrix(y_true, y_p, labels=CLASSES)
    cms.append(cm)
    acc = accuracy_score(y_true, y_p) * 100
    bacc = balanced_accuracy_score(y_true, y_p) * 100
    macro_f1 = f1_score(y_true, y_p, average="macro")
    per_class_f1 = f1_score(y_true, y_p, average=None, labels=CLASSES)
    disp_f1 = per_class_f1[1]
    metrics.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced_Acc": bacc,
        "Macro_F1": macro_f1,
        "Disp_F1": disp_f1,
    })

# 2x3 Figure Layout
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
axes_flat = axes.flatten()

for idx in range(5):
    ax = axes_flat[idx]
    cm = cms[idx]
    m = metrics[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[idx], cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{m['Model']}\nAccuracy: {m['Accuracy']:.1f}% | Macro-F1: {m['Macro_F1']:.4f}\nDispersive F1: {m['Disp_F1']:.4f}",
                 fontsize=11.5, fontweight="bold", pad=8)
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")

# 6th Cell: Comparative Summary Bar Chart
ax_bar = axes_flat[5]
df_metrics = pd.DataFrame(metrics)
short_names = ["M1: Phikon", "M2: Named", "M3: HybridPath", "M4: Specialist", "M5: Looping"]

x = np.arange(len(short_names))
width = 0.28

bars1 = ax_bar.bar(x - width, df_metrics["Macro_F1"] * 100, width, label="Macro-F1 (%)", color="#2b5c8f")
bars2 = ax_bar.bar(x, df_metrics["Disp_F1"] * 100, width, label="Dispersive F1 (%)", color="#e27c38")
bars3 = ax_bar.bar(x + width, df_metrics["Accuracy"], width, label="Accuracy (%)", color="#2a9d8f")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5, fontweight="bold")
ax_bar.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
ax_bar.set_title("Performance Evolution Across Models\n(Locked Test Set, N=267)", fontsize=11.5, fontweight="bold", pad=8)
ax_bar.set_ylim(50, 100)
ax_bar.legend(loc="lower right", fontsize=9)
ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

# Value labels on bars
for bar in bars1:
    yval = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

plt.suptitle("Complete Head-to-Head Confusion Matrix & Metric Comparison\nEvaluated on Locked Test Set (N=267 Tiles)", 
             fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout()

out1 = Path(r"D:\Project\all_5_models_confusion_matrix.png")
out2 = Path(r"C:\Users\anike\.gemini\antigravity-ide\brain\40d4b308-7194-4358-b8e7-0f6a076836ae\all_5_models_confusion_matrix.png")

plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()

print(f"Grand comparison saved to:\n  - {out1}\n  - {out2}")
