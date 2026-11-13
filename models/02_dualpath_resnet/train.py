"""
Step 3 (v2): Train linear classifier on Phikon + named features.

Usage:
    cd D:\Project\second_model
    python train.py

Same as baseline train.py but reads embeddings_v2.npy (1,546-dim)
which includes deep features + 10 named morphological/color features.
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_one_fold(X_train, y_train, X_val, y_val,
                   in_dim, num_classes, device,
                   lr=1e-3, epochs=200, patience=15):
    model = LinearHead(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val

    best_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val_np, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}, best macro-F1: {best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return model, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="embeddings_v2.npy")
    ap.add_argument("--labels", default="embedding_labels_v2.csv")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    X = np.load(args.embeddings)
    df = pd.read_csv(args.labels)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    # groups not used — mixed dataset
    num_classes = len(le.classes_)
    in_dim = X.shape[1]

    print(f"\nData: {X.shape[0]} tiles, {in_dim} features "
          f"(1536 deep + {in_dim - 1536} named), {num_classes} classes")
    print(f"Classes: {list(le.classes_)}")
    
    print(f"\nClass distribution:")
    for cls_name, cls_id in zip(le.classes_, range(num_classes)):
        count = (y == cls_id).sum()
        print(f"  {cls_name}: {count} ({count/len(y):.1%})")

    # ---- Named feature importance preview ----
    named_cols = [
        "nuclei_count", "nuclei_density",
        "mean_nucleus_area", "std_nucleus_area",
        "mean_circularity", "mean_eccentricity",
        "mean_nn_distance", "std_nn_distance",
        "red_fraction", "pink_fraction",
    ]
    available = [c for c in named_cols if c in df.columns]
    if available:
        print(f"\nNamed feature means by class:")
        print(df.groupby("label")[available].mean().round(2).to_string())

    # ---- Patient-level stratified cross-validation ----
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_f1s = []
    all_preds = np.full(len(y), -1, dtype=int)
    all_probs = np.zeros((len(y), num_classes), dtype=float)
    best_overall_f1 = 0.0
    best_model = None

    print(f"\n{'='*60}")
    print(f"TRAINING (v2: deep + named features)")
    print(f"{args.folds}-fold patient-level cross-validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        
        

        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        print(f"  Train: {len(train_idx)} tiles")
        print(f"  Val:   {len(val_idx)} tiles")

        model, f1 = train_one_fold(
            X_train, y_train, X_val, y_val,
            in_dim, num_classes, args.device
        )
        fold_f1s.append(f1)
        print(f"  Macro-F1: {f1:.4f}")

        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(args.device)
            logits = model(X_val_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        all_preds[val_idx] = preds
        all_probs[val_idx] = probs

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_model = model

    # ---- Save model ----
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)

    torch.save(best_model.state_dict(), model_dir / "linear_head.pt")
    np.save(model_dir / "label_classes.npy", le.classes_)

    config = {
        "in_dim": in_dim,
        "num_classes": num_classes,
        "classes": le.classes_.tolist(),
        "backbone": "owkin/phikon",
        "pooling": "mean_max",
        "grid_size": 4,
        "crop_size": 224,
        "embedding_dim": 768,
        "named_features": in_dim - 1536,
        "model_version": "v2_deep_plus_named",
        "folds": args.folds,
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Results ----
    print(f"\n{'='*60}")
    print("RESULTS (v2: deep + named features)")
    print(f"{'='*60}")
    print(f"Mean macro-F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"Per-fold F1:   {[f'{f:.4f}' for f in fold_f1s]}")

    print(f"\nClassification Report:")
    print(classification_report(y, all_preds, target_names=le.classes_))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, all_preds)
    print(f"{'':>15}", end="")
    for name in le.classes_:
        print(f"{name:>14}", end="")
    print()
    for i, name in enumerate(le.classes_):
        print(f"{name:>15}", end="")
        for j in range(num_classes):
            print(f"{cm[i][j]:>14}", end="")
        print()

    # ---- Feature importance (weight magnitude) ----
    w = best_model.fc.weight.detach().cpu().numpy()  # [num_classes, in_dim]
    importance = np.abs(w).mean(axis=0)  # mean abs weight per feature
    named_importance = importance[-len(named_cols):]
    print(f"\nNamed feature importance (mean |weight| across classes):")
    for name, imp in sorted(zip(named_cols, named_importance),
                            key=lambda x: -x[1]):
        bar = "#" * int(imp / named_importance.max() * 30)
        print(f"  {name:>25}: {imp:.4f}  {bar}")

    deep_mean_imp = importance[:1536].mean()
    named_mean_imp = named_importance.mean()
    print(f"\n  Deep features avg importance:  {deep_mean_imp:.4f}")
    print(f"  Named features avg importance: {named_mean_imp:.4f}")

    # ---- Save predictions ----
    df["predicted"] = le.inverse_transform(all_preds)
    df["confidence"] = all_probs.max(axis=1)
    df["needs_human"] = df["confidence"] < 0.7
    for i, cls_name in enumerate(le.classes_):
        df[f"prob_{cls_name}"] = all_probs[:, i]
    df.to_csv("predictions_cv_v2.csv", index=False)

    needs_review = df["needs_human"].sum()
    print(f"\nTiles needing human review (confidence < 70%): "
          f"{needs_review} ({needs_review/len(df):.1%})")
    print(f"\nModel saved to {model_dir}/")
    print(f"Predictions saved to predictions_cv_v2.csv")


if __name__ == "__main__":
    main()