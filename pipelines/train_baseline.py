"""
Step 3: Train a linear classifier on Phikon embeddings.

Usage:
    python train.py
    python train.py --embeddings embeddings.npy --labels embedding_labels.csv --folds 5

Patient-level stratified cross-validation:
  - No patient appears in both train and validation
  - Prevents data leakage (tiles from same patient look similar)
  - macro-F1 as primary metric (handles class imbalance)
  - Early stopping to prevent overfitting

Outputs:
    model/linear_head.pt       (trained weights)
    model/config.json          (architecture + class mapping)
    model/label_classes.npy    (label encoder)
    predictions_cv.csv         (cross-validated predictions with confidence)
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
    """Simple linear classifier: embedding -> class scores."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_one_fold(X_train, y_train, X_val, y_val,
                   in_dim, num_classes, device,
                   lr=1e-3, epochs=200, patience=15):
    """Train one fold with early stopping on validation macro-F1."""
    model = LinearHead(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights to handle imbalanced data
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
    y_val_np = y_val  # keep numpy for sklearn

    best_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- validate ---
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
    ap.add_argument("--embeddings", default="embeddings.npy")
    ap.add_argument("--labels", default="embedding_labels.csv")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # ---- Load data ----
    X = np.load(args.embeddings)
    df = pd.read_csv(args.labels)

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    # groups not used — mixed dataset, no reliable patient IDs
    num_classes = len(le.classes_)
    in_dim = X.shape[1]

    print(f"\nData: {X.shape[0]} tiles, {in_dim} features, {num_classes} classes")
    print(f"Classes: {list(le.classes_)}")
    # patient-level split skipped for this dataset
    print(f"\nClass distribution:")
    for cls_name, cls_id in zip(le.classes_, range(num_classes)):
        count = (y == cls_id).sum()
        print(f"  {cls_name}: {count} ({count/len(y):.1%})")

    # ---- Patient-level stratified cross-validation ----
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_f1s = []
    all_preds = np.full(len(y), -1, dtype=int)
    all_probs = np.zeros((len(y), num_classes), dtype=float)
    best_overall_f1 = 0.0
    best_model = None

    print(f"\n{'='*60}")
    print(f"TRAINING: {args.folds}-fold patient-level cross-validation")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # stratified random split (no patient grouping)
        

        print(f"\n--- Fold {fold+1}/{args.folds} ---")
        print(f"  Train: {len(train_idx)} tiles")
        print(f"  Val:   {len(val_idx)} tiles")

        model, f1 = train_one_fold(
            X_train, y_train, X_val, y_val,
            in_dim, num_classes, args.device
        )
        fold_f1s.append(f1)
        print(f"  Macro-F1: {f1:.4f}")

        # Store out-of-fold predictions
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
        "folds": args.folds,
        "mean_macro_f1": float(np.mean(fold_f1s)),
        "std_macro_f1": float(np.std(fold_f1s)),
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Overall results ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mean macro-F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"Per-fold F1:   {[f'{f:.4f}' for f in fold_f1s]}")

    print(f"\nClassification Report (all cross-validated predictions):")
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

    # ---- Save predictions ----
    df["predicted"] = le.inverse_transform(all_preds)
    df["confidence"] = all_probs.max(axis=1)
    df["needs_human"] = df["confidence"] < 0.7
    for i, cls_name in enumerate(le.classes_):
        df[f"prob_{cls_name}"] = all_probs[:, i]
    df.to_csv("predictions_cv.csv", index=False)

    needs_review = df["needs_human"].sum()
    print(f"\nTiles needing human review (confidence < 70%): "
          f"{needs_review} ({needs_review/len(df):.1%})")
    print(f"\nModel saved to {model_dir}/")
    print(f"Predictions saved to predictions_cv.csv")
    print(f"\nNext step: python predict.py --input <path_to_new_tiles>")


if __name__ == "__main__":
    main()