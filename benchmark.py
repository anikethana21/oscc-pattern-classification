"""
benchmark.py - honest model benchmark for the oral-cancer tile classifier.

What it does
  * Locks a TEST set of ~10% of patients (saved to benchmark_split.csv, never used here unless --final).
  * Runs repeated, patient-grouped, class- and annotator-stratified K-fold CV on the rest, the SAME folds
    for every model and feature set, so differences are paired.
  * Models: majority baseline, logistic regression (3 strengths), linear SVM, RBF SVM, MLP,
    Random Forest (bagging), HistGradientBoosting (boosting).
  * Reports macro-F1 (mean +- SD over folds, pooled out-of-fold), balanced accuracy, per-class F1,
    per-annotator macro-F1, and patient-level bootstrap 95% CIs, including the paired delta vs a reference.

Usage (rows of every .npy must be in the same order as the rows of tiles.csv):
  python benchmark.py --tiles tiles.csv --features deep=X_deep.npy named=X_named.npy --combos deep+named
  python benchmark.py ... --models majority logreg_c0.1 mlp --seeds 3 --folds 5 --jobs 4
  python benchmark.py ... --final logreg_c0.1@deep+named      # touches the locked test set ONCE
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")
ALL_MODELS = ["majority", "logreg_c0.01", "logreg_c0.1", "logreg_c1", "linear_svm", "svm_rbf",
              "mlp", "random_forest", "hist_gb"]


# ------------------------------------------------------------------------------ helpers
def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(1) > 0
    return float(f1[present].mean())


def make_model(name, n_feat, seed):
    pre = [StandardScaler()]
    pca = [PCA(min(64, n_feat), random_state=seed)] if n_feat > 64 else []
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name.startswith("logreg_c"):
        c = float(name.split("_c")[1])
        return make_pipeline(*pre, LogisticRegression(C=c, class_weight="balanced", max_iter=3000))
    if name == "linear_svm":
        return make_pipeline(*pre, LinearSVC(C=0.01, class_weight="balanced", dual="auto", max_iter=5000))
    if name == "svm_rbf":
        return make_pipeline(*pre, SVC(C=1.0, gamma="scale", class_weight="balanced"))
    if name == "mlp":   # sklearn MLP has no class weights: expect weaker minority-class recall
        return make_pipeline(*pre, MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True,
                                                 n_iter_no_change=15, max_iter=300, random_state=seed))
    if name == "random_forest":
        return make_pipeline(*pre, *pca, RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed))
    if name == "hist_gb":
        return make_pipeline(*pre, *pca, HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, early_stopping=True, class_weight="balanced", random_state=seed))
    raise ValueError(name)


def fit_eval(model, fset, X, y, tr, va, seed, fold):
    t0 = time.time()
    m = make_model(model, X.shape[1], seed)
    m.fit(X[tr], y[tr])
    return dict(model=model, fset=fset, seed=seed, fold=fold, va=va, pred=m.predict(X[va]),
                secs=time.time() - t0)


def choose_test_patients(df, strat, k, seed):
    overall = np.bincount(df["y"]) / len(df)
    n_cls = len(overall)
    best = None
    for s in range(seed, seed + 40):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        for _, te in cv.split(np.zeros(len(df)), strat, df["patient"]):
            if len(set(df["y"].iloc[te])) < n_cls:
                continue
            p = np.bincount(df["y"].iloc[te], minlength=n_cls) / len(te)
            score = 0.5 * np.abs(p - overall).sum()
            if best is None or score < best[0]:
                best = (score, te)
    if best is None:
        raise SystemExit("No patient-level test split contains every class. You need more patients "
                         "per class (check the 'patients per class' table).")
    return set(df["patient"].iloc[best[1]])


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--features", nargs="+", required=True, help="name=path.npy ...")
    ap.add_argument("--combos", nargs="*", default=[], help="e.g. deep+named")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--reference", default=None, help="model@fset for paired deltas")
    ap.add_argument("--split-file", default="benchmark_split.csv")
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--final", default=None, help="model@fset: evaluate ONCE on the locked test set")
    ap.add_argument("--scores-only", action="store_true", help="Print scores only and skip plot generation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(args.tiles)
    if "patient" not in df:
        slide = df["path"].apply(lambda p: Path(p).stem).str.replace(r"_x\d+_y\d+$", "", regex=True)
        df["patient"] = slide.str.extract(r"^(\d+)")[0].fillna(slide)
    df["patient"] = df["patient"].astype(str)
    classes = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    k = len(classes)
    has_ann = "annotator" in df
    strat = pd.factorize(df["label"] + ("|" + df["annotator"].astype(str) if has_ann else ""))[0]

    feats = {}
    for item in args.features:
        name, path = item.split("=", 1)
        feats[name] = np.load(path).astype(np.float32)
        assert len(feats[name]) == len(df), f"{name}: {len(feats[name])} rows vs {len(df)} tiles"
    for combo in args.combos:
        feats[combo] = np.hstack([feats[p] for p in combo.split("+")])

    print(f"tiles {len(df)} | patients {df.patient.nunique()} | classes {classes}")
    print(df.groupby("label")["patient"].nunique().rename("patients per class").to_string())
    if has_ann:
        print(pd.crosstab(df["annotator"], df["label"]).to_string())

    # locked test set (chosen once, then reused from file)
    sp = Path(args.split_file)
    if sp.exists():
        test_p = set(pd.read_csv(sp, dtype=str)["patient"])
    else:
        if df.patient.nunique() < 15:
            print(f"Note: Dataset has {df.patient.nunique()} patients (< 15). Evaluating via full cross-validation without locked holdout test set.")
            test_p = set()
        else:
            test_p = choose_test_patients(df, strat, 10, 42)
            pd.DataFrame({"patient": sorted(test_p)}).to_csv(sp, index=False)
            print(f"Locked test set: {len(test_p)} patients -> {sp} (keep this file)")
    df["is_test"] = df["patient"].isin(test_p)
    tv_idx = np.where(~df["is_test"])[0]
    y = df["y"].to_numpy()
    print(f"train/val tiles {len(tv_idx)} | locked test tiles {int(df.is_test.sum())} (untouched)")

    # ---------------------------------------------------------------- final mode
    if args.final:
        mname, fname = args.final.split("@")
        te_idx = np.where(df["is_test"])[0]
        if len(te_idx) == 0:
            raise SystemExit("No test set locked to run --final.")
        m = make_model(mname, feats[fname].shape[1], 0)
        m.fit(feats[fname][tv_idx], y[tv_idx])
        pred = m.predict(feats[fname][te_idx])
        f1 = macro_f1(y[te_idx], pred, k)
        rng = np.random.default_rng(0)
        pats = df["patient"].iloc[te_idx].to_numpy()
        by_p = {p: np.where(pats == p)[0] for p in set(pats)}
        keys = list(by_p)
        boots = []
        for _ in range(1000):
            idx = np.concatenate([by_p[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            boots.append(macro_f1(y[te_idx][idx], pred[idx], k))
        print(f"\n=== FINAL TEST: {args.final} ===")
        print(f"macro-F1 {f1:.4f}  (95% patient-bootstrap CI {np.percentile(boots, 2.5):.4f}-"
              f"{np.percentile(boots, 97.5):.4f})  balanced acc {balanced_accuracy_score(y[te_idx], pred):.4f}")
        print(pd.DataFrame(confusion_matrix(y[te_idx], pred, labels=range(k)), index=classes, columns=classes))
        print(classification_report(y[te_idx], pred, target_names=classes, zero_division=0))
        print("Test set used. Do not tune on it any more.")
        return

    # ---------------------------------------------------------------- CV benchmark
    n_groups = len(set(df["patient"].to_numpy()[tv_idx]))
    actual_folds = min(args.folds, n_groups)
    if actual_folds < args.folds:
        print(f"Note: Clamping folds from {args.folds} to {actual_folds} because only {n_groups} patient groups exist.")

    folds = {}   # (seed) -> list of (tr, va) index arrays
    for seed in range(args.seeds):
        cv = StratifiedGroupKFold(actual_folds, shuffle=True, random_state=seed)
        folds[seed] = [(tv_idx[a], tv_idx[b]) for a, b in
                       cv.split(np.zeros(len(tv_idx)), strat[tv_idx], df["patient"].to_numpy()[tv_idx])]
    tasks = [(m, f, seed, i, tr, va) for f in feats for m in args.models
             for seed, fl in folds.items() for i, (tr, va) in enumerate(fl)]
    print(f"\nrunning {len(tasks)} fits on {args.jobs} job(s) ...")
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_eval)(m, f, feats[f], y, tr, va, seed, i) for m, f, seed, i, tr, va in tasks)

    rows, oof = [], {}
    for r in res:
        key = (r["model"], r["fset"])
        f1 = macro_f1(y[r["va"]], r["pred"], k)
        rows.append(dict(model=r["model"], fset=r["fset"], seed=r["seed"], fold=r["fold"], f1=f1,
                         bacc=balanced_accuracy_score(y[r["va"]], r["pred"]), secs=r["secs"]))
        arr = oof.setdefault((key, r["seed"]), np.full(len(df), -1))
        arr[r["va"]] = r["pred"]
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out / "benchmark_folds.csv", index=False)

    # patient bootstrap (same resamples for every model -> paired)
    pats = df["patient"].to_numpy()[tv_idx]
    by_p = {p: tv_idx[np.where(pats == p)[0]] for p in set(pats)}
    keys_p = list(by_p)
    rng = np.random.default_rng(0)
    resamples = [np.concatenate([by_p[keys_p[i]] for i in rng.integers(0, len(keys_p), len(keys_p))])
                 for _ in range(args.boot)]

    ref = args.reference or f"{args.models[0] if 'logreg_c0.1' not in args.models else 'logreg_c0.1'}@{list(feats)[0]}"
    ref_m, ref_f = ref.split("@")
    ref_oof = oof[((ref_m, ref_f), 0)]
    ref_boot = np.array([macro_f1(y[ix], ref_oof[ix], k) for ix in resamples])

    summary, perclass, perann = [], [], []
    for key in sorted({kk for kk, _ in oof}):
        sub = fold_df[(fold_df.model == key[0]) & (fold_df.fset == key[1])]
        pooled = np.mean([macro_f1(y[tv_idx], oof[(key, s)][tv_idx], k) for s in range(args.seeds)])
        p0 = oof[(key, 0)]
        boot = np.array([macro_f1(y[ix], p0[ix], k) for ix in resamples])
        d = boot - ref_boot
        summary.append(dict(
            fset=key[1], model=key[0], fold_f1=f"{sub.f1.mean():.4f} +- {sub.f1.std(ddof=1):.4f}",
            pooled_f1=round(pooled, 4), ci95=f"{np.percentile(boot, 2.5):.3f}-{np.percentile(boot, 97.5):.3f}",
            bal_acc=round(sub.bacc.mean(), 4),
            delta_vs_ref=f"{np.mean(d):+.4f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]",
            secs_per_fit=round(sub.secs.mean(), 1)))
        for ci, c in enumerate(classes):
            t = y[tv_idx] == ci
            tp = np.sum(p0[tv_idx][t] == ci)
            prec = tp / max(np.sum(p0[tv_idx] == ci), 1)
            rec = tp / max(t.sum(), 1)
            perclass.append(dict(fset=key[1], model=key[0], cls=c, precision=round(prec, 3),
                                 recall=round(rec, 3), f1=round(2 * prec * rec / max(prec + rec, 1e-12), 3),
                                 support=int(t.sum())))
        if has_ann:
            for a in sorted(df["annotator"].unique()):
                ix = tv_idx[df["annotator"].to_numpy()[tv_idx] == a]
                perann.append(dict(fset=key[1], model=key[0], annotator=a,
                                   macro_f1=round(macro_f1(y[ix], p0[ix], k), 4), tiles=len(ix)))

    sm = pd.DataFrame(summary).sort_values("pooled_f1", ascending=False)
    sm.to_csv(out / "benchmark_summary.csv", index=False)
    pd.DataFrame(perclass).to_csv(out / "benchmark_perclass.csv", index=False)
    if perann:
        pd.DataFrame(perann).to_csv(out / "benchmark_annotator.csv", index=False)
    with open(out / "benchmark.md", "w") as fh:
        try:
            fh.write(sm.to_markdown(index=False))
        except (ImportError, AttributeError):
            fh.write(sm.to_string(index=False))

    print(f"\n=== BENCHMARK (patient-grouped CV, {args.seeds} seeds x {actual_folds} folds; reference {ref}) ===")
    print(sm.to_string(index=False))
    if perann:
        print("\nmacro-F1 per annotator (seed-0 out-of-fold):")
        print(pd.DataFrame(perann).pivot_table(index=["fset", "model"], columns="annotator",
                                               values="macro_f1").to_string())
    print("\nIf a delta interval includes 0, the difference is not distinguishable from noise.")

    if not args.scores_only:
        top = sm.head(12).iloc[::-1]
        lo = [float(s.split("-")[0]) for s in top["ci95"]]
        hi = [float(s.split("-")[1]) for s in top["ci95"]]
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(top) + 1.5))
        lab = [f"{a} | {b}" for a, b in zip(top["fset"], top["model"])]
        err_lo = np.maximum(0, np.array(top["pooled_f1"]) - lo); err_hi = np.maximum(0, np.array(hi) - np.array(top["pooled_f1"])); ax.barh(lab, top["pooled_f1"], xerr=[err_lo, err_hi],
                color="#4c78a8", capsize=3)
        ax.set_xlabel("macro-F1 (pooled out-of-fold, 95% patient-bootstrap CI)")
        fig.tight_layout()
        fig.savefig(out / "benchmark.png", dpi=150)
        print(f"\nSaved tables and plot to {out}/")
    else:
        print(f"\nSaved tables to {out}/ (--scores-only: skipped plot generation)\n")
if __name__ == "__main__":
    main()
