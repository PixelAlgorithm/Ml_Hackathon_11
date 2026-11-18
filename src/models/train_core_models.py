import argparse
import json
import joblib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, auc, f1_score, log_loss,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "distance_m", "log_distance_m", "arc_length_m", "heading_alignment",
    "speed_mps", "log_dt_prev_s", "progress_on_arc",
    "endpoint_distance_m", "out_degree_from", "in_degree_to",
]

SEED = 20261118
COLORS = {"KNN": "#DD8452", "RandomForest": "#55A868"}


def make_models(k=15):
    return {
        # KNN is distance based, so scaling is REQUIRED.
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
    }


def top1_proxy(df, p):
    tmp = df[["trip_id", "obs_idx", "label"]].copy()
    tmp["p"] = p
    best = tmp.loc[tmp.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    return float(best["label"].mean())


def evaluate(df, p):
    y = df["label"].astype(int).to_numpy()
    pred = (p >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top1_proxy": top1_proxy(df, p),
        "log_loss": float(log_loss(y, np.clip(p, 1e-10, 1 - 1e-10), labels=[0, 1])),
    }


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_test_comparison(test_scores, out):
    names = list(test_scores)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(
            axes, [("macro_f1", "Candidate macro F1 (higher = better)"),
                   ("top1_proxy", "Top-1 accuracy proxy (higher = better)"),
                   ("log_loss", "Log loss (lower = better)")]):
        vals = [test_scores[n][key] for n in names]
        bars = ax.bar(names, vals, color=[COLORS[n] for n in names])
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Model comparison on the held-out student test set")
    save(fig, out / "core_model_comparison.png")


def plot_cv(cv_scores, out):
    names = list(cv_scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, title) in zip(axes, [("macro_f1", "Macro F1"),
                                       ("top1_proxy", "Top-1 accuracy proxy")]):
        means = [np.mean([f[key] for f in cv_scores[n]]) for n in names]
        stds = [np.std([f[key] for f in cv_scores[n]]) for n in names]
        ax.bar(names, means, yerr=stds, capsize=6, color=[COLORS[n] for n in names])
        ax.set_title(f"{title} (mean ± std over trip-grouped folds)", fontsize=10)
        ax.set_ylim(0, 1)
    save(fig, out / "core_model_cv.png")


def plot_curves(test_df, test_probs, out):
    y = test_df["label"].astype(int).to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=COLORS[n], label=f"{n} (AUC={auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=COLORS[n], label=f"{n} (AUC={auc(rc, pr):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    a1.legend(); a2.legend()
    save(fig, out / "core_model_roc_pr.png")


def plot_confusions(test_df, test_probs, out):
    y = test_df["label"].astype(int).to_numpy()
    fig, axes = plt.subplots(1, len(test_probs), figsize=(4.2 * len(test_probs), 4))
    for ax, (n, p) in zip(np.atleast_1d(axes), test_probs.items()):
        ConfusionMatrixDisplay.from_predictions(
            y, (p >= 0.5).astype(int), ax=ax, colorbar=False, cmap="Blues",
            display_labels=["wrong road", "right road"])
        ax.set_title(n)
    save(fig, out / "core_model_confusion.png")


def plot_rf_importance(rf, out):
    order = np.argsort(rf.feature_importances_)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], rf.feature_importances_[order],
            color=COLORS["RandomForest"])
    ax.set_xlabel("Importance"); ax.set_title("Random Forest feature importance")
    save(fig, out / "rf_feature_importance.png")


def plot_knn_k(train, out, ks=(1, 3, 5, 9, 15, 25, 41)):
    x, y, g = train[FEATURE_COLUMNS], train["label"].astype(int), train["trip_id"]
    f1s, top1s = [], []
    for k in ks:
        f, t = [], []
        for tr, va in GroupKFold(n_splits=3).split(x, y, g):
            m = make_models(k)["KNN"].fit(x.iloc[tr], y.iloc[tr])
            p = m.predict_proba(x.iloc[va])[:, 1]
            r = evaluate(train.iloc[va], p)
            f.append(r["macro_f1"]); t.append(r["top1_proxy"])
        f1s.append(np.mean(f)); top1s.append(np.mean(t))
        print(f"KNN k={k}: F1={f1s[-1]:.3f} top1={top1s[-1]:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, f1s, "o-", color=COLORS["KNN"], label="Macro F1")
    ax.plot(ks, top1s, "s--", color="gray", label="Top-1 proxy")
    ax.set(xlabel="k (number of neighbours)", ylabel="CV score",
           title="KNN: effect of k (trip-grouped CV)")
    ax.legend()
    save(fig, out / "knn_k_sweep.png")


def main(train_path, test_path, out, skip_k_sweep=False):
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path, dtype={"trip_id": str})
    test = pd.read_csv(test_path, dtype={"trip_id": str})
    missing = [c for c in FEATURE_COLUMNS + ["trip_id", "obs_idx", "label"] if c not in train.columns]
    if missing:
        raise SystemExit(f"Train CSV is missing columns: {missing}")
    assert set(train.trip_id).isdisjoint(set(test.trip_id)), "train/test trips overlap"
    x, y, g = train[FEATURE_COLUMNS], train["label"].astype(int), train["trip_id"]

    cv_scores, test_scores, test_probs, fitted = {}, {}, {}, {}
    for name, _ in make_models().items():
        cv_scores[name] = []
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=3).split(x, y, g), 1):
            m = make_models()[name]
            t0 = time.perf_counter(); m.fit(x.iloc[tr], y.iloc[tr])
            p = m.predict_proba(x.iloc[va])[:, 1]
            r = evaluate(train.iloc[va], p); r["fold"] = fold
            r["fit_seconds"] = round(time.perf_counter() - t0, 2)
            cv_scores[name].append(r)
            print(name, "fold", fold, r, flush=True)
        m = make_models()[name]
        t0 = time.perf_counter(); m.fit(x, y); fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        test_probs[name] = p
        test_scores[name] = evaluate(test, p)
        test_scores[name].update(train_seconds=round(fit_s, 2),
                                 test_seconds=round(time.perf_counter() - t0, 2))
        fitted[name] = m
        print(name, "TEST", test_scores[name], flush=True)

    rows = []
    for n in test_scores:
        rows.append({"model": n,
                     "cv_macro_f1": np.mean([f["macro_f1"] for f in cv_scores[n]]),
                     "cv_top1_proxy": np.mean([f["top1_proxy"] for f in cv_scores[n]]),
                     "test_macro_f1": test_scores[n]["macro_f1"],
                     "test_top1_proxy": test_scores[n]["top1_proxy"],
                     "test_log_loss": test_scores[n]["log_loss"],
                     "train_seconds": test_scores[n]["train_seconds"],
                     "test_seconds": test_scores[n]["test_seconds"]})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(out / "core_model_comparison_table.csv", index=False)
    with open(out / "core_models_results.json", "w") as f:
        json.dump({"cv": cv_scores, "test": test_scores}, f, indent=2)
    print(table.to_string(index=False))

    # Save the trained Random Forest (selected model) for Person 4's decoder.
    # Features are saved with it so nobody mixes up the column order.
    joblib.dump({"model": fitted["RandomForest"], "features": FEATURE_COLUMNS},
                out / "random_forest.joblib", compress=3)
    print("saved", out / "random_forest.joblib")

    plot_test_comparison(test_scores, out)
    plot_cv(cv_scores, out)
    plot_curves(test, test_probs, out)
    plot_confusions(test, test_probs, out)
    plot_rf_importance(fitted["RandomForest"], out)
    if not skip_k_sweep:
        plot_knn_k(train, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("results/train_candidates.csv"))
    ap.add_argument("--test", type=Path, default=Path("results/student_test_candidates.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-k-sweep", action="store_true")
    a = ap.parse_args()
    main(a.train, a.test, a.out, a.skip_k_sweep)