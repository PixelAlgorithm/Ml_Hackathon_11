"""Advanced models + Bayesian tuning (Person 3's scope; script added by Person 4 so it is reproducible).

    python -m src.models.train_advanced_models --out artifacts/models --results results --trials 15

Data  : results/train_candidates.csv (fit + tune) and results/student_test_candidates.csv
        (reported ONCE at the end, never used for choices). Evaluator data is never read.
Models: lightgbm_baseline, lightgbm_tuned (Optuna TPE = Bayesian), xgboost_baseline, mlp_baseline.
Tuning: objective = mean log loss over 3-fold GroupKFold (folds split by TRIP, so no trip is in
        both fit and validation). Log loss is used because the HMM consumes probabilities,
        not hard labels. Seed fixed -> reproducible.
Labels are Person 1's nearest-route-arc pseudo-labels (see src/features/build_candidates.py).
Writes artifacts/models/<name>/{model.joblib,manifest.json}; results/advanced_models_results.json;
results/lightgbm_tuning_trials.csv.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = ["distance_m", "log_distance_m", "arc_length_m", "heading_alignment", "speed_mps",
            "log_dt_prev_s", "progress_on_arc", "endpoint_distance_m", "out_degree_from", "in_degree_to"]
SEED = 0


def load(path):
    d = pd.read_csv(path, dtype={"trip_id": str})
    return d[FEATURES].astype("float32"), d["label"].astype(int).to_numpy(), d


def top1_proxy(df, p):
    """Share of GPS points whose highest-probability candidate carries label 1."""
    t = df[["trip_id", "obs_idx", "label"]].copy()
    t["p"] = p
    return float(t.loc[t.groupby(["trip_id", "obs_idx"])["p"].idxmax()]["label"].mean())


def scores(df, y, p):
    return dict(macro_f1=float(f1_score(y, (p >= .5).astype(int), average="macro")),
                top1_proxy=top1_proxy(df, p),
                log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])))


def lgbm(**kw):
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbose=-1)
    return LGBMClassifier(**{**base, **kw})


def tune_lightgbm(X, y, groups, trials, csv_path):
    """Optuna TPE over a small LightGBM space; objective = mean 3-fold group CV log loss."""
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))

    def objective(t):
        params = dict(num_leaves=t.suggest_int("num_leaves", 15, 127, log=True),
                      learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                      n_estimators=t.suggest_int("n_estimators", 100, 500, step=50),
                      min_child_samples=t.suggest_int("min_child_samples", 20, 400, log=True),
                      subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                      colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                      reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
        ll = []
        for tr, va in folds:
            m = lgbm(**params).fit(X.iloc[tr], y[tr])
            ll.append(log_loss(y[va], np.clip(m.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)))
        print(f"trial {t.number}: cv log loss {np.mean(ll):.4f}", flush=True)
        return float(np.mean(ll))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(csv_path, index=False)
    return study.best_params, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/models"))
    ap.add_argument("--trials", type=int, default=15)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    Xtr, ytr, dtr = load(a.results / "train_candidates.csv")
    Xte, yte, dte = load(a.results / "student_test_candidates.csv")
    groups = dtr["trip_id"].to_numpy()
    print(f"train rows {len(Xtr)} ({dtr.trip_id.nunique()} trips), student_test rows {len(Xte)}", flush=True)

    best, best_cv = tune_lightgbm(Xtr, ytr, groups, a.trials, a.results / "lightgbm_tuning_trials.csv")
    print("best params", best, "cv log loss", best_cv, flush=True)

    models = {
        "lightgbm_baseline": lgbm(),
        "lightgbm_tuned": lgbm(**best, subsample_freq=1),
        "xgboost_baseline": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                          tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "mlp_baseline": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=40, random_state=SEED)),
    }
    folds = list(GroupKFold(n_splits=3).split(Xtr, ytr, groups))
    report = {"seed": SEED, "tuning": {"method": "Optuna TPE", "trials": a.trials, "objective": "3-fold group CV log loss",
                                       "best_params": best, "best_cv_log_loss": best_cv}, "models": {}}
    for name, model in models.items():
        t0 = time.time()
        cv = []
        for tr, va in folds:        # CV on train trips only
            m = __import__("sklearn.base", fromlist=["clone"]).clone(model).fit(Xtr.iloc[tr], ytr[tr])
            cv.append(scores(dtr.iloc[va], ytr[va], m.predict_proba(Xtr.iloc[va])[:, 1]))
        model.fit(Xtr, ytr)         # final fit on all train trips
        test = scores(dte, yte, model.predict_proba(Xte)[:, 1])   # held-out dev check, reported once
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, d / "model.joblib", compress=3)
        (d / "manifest.json").write_text(json.dumps({"positive_class": 1, "temperature": 1.0, "features": FEATURES}))
        report["models"][name] = dict(cv_mean={k: float(np.mean([c[k] for c in cv])) for k in cv[0]},
                                      student_test=test, seconds=round(time.time() - t0, 1))
        print(name, json.dumps(report["models"][name]), flush=True)
    (a.results / "advanced_models_results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
