"""Decoder hyper-parameter search (Optuna TPE = Bayesian) on DEVELOPMENT data only.

    python -m src.hmm.tune_decoder --data-dir "<Student folder>" --scorer distance \
        --n-trials 60 --seed 0 --tag distance

* Objective (maximised): mean length-weighted route F1 of the classifier+HMM route vs the
  reference route over the TRAIN trips.
* student_test is evaluated for every trial too but is only a held-out CHECK, never the
  selection criterion. Evaluator data is refused (see DEV_PARTITIONS).
* Every trial is appended to results/decoder_tuning_<tag>.csv (params, seed, metrics, time),
  so the full search record is kept. Trial 0 is the default config for a fair comparison.
* The best config is written to results/decoder_best_<tag>.json and, with --write-artifact,
  to artifacts/hmm_config.json which the API loads at start-up.

sigma_m only affects the distance scorer; with the classifier scorer it is not searched
(the classifier probability replaces the Gaussian). emission_weight and sigma_m are partly
redundant for the Gaussian emission (both rescale emission vs transition) -- expected.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .decoder import HMMConfig
from .graph import list_trip_ids
from .runner import DEV_PARTITIONS, prepare, run
from .scoring import DistanceScorer, build_scorer

# ---- worker-process state (each worker keeps its prepared trips across trials) ----
_W: dict = {}


def _init(data_dir, scorer_kind, artifacts):
    _W.update(data_dir=data_dir, preps={}, scorer_kind=scorer_kind, artifacts=artifacts, scorer=None)


def _eval(args):
    tid, cfg_dict = args
    if _W["scorer_kind"] == "distance":
        scorer = DistanceScorer(cfg_dict["sigma_m"])
    else:
        if _W["scorer"] is None:
            _W["scorer"] = build_scorer("classifier", _W["artifacts"])
        scorer = _W["scorer"]
    if tid not in _W["preps"]:
        _W["preps"][tid] = prepare(_W["data_dir"], tid, scorer=scorer)
    out = run(_W["preps"][tid], HMMConfig.from_dict(cfg_dict), scorer)
    m = out["hmm"]
    return dict(f1=m["route_f1"], precision=m["route_precision"], recall=m["route_recall"],
                continuity=m["continuity"], jumps=m["jumps"], breaks=out["breaks"],
                cls_f1=out["classifier_only"]["route_f1"])


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def evaluate_config(pool, trips, cfg_dict, prefix):
    rows = pool.map(_eval, [(t, cfg_dict) for t in trips])
    return {f"{prefix}_f1": _mean(rows, "f1"), f"{prefix}_precision": _mean(rows, "precision"),
            f"{prefix}_recall": _mean(rows, "recall"), f"{prefix}_continuity": _mean(rows, "continuity"),
            f"{prefix}_jumps": float(np.sum([r["jumps"] for r in rows])),
            f"{prefix}_breaks": float(np.sum([r["breaks"] for r in rows])),
            f"{prefix}_classifier_only_f1": _mean(rows, "cls_f1")}


def suggest(trial, scorer_kind) -> dict:
    d = HMMConfig().to_dict()
    d["beta_m"] = trial.suggest_float("beta_m", 1.0, 50.0, log=True)
    d["route_cutoff_m"] = trial.suggest_float("route_cutoff_m", 200.0, 1500.0)
    d["max_gap_s"] = float(trial.suggest_int("max_gap_s", 5, 60))
    d["uturn_penalty"] = trial.suggest_float("uturn_penalty", 0.0, 10.0)
    d["emission_weight"] = trial.suggest_float("emission_weight", 0.2, 3.0, log=True)
    if scorer_kind == "distance":
        d["sigma_m"] = trial.suggest_float("sigma_m", 5.0, 40.0, log=True)
    return d


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scorer", choices=["distance", "classifier"], default="distance")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--train-limit", type=int, default=None, help="random subset of train trips (seeded)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--write-artifact", action="store_true", help="write artifacts/hmm_config.json")
    a = ap.parse_args()
    tag = a.tag or a.scorer

    train = list_trip_ids(a.data_dir, "train")
    val = list_trip_ids(a.data_dir, "student_test")
    assert train and val, f"expected {DEV_PARTITIONS} folders under {a.data_dir}"
    if a.train_limit:
        rng = np.random.default_rng(a.seed)
        train = sorted(rng.choice(train, size=min(a.train_limit, len(train)), replace=False).tolist())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"decoder_tuning_{tag}.csv"
    if csv_path.exists():
        csv_path.unlink()   # one search = one file (re-running with the same tag restarts it)

    pool = mp.Pool(a.jobs, initializer=_init, initargs=(a.data_dir, a.scorer, a.artifacts))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=a.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    default = HMMConfig().to_dict()
    keys = ["beta_m", "route_cutoff_m", "max_gap_s", "uturn_penalty", "emission_weight"]
    if a.scorer == "distance":
        keys.append("sigma_m")
    study.enqueue_trial({k: default[k] for k in keys})   # trial 0 = default config
    records = []

    def objective(trial):
        cfg = suggest(trial, a.scorer)
        t0 = time.perf_counter()
        rec = dict(tag=tag, trial=trial.number, seed=a.seed, sampler="TPE", scorer=a.scorer,
                   n_train_trips=len(train), n_val_trips=len(val), **cfg)
        rec.update(evaluate_config(pool, train, cfg, "train"))
        rec.update(evaluate_config(pool, val, cfg, "student_test"))   # held-out check only
        rec["seconds"] = time.perf_counter() - t0
        records.append(rec)
        pd.DataFrame([rec]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        print(f"trial {trial.number:3d} train F1 {rec['train_f1']:.4f} | student_test F1 "
              f"{rec['student_test_f1']:.4f} | {rec['seconds']:.1f}s", flush=True)
        return rec["train_f1"]

    study.optimize(objective, n_trials=a.n_trials)
    pool.close()

    df = pd.DataFrame(records)
    best = df.loc[df["train_f1"].idxmax()]
    base = df.loc[df["trial"] == 0].iloc[0]
    cfg_keys = list(HMMConfig().to_dict())
    best_cfg = {k: float(best[k]) for k in cfg_keys if k in best}
    summary = dict(tag=tag, scorer=a.scorer, seed=a.seed, n_trials=len(df), n_train_trips=len(train),
                   selection_metric="mean route F1 on train trips",
                   best_trial=int(best["trial"]), config=best_cfg, tuned_for_scorer=a.scorer,
                   default_config={k: float(base[k]) for k in cfg_keys if k in base},
                   train_f1_default=float(base["train_f1"]), train_f1_best=float(best["train_f1"]),
                   student_test_f1_default=float(base["student_test_f1"]),
                   student_test_f1_best=float(best["student_test_f1"]),
                   student_test_classifier_only_f1=float(best["student_test_classifier_only_f1"]))
    (out_dir / f"decoder_best_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.write_artifact:
        art = Path(a.artifacts)
        art.mkdir(parents=True, exist_ok=True)
        (art / "hmm_config.json").write_text(json.dumps(dict(best_cfg, tuned_for_scorer=a.scorer), indent=2))
        print("wrote", art / "hmm_config.json")


if __name__ == "__main__":
    main()
