"""Classifier-only vs classifier+HMM on a development partition (default: student_test).

    python -m src.hmm.compare --data-dir "<Student folder>" --partition student_test \
        --scorer distance --tag distance

"classifier-only" = per GPS point, the candidate with the highest emission score, joined
into a route (consecutive duplicates removed). "classifier+HMM" = the same scores decoded
with Viterbi. With --scorer distance the "classifier" is just nearest-arc: the numbers then
measure the HMM on a distance baseline, NOT on the trained model -- the `scorer` column and
plot titles say so. Re-run with --scorer classifier once artifacts/ is integrated.

Writes (results/):
  comparison_<tag>_per_trip.csv      one row per trip, both methods + deltas
  comparison_<tag>_summary.csv       means over trips
  comparison_<tag>_not_helped.csv    trips where the HMM did not improve F1 / continuity
  comparison_<tag>_f1.png, _jumps.png
Pointwise accuracy needs per-GPS-point labels, which do not exist in the raw data. If the
shared alignment (Person 1) is exported as <dir>/<trip_id>.npy (int arc id per GPS point,
-1 = unknown), pass --point-labels-dir and it is added; otherwise it is reported as N/A.
Never run on the evaluator partition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decoder import HMMConfig
from .graph import list_trip_ids
from .metrics import pointwise_accuracy
from .runner import DEV_PARTITIONS, prepare, run
from .scoring import build_scorer, DistanceScorer


def load_cfg(path):
    p = Path(path)
    if p.exists():
        return HMMConfig.from_dict(json.loads(p.read_text())), str(p)
    return HMMConfig(), "defaults"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--partition", default="student_test", choices=DEV_PARTITIONS)
    ap.add_argument("--scorer", choices=["distance", "classifier"], default="distance")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--hmm-config", default=None, help="json; default artifacts/hmm_config.json")
    ap.add_argument("--point-labels-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    tag = a.tag or a.scorer
    cfg, cfg_src = load_cfg(a.hmm_config or Path(a.artifacts) / "hmm_config.json")
    scorer = DistanceScorer(cfg.sigma_m) if a.scorer == "distance" else build_scorer("classifier", a.artifacts)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for tid in list_trip_ids(a.data_dir, a.partition):
        prep = prepare(a.data_dir, tid, scorer=scorer)
        r = run(prep, cfg, scorer)
        row = dict(trip=tid, partition=a.partition, scorer=scorer.name, n_points=len(prep.trip.track_xy),
                   hmm_breaks=r["breaks"], points_without_candidates=r["empty"], decode_s=r["decode_s"])
        for name in ("classifier_only", "hmm"):
            for k, v in r[name].items():
                row[f"{name}_{k}"] = v
        if a.point_labels_dir:
            lp = Path(a.point_labels_dir) / f"{tid}.npy"
            if lp.exists():
                lab = np.load(lp)
                row["classifier_only_pointwise_acc"] = pointwise_accuracy(r["argmax_per_point"], lab)
                row["hmm_pointwise_acc"] = pointwise_accuracy(r["hmm_per_point"], lab)
        rows.append(row)
        print(f"{tid}: F1 {row['classifier_only_route_f1']:.3f} -> {row['hmm_route_f1']:.3f} | "
              f"jumps {row['classifier_only_jumps']} -> {row['hmm_jumps']}", flush=True)

    df = pd.DataFrame(rows)
    df["delta_f1"] = df["hmm_route_f1"] - df["classifier_only_route_f1"]
    df["delta_jumps"] = df["hmm_jumps"] - df["classifier_only_jumps"]
    df["hmm_improved_f1"] = df["delta_f1"] > 1e-9
    df.to_csv(out / f"comparison_{tag}_per_trip.csv", index=False)

    # trips where the HMM did not help: F1 not better, or more discontinuities than before
    bad = df[(~df["hmm_improved_f1"]) | (df["delta_jumps"] > 0)]
    bad.to_csv(out / f"comparison_{tag}_not_helped.csv", index=False)

    metrics = ["route_precision", "route_recall", "route_f1", "continuity", "jumps"]
    summ = [dict(metric=m, classifier_only=df[f"classifier_only_{m}"].mean(), hmm=df[f"hmm_{m}"].mean())
            for m in metrics]
    if "hmm_pointwise_acc" in df:
        summ.append(dict(metric="pointwise_accuracy", classifier_only=df["classifier_only_pointwise_acc"].mean(),
                         hmm=df["hmm_pointwise_acc"].mean()))
    else:
        summ.append(dict(metric="pointwise_accuracy", classifier_only="N/A", hmm="N/A"))   # no point labels yet
    s = pd.DataFrame(summ)
    s.insert(0, "scorer", scorer.name)
    s["n_trips"] = len(df)
    s["hmm_config"] = cfg_src
    s.to_csv(out / f"comparison_{tag}_summary.csv", index=False)

    # plots: only real per-trip numbers
    x = np.arange(len(df))
    for key, ylab, fname in [("route_f1", "route F1 (length-weighted)", "f1"), ("jumps", "# route discontinuities", "jumps")]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, df[f"classifier_only_{key}"], 0.4, label="classifier-only (argmax)")
        ax.bar(x + 0.2, df[f"hmm_{key}"], 0.4, label="classifier + HMM")
        ax.set_xticks(x, df["trip"], rotation=45)
        ax.set_ylabel(ylab)
        ax.set_title(f"{a.partition}: {ylab} per trip (scorer: {scorer.name})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"comparison_{tag}_{fname}.png", dpi=150)
        plt.close(fig)

    print(s.to_string(index=False))
    print(f"\nHMM did not help (F1 not better or more jumps) on {len(bad)}/{len(df)} trips: {list(bad['trip'])}")


if __name__ == "__main__":
    main()
