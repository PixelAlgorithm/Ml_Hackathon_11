"""Compare every trained model, each used as the HMM emission model, on student_test.

    python -m src.hmm.compare_models --data-dir "<Student folder>" --models artifacts/models \
        --point-labels-dir results/point_labels/student_test

Each sub-folder of --models holds model.joblib (+ manifest.json). All models are decoded with the
SAME default HMM config (no per-model tuning), so differences come from the classifier alone.
For each model we report, averaged over the student_test trips:
  classifier-only route F1 / continuity / jumps   (argmax candidate per GPS point)
  classifier+HMM  route F1 / continuity / jumps
  pointwise accuracy vs Person 1's pseudo-labels, for both (labels are heuristic, see
  src/data/export_point_labels.py)
Writes results/model_comparison_hmm.csv and results/model_comparison_hmm.png.
Only development partitions are allowed. Nothing here is tuned on the evaluator data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .decoder import HMMConfig
from .graph import list_trip_ids
from .metrics import pointwise_accuracy
from .runner import prepare, run
from .scoring import ClassifierScorer


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", default="artifacts/models")
    ap.add_argument("--extra", nargs="*", default=[], help="extra artifact folders, e.g. artifacts (logistic)")
    ap.add_argument("--partition", default="student_test", choices=["student_test", "train"])
    ap.add_argument("--point-labels-dir", default=None)
    ap.add_argument("--out", default="results/model_comparison_hmm")
    a = ap.parse_args()

    dirs = sorted(p for p in Path(a.models).iterdir() if (p / "model.joblib").exists()) + [Path(e) for e in a.extra]
    cfg = HMMConfig()
    trips = list_trip_ids(a.data_dir, a.partition)
    rows = []
    for d in dirs:
        scorer = ClassifierScorer(d)
        acc = {"cls": [], "hmm": []}
        res = []
        for tid in trips:
            prep = prepare(a.data_dir, tid, scorer=scorer)
            r = run(prep, cfg, scorer)
            res.append(r)
            if a.point_labels_dir:
                lab = np.load(Path(a.point_labels_dir) / f"{tid}.npy")
                acc["cls"].append(pointwise_accuracy(r["argmax_per_point"], lab))
                acc["hmm"].append(pointwise_accuracy(r["hmm_per_point"], lab))
        row = dict(model=d.name if d.name != "artifacts" else "logistic_regression", n_trips=len(trips))
        for key, name in (("classifier_only", "cls"), ("hmm", "hmm")):
            for m in ("route_f1", "continuity", "jumps"):
                row[f"{name}_{m}"] = float(np.mean([r[key][m] for r in res]))
            row[f"{name}_pointwise_acc"] = float(np.nanmean(acc[name])) if acc[name] else float("nan")
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows).sort_values("hmm_route_f1", ascending=False)
    df.to_csv(a.out + ".csv", index=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(x - .2, df.cls_route_f1, .4, label="classifier-only"); ax[0].bar(x + .2, df.hmm_route_f1, .4, label="classifier + HMM")
    ax[0].set_ylabel("route F1 (student_test mean)"); ax[0].set_ylim(0, 1.05)
    ax[1].bar(x - .2, df.cls_pointwise_acc, .4, label="classifier-only"); ax[1].bar(x + .2, df.hmm_pointwise_acc, .4, label="classifier + HMM")
    ax[1].set_ylabel("pointwise accuracy vs pseudo-labels"); ax[1].set_ylim(0, 1.05)
    for axi in ax:
        axi.set_xticks(x, df.model, rotation=30, ha="right"); axi.legend()
    fig.suptitle("Each model as HMM emission (default decoder config, student_test)")
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
