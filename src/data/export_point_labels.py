"""Export per-GPS-point arc labels (Person 1's pseudo-labels) as <trip_id>.npy files.

    python -m src.data.export_point_labels --split train
    python -m src.data.export_point_labels --split student_test

Input : results/<split>_reference_audit.csv  (written by src.features.build_candidates)
Output: results/point_labels/<split>/<trip_id>.npy  -> int array, one arc id per GPS point
        (-1 = no label). Arc ids are LOCAL to each trip.

CAUTION: these are Person 1's heuristic pseudo-labels (nearest arc on the reference route),
NOT GPS-aligned ground truth. Pointwise accuracy computed from them is only "agreement with
that heuristic". Use only for train / student_test, never for evaluator data.
The files are consumed by `python -m src.hmm.compare --point-labels-dir <dir>`.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "student_test"], required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args()
    audit = pd.read_csv(a.results / f"{a.split}_reference_audit.csv", dtype={"trip_id": str})
    out = a.results / "point_labels" / a.split
    out.mkdir(parents=True, exist_ok=True)
    for tid, g in audit.groupby("trip_id"):
        lab = np.full(int(g["obs_idx"].max()) + 1, -1, dtype=np.int64)
        lab[g["obs_idx"].to_numpy()] = g["pseudo_reference_arc_id"].to_numpy()
        np.save(out / f"{tid}.npy", lab)
    print(f"wrote {audit['trip_id'].nunique()} label files to {out}")


if __name__ == "__main__":
    main()
