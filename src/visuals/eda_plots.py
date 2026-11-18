"""EDA figures for the report (Person 1's missing deliverable, added by Person 4).

    python -m src.visuals.eda_plots --data-root "<Student folder>" --results results

Reads the raw trips (train + student_test only) and the candidate tables, and writes PNGs to
reports/figures/eda/. Every number in the figures is computed here from the data.
Speed is computed in a per-trip local metric projection (metres), not on raw degrees.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.raw import list_trips, load_trip, to_local_xy


def trip_stats(root, split):
    rows, speeds, gaps = [], [], []
    for folder in list_trips(root, split):
        t = load_trip(folder, split)
        xy = to_local_xy(t.track[:, :2], t.track[0, 0], t.track[0, 1])
        dt = np.diff(t.track[:, 2])
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        ok = dt > 0
        speeds.append(step[ok] / dt[ok])
        gaps.append(dt)
        rows.append(dict(split=split, trip_id=t.trip_id, n_points=len(t.track),
                         duration_min=(t.track[-1, 2] - t.track[0, 2]) / 60,
                         length_km=step.sum() / 1000, n_arcs=len(t.arcs), n_nodes=len(t.nodes),
                         route_arcs=len(t.route), median_dt_s=float(np.median(dt)),
                         lon=t.track[0, 0], lat=t.track[0, 1]))
    return pd.DataFrame(rows), np.concatenate(speeds), np.concatenate(gaps)


def hist(ax, data, title, xlabel, bins=40, logy=False):
    ax.hist(data, bins=bins, color="#4c78a8")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    if logy:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures/eda"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    parts = [trip_stats(a.data_root, s) for s in ("train", "student_test")]
    stats = pd.concat([p[0] for p in parts], ignore_index=True)
    speeds = np.concatenate([p[1] for p in parts])
    gaps = np.concatenate([p[2] for p in parts])
    stats.to_csv(a.out / "trip_summary.csv", index=False)

    # 1. per-trip size / length / network
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    hist(ax[0, 0], stats.n_points, "GPS points per trip", "points", 20)
    hist(ax[0, 1], stats.duration_min, "Trip duration", "minutes", 20)
    hist(ax[0, 2], stats.length_km, "Trip length (GPS polyline)", "km", 20)
    hist(ax[1, 0], stats.n_arcs, "Road arcs in each trip's network", "arcs", 20)
    hist(ax[1, 1], stats.route_arcs, "Arcs in reference route", "arcs", 20)
    hist(ax[1, 2], stats.median_dt_s, "Median sampling interval per trip", "seconds", 20)
    fig.suptitle(f"Trip-level overview ({len(stats)} trips: train + student_test)")
    fig.tight_layout(); fig.savefig(a.out / "01_trip_overview.png", dpi=150); plt.close(fig)

    # 2. point-level: speed + sampling gap
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    hist(ax[0], np.clip(speeds, 0, 60), "Speed between consecutive fixes (clipped at 60 m/s)", "m/s", 60, True)
    hist(ax[1], np.clip(gaps, 0, 60), "Time gap between fixes (clipped at 60 s)", "seconds", 60, True)
    fig.tight_layout(); fig.savefig(a.out / "02_speed_and_gaps.png", dpi=150); plt.close(fig)

    # 3. where the trips are (start points)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, c in (("train", "#4c78a8"), ("student_test", "#e45756")):
        d = stats[stats.split == s]
        ax.scatter(d.lon, d.lat, s=18, c=c, label=s, alpha=.8)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title("Trip start locations (trips span several regions)")
    fig.tight_layout(); fig.savefig(a.out / "03_trip_locations.png", dpi=150); plt.close(fig)

    # 4. candidate table: features vs pseudo-label
    cand_path = a.results / "train_candidates.csv"
    if cand_path.exists():
        c = pd.read_csv(cand_path, dtype={"trip_id": str})
        feats = ["distance_m", "heading_alignment", "speed_mps", "arc_length_m", "progress_on_arc", "endpoint_distance_m"]
        fig, ax = plt.subplots(2, 3, figsize=(14, 7))
        for axi, f in zip(ax.ravel(), feats):
            lo, hi = c[f].quantile([0.005, 0.995])
            bins = np.linspace(lo, hi, 50)
            axi.hist(c.loc[c.label == 0, f].clip(lo, hi), bins, alpha=.6, density=True, label="label 0", color="#9aa4b2")
            axi.hist(c.loc[c.label == 1, f].clip(lo, hi), bins, alpha=.6, density=True, label="label 1", color="#e45756")
            axi.set_title(f); axi.legend()
        fig.suptitle("Train candidate features by pseudo-label (density)")
        fig.tight_layout(); fig.savefig(a.out / "04_features_by_label.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        counts = c.label.value_counts().sort_index()
        ax[0].bar(["not true arc (0)", "true arc (1)"], counts.values, color=["#9aa4b2", "#e45756"])
        ax[0].set_title(f"Class balance ({counts[1] / counts.sum():.1%} positive)"); ax[0].set_ylabel("candidate rows")
        fcols = ["distance_m", "log_distance_m", "arc_length_m", "heading_alignment", "speed_mps",
                 "log_dt_prev_s", "progress_on_arc", "endpoint_distance_m", "out_degree_from", "in_degree_to"]
        corr = c[fcols + ["label"]].corr()
        im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax[1].set_xticks(range(len(corr)), corr.columns, rotation=75, fontsize=7)
        ax[1].set_yticks(range(len(corr)), corr.columns, fontsize=7)
        ax[1].set_title("Feature / label correlation"); fig.colorbar(im, ax=ax[1], fraction=.046)
        fig.tight_layout(); fig.savefig(a.out / "05_class_balance_correlation.png", dpi=150); plt.close(fig)

    # 5. pseudo-label coverage per trip (is the true arc among the 8 candidates?)
    aud_path = a.results / "train_reference_audit.csv"
    if aud_path.exists():
        au = pd.read_csv(aud_path, dtype={"trip_id": str})
        cov = au.groupby("trip_id").reference_in_candidates.mean().sort_values()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(len(cov)), cov.values, color="#4c78a8")
        ax[0].set_ylim(cov.min() - .01, 1.0); ax[0].set_xlabel("train trips (sorted)")
        ax[0].set_title(f"Candidate coverage per trip (mean {cov.mean():.1%})")
        ax[1].hist(au.pseudo_reference_distance_m.clip(0, 100), 50, color="#4c78a8")
        ax[1].set_title("Distance GPS point -> pseudo-reference arc (clipped 100 m)"); ax[1].set_xlabel("metres")
        fig.tight_layout(); fig.savefig(a.out / "06_candidate_coverage.png", dpi=150); plt.close(fig)

    print("wrote figures + trip_summary.csv to", a.out)
    print(stats.describe().round(2).to_string())


if __name__ == "__main__":
    main()
