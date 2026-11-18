"""Distance-only baseline: nearest-arc argmax vs. the same scores + HMM/Viterbi.

Needs no trained classifier, so it can run before the model work is done. When the
real classifier exists, swap `log_emission` for `proba_log_emission(model_probs)`.
Metrics are route-level (reference .route); pointwise accuracy is added once the
shared point->arc label alignment exists.

    python -m src.hmm.run_baseline --data-dir <Student folder> --partition train --limit 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import ArcIndex
from .decoder import HMMConfig, MapMatchHMM, gaussian_log_emission
from .graph import list_trip_ids, load_trip
from .metrics import n_discontinuities, route_continuity, route_overlap, dedup


def run_trip(data_dir, trip_id, cfg: HMMConfig, radius_m=50.0, max_k=15):
    trip = load_trip(data_dir, trip_id)
    g = trip.graph
    t0 = time.perf_counter()
    arcs, dist = ArcIndex(g).candidates(trip.track_xy, radius_m=radius_m, max_k=max_k)
    t_cand = time.perf_counter() - t0
    em = [gaussian_log_emission(d, cfg.sigma_m) for d in dist]

    # candidate miss: reference route arc set never contains any candidate of the point
    ref = set(map(int, trip.route))
    cand_hit = np.mean([bool(ref & set(map(int, a))) if len(a) else False for a in arcs])
    no_cand = float(np.mean([len(a) == 0 for a in arcs]))

    argmax = np.array([a[np.argmax(e)] if len(a) else -1 for a, e in zip(arcs, em)])
    dec = MapMatchHMM(g, cfg)
    t0 = time.perf_counter()
    res = dec.decode(trip.track_xy, trip.track_t, arcs, em)
    t_dec = time.perf_counter() - t0
    hmm_route = dec.expand_route(res.arc_ids, res.segment_id)

    row = dict(trip=trip_id, n_points=len(trip.track_xy), n_arcs=g.n_arcs,
               ref_route_arcs=len(trip.route), cand_hit_rate=cand_hit, no_candidate_rate=no_cand,
               t_candidates_s=t_cand, t_decode_s=t_dec, hmm_breaks=res.n_breaks)
    for name, arcs_pred in [("argmax", dedup(argmax)), ("hmm", np.array(hmm_route))]:
        row[f"{name}_continuity"] = route_continuity(arcs_pred, g)
        row[f"{name}_jumps"] = n_discontinuities(arcs_pred, g)
        for k, v in route_overlap(arcs_pred, trip.route, g).items():
            row[f"{name}_{k}"] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--partition", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trips", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--sigma", type=float, default=15.0)
    a = ap.parse_args()
    cfg = HMMConfig(beta_m=a.beta, sigma_m=a.sigma)
    ids = a.trips or list_trip_ids(a.data_dir, a.partition)[: a.limit]
    rows = []
    for tid in ids:
        r = run_trip(a.data_dir, tid, cfg, radius_m=a.radius)
        rows.append(r)
        print(f"{tid}: F1 argmax {r['argmax_route_f1']:.3f} -> hmm {r['hmm_route_f1']:.3f} | "
              f"jumps {r['argmax_jumps']} -> {r['hmm_jumps']} | cand-hit {r['cand_hit_rate']:.2f} | "
              f"decode {r['t_decode_s']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
    print(df.drop(columns=["trip"]).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
