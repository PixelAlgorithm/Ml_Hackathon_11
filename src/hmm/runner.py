"""Shared evaluation loop used by tuning, comparison and map scripts.

    prepare(trip)  -> candidates (+ cached classifier scores)   [expensive, once per trip]
    run(prep, cfg) -> classifier-only route, classifier+HMM route and their metrics

Only call this with train / student_test trips: it loads the reference route to score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .candidates import ArcIndex
from .decoder import HMMConfig, MapMatchHMM
from .graph import Trip, load_trip
from .metrics import dedup, n_discontinuities, route_continuity, route_overlap

DEV_PARTITIONS = ("train", "student_test")   # the only partitions development code may touch


@dataclass
class Prepared:
    trip: Trip
    arcs: List[np.ndarray]
    dist: List[np.ndarray]
    cache: Dict[str, List[np.ndarray]] = field(default_factory=dict)   # scorer.name -> scores


def prepare(data_dir, trip_id: str, radius_m: float = 50.0, max_k: int = 15, scorer=None) -> Prepared:
    trip = load_trip(data_dir, trip_id, load_route=True)
    if getattr(scorer, "provides_candidates", False):      # scorer brings its own candidates
        arcs, dist, scores = scorer.candidates_and_scores(trip)
        return Prepared(trip, arcs, dist, {scorer.name: scores})
    arcs, dist = ArcIndex(trip.graph).candidates(trip.track_xy, radius_m, max_k)
    return Prepared(trip, arcs, dist)


def scores_for(prep: Prepared, scorer) -> List[np.ndarray]:
    if getattr(scorer, "static", False):
        if scorer.name not in prep.cache:
            prep.cache[scorer.name] = scorer.score(prep.trip, prep.arcs, prep.dist)
        return prep.cache[scorer.name]
    return scorer.score(prep.trip, prep.arcs, prep.dist)


def route_metrics(arcs, ref, graph) -> Dict[str, float]:
    m = route_overlap(arcs, ref, graph)
    m["continuity"] = route_continuity(arcs, graph)
    m["jumps"] = n_discontinuities(arcs, graph)
    return m


def run(prep: Prepared, cfg: HMMConfig, scorer) -> dict:
    """Decode one prepared trip. Returns routes, per-point arcs and metrics for both methods."""
    t = prep.trip
    scores = scores_for(prep, scorer)
    argmax = np.array([a[np.argmax(s)] if len(a) else -1 for a, s in zip(prep.arcs, scores)])
    dec = MapMatchHMM(t.graph, cfg)
    t0 = time.perf_counter()
    res = dec.decode(t.track_xy, t.track_t, prep.arcs, scores)
    route = dec.expand_route(res.arc_ids, res.segment_id)
    dt = time.perf_counter() - t0
    cls_route = dedup(argmax)
    return dict(
        argmax_per_point=argmax, hmm_per_point=res.arc_ids, hmm_route=np.asarray(route),
        argmax_route=cls_route, breaks=res.n_breaks, empty=res.n_empty_steps, decode_s=dt,
        hmm=route_metrics(route, t.route, t.graph),
        classifier_only=route_metrics(cls_route, t.route, t.graph))
