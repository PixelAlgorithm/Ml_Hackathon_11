"""Server-side prediction pipeline.

    trip_id -> load track + road network (NO labels) -> candidates -> emission scores
            -> classifier-only argmax  and  HMM/Viterbi route -> metrics vs reference

The reference route (.route) is read only inside `evaluate()`, after prediction, and is
never passed to the scorer, the decoder or the UI.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.hmm.candidates import ArcIndex
from src.hmm.decoder import HMMConfig, MapMatchHMM
from src.hmm.scoring import ClassifierScorer, DistanceScorer, Scorer, build_scorer   # noqa: F401 (re-exported)
from src.hmm.folium_map import arcs_to_latlon
from src.hmm.graph import Trip, find_trip_dir, list_trip_ids, load_trip
from src.hmm.metrics import dedup, n_discontinuities, route_continuity, route_overlap
from src.hmm.graph import _read_ws

log = logging.getLogger("pathsnap.pipeline")


def load_hmm_config(artifacts_dir: Path, scorer_kind: Optional[str] = None) -> HMMConfig:
    """artifacts/hmm_config.json (written by src.hmm.tune_decoder) or the defaults."""
    p = Path(artifacts_dir) / "hmm_config.json"
    if p.exists():
        raw = json.loads(p.read_text())
        cfg = HMMConfig.from_dict(raw)
        if scorer_kind and raw.get("tuned_for_scorer") not in (None, scorer_kind):
            log.warning("hmm_config.json was tuned for scorer=%r but PATHSNAP_SCORER=%r -- re-tune",
                        raw["tuned_for_scorer"], scorer_kind)
        log.info("HMM config loaded from %s: %s", p, cfg.to_dict())
        return cfg
    log.warning("no %s -> using default HMMConfig (run src.hmm.tune_decoder)", p)
    return HMMConfig()


class Pipeline:
    def __init__(self, data_dir: Path, scorer: Scorer, hmm_cfg: Optional[HMMConfig] = None,
                 partition: Optional[str] = None, radius_m: float = 50.0, max_k: int = 15,
                 cache_trips: int = 32):
        self.data_dir, self.scorer = Path(data_dir), scorer
        self.cfg = hmm_cfg or HMMConfig()
        self.partition, self.radius_m, self.max_k = partition, radius_m, max_k
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._cache_trips = cache_trips
        self._lock = threading.RLock()   # caches + Dijkstra memo are not thread-safe

    def trip_ids(self) -> List[str]:
        return list_trip_ids(self.data_dir, self.partition)

    def _load(self, trip_id: str):
        with self._lock:
            return self._load_locked(trip_id)

    def _load_locked(self, trip_id: str):
        if trip_id in self._cache:
            self._cache.move_to_end(trip_id)
            return self._cache[trip_id]
        if trip_id not in set(self.trip_ids()):
            raise KeyError(trip_id)
        trip = load_trip(self.data_dir, trip_id, load_route=False)   # labels NOT loaded
        entry = (trip, ArcIndex(trip.graph))
        self._cache[trip_id] = entry
        while len(self._cache) > self._cache_trips:
            self._cache.popitem(last=False)
        return entry

    def preload(self, warmup: bool = True):
        """Load every graph + spatial index; optionally run one full prediction per trip so
        (a) the first demo request is fast and (b) a broken trip fails at startup, not live.
        Warm-up never reads reference routes."""
        for tid in self.trip_ids():
            t0 = time.perf_counter()
            self._load(tid)
            if warmup:
                self.predict(tid)
            log.info("preloaded%s %s in %.1fs", "+warmed" if warmup else "", tid, time.perf_counter() - t0)

    # ---------------------------------------------------------------- predict
    def predict(self, trip_id: str) -> dict:
        with self._lock:
            return self._predict(trip_id)

    def _predict(self, trip_id: str) -> dict:
        t0 = time.perf_counter()
        trip, index = self._load(trip_id)
        g = trip.graph
        t1 = time.perf_counter()
        if getattr(self.scorer, "provides_candidates", False):
            arcs, dist, scores = self.scorer.candidates_and_scores(trip)   # Person 1's candidates
            t2 = time.perf_counter()
        else:
            arcs, dist = index.candidates(trip.track_xy, self.radius_m, self.max_k)
            t2 = time.perf_counter()
            scores = self.scorer.score(trip, arcs, dist)
        t3 = time.perf_counter()
        argmax = np.array([a[np.argmax(s)] if len(a) else -1 for a, s in zip(arcs, scores)])
        dec = MapMatchHMM(g, self.cfg)
        res = dec.decode(trip.track_xy, trip.track_t, arcs, scores)
        route = dec.expand_route(res.arc_ids, res.segment_id)
        t4 = time.perf_counter()
        return dict(trip=trip, graph=g, argmax=argmax, hmm_arcs=res.arc_ids, route=route,
                    breaks=res.n_breaks, empty=res.n_empty_steps,
                    timings=dict(load_ms=(t1 - t0) * 1e3, candidates_ms=(t2 - t1) * 1e3,
                                 scoring_ms=(t3 - t2) * 1e3, hmm_ms=(t4 - t3) * 1e3,
                                 predict_ms=(t4 - t0) * 1e3))

    # --------------------------------------------------------------- evaluate
    def _reference(self, trip_id: str) -> Optional[np.ndarray]:
        p = find_trip_dir(self.data_dir, trip_id) / f"{trip_id}.route"
        return _read_ws(p, dtype=np.int64).ravel() if p.exists() else None

    def evaluate(self, pred: dict) -> dict:
        g, trip = pred["graph"], pred["trip"]
        ref = self._reference(trip.trip_id)
        out = {}
        for name, arcs in [("hmm", pred["route"]), ("classifier_only", dedup(pred["argmax"]))]:
            m = route_overlap(arcs, ref, g) if ref is not None else dict(
                route_precision=None, route_recall=None, route_f1=None)   # no reference -> null
            m["route_continuity"] = route_continuity(arcs, g)
            m["n_discontinuities"] = n_discontinuities(arcs, g)
            out[name] = m
        return out

    def run(self, trip_id: str) -> dict:
        pred = self.predict(trip_id)
        t0 = time.perf_counter()
        metrics = self.evaluate(pred)
        pred["timings"]["evaluate_ms"] = (time.perf_counter() - t0) * 1e3
        g, trip = pred["graph"], pred["trip"]
        argmax_route = [int(a) for a in dedup(pred["argmax"])]
        return dict(
            trip_id=trip.trip_id, scorer=self.scorer.name, n_points=len(trip.track_xy),
            gps_latlon=[[float(la), float(lo)] for lo, la in trip.track_lonlat],
            predicted_route_arcs=[int(a) for a in pred["route"]],
            predicted_route_latlon=[[list(map(float, p)) for p in seg] for seg in arcs_to_latlon(g, pred["route"])],
            classifier_only_latlon=[[list(map(float, p)) for p in seg] for seg in arcs_to_latlon(g, argmax_route)],
            per_point_arc=[int(a) for a in pred["hmm_arcs"]],
            metrics_hmm=metrics["hmm"], metrics_classifier_only=metrics["classifier_only"],
            hmm_breaks=pred["breaks"], points_without_candidates=pred["empty"],
            timings_ms=pred["timings"])
