"""Emission scorers: turn (trip, candidate arcs) into per-candidate log-emission scores.

Two implementations:
  * DistanceScorer   - Gaussian on point-to-arc distance. NO ML model; a dev baseline only.
  * ClassifierScorer - the team's trained classifier: features -> predict_proba -> log.

ClassifierScorer contract (what teammates must provide)
-------------------------------------------------------
artifacts/model.joblib          fitted estimator with predict_proba (a sklearn Pipeline
                                that already contains its scaler is fine)
artifacts/preprocessor.joblib   OPTIONAL separate transformer, fitted on TRAIN trips only,
                                applied before the model
artifacts/manifest.json         OPTIONAL {"positive_class": 1, "temperature": 1.0}
src.features.build_candidate_features(trip, cand_arcs, cand_dist) -> list of arrays,
                                one (K_t, F) array per GPS step, rows aligned with
                                cand_arcs[t]. `trip` carries graph, track_xy (metres),
                                track_lonlat, track_t and NO labels (trip.route is None).

If any piece is missing the constructor raises -- there is deliberately NO silent
fallback to distance scoring, so a mis-deployed demo cannot look like it works.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import List, Protocol

import numpy as np

from .decoder import gaussian_log_emission, proba_log_emission
from .graph import Trip


class Scorer(Protocol):
    name: str

    def score(self, trip: Trip, cand_arcs: List[np.ndarray], cand_dist: List[np.ndarray]) -> List[np.ndarray]:
        """Per-step log-emission scores aligned with cand_arcs."""


class DistanceScorer:
    """Placeholder emission: Gaussian on point-to-arc distance (no ML model)."""
    name = "distance-gaussian"
    static = False        # depends on sigma_m, which the tuner varies -> do not cache scores

    def __init__(self, sigma_m: float = 15.0):
        self.sigma_m = sigma_m

    def score(self, trip, cand_arcs, cand_dist):
        return [gaussian_log_emission(d, self.sigma_m) for d in cand_dist]


class ClassifierScorer:
    """Emission = log P(candidate is the true arc | features) from the trained model."""
    name = "classifier"
    static = True         # scores do not depend on decoder params -> cacheable per trip
    FEATURE_FN = "build_candidate_features"

    def __init__(self, artifacts_dir: Path | str, feature_module: str = "src.features"):
        import joblib   # imported here so distance-only runs need no sklearn
        d = Path(artifacts_dir)
        model_path = d / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Ask the model owner to export the selected model "
                f"with joblib.dump(model, 'artifacts/model.joblib').")
        try:
            mod = importlib.import_module(feature_module)
        except ImportError as e:
            raise RuntimeError(f"cannot import {feature_module}: {e}") from e
        self._features = getattr(mod, self.FEATURE_FN, None)
        if self._features is None and not hasattr(mod, "build_inference_candidates"):
            try:   # Person 1 keeps the builder in the submodule src.features.build_candidates
                mod = importlib.import_module(feature_module + ".build_candidates")
            except ImportError:
                pass
        # Mode B (Person 1's builder): it generates its OWN candidates + features per trip.
        self._own = getattr(mod, "build_inference_candidates", None) if self._features is None else None
        if self._features is None and self._own is None:
            raise RuntimeError(
                f"{feature_module}.{self.FEATURE_FN}(trip, cand_arcs, cand_dist) does not exist yet. "
                f"The classifier cannot be used without the team's feature builder "
                f"(see the contract in src/hmm/scoring.py).")
        self.feature_columns = list(getattr(mod, "FEATURE_COLUMNS", []))
        self.provides_candidates = self._own is not None
        self.model = joblib.load(model_path)
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"{type(self.model).__name__} has no predict_proba; the HMM needs "
                            f"probabilities (calibrate it or pick a probabilistic model).")
        pre = d / "preprocessor.joblib"
        self.preprocessor = joblib.load(pre) if pre.exists() else None
        man = d / "manifest.json"
        meta = json.loads(man.read_text()) if man.exists() else {}
        self.temperature = float(meta.get("temperature", 1.0))
        classes = list(getattr(self.model, "classes_", [0, 1]))
        pos = meta.get("positive_class", 1)
        if pos not in classes:
            raise ValueError(f"positive class {pos!r} not in model.classes_ {classes}")
        self._pos_col = classes.index(pos)

    def candidates_and_scores(self, trip: Trip):
        """Mode B: candidates, distances and log-emissions from Person 1's own builder.

        Our Trip is converted to their Trip (lon/lat track + nodes + arcs, route=None, so no
        labels can reach the builder). Their rows come ordered (obs_idx, rank) with top_k
        candidates per GPS point; arc ids stay local to this trip."""
        from src.data.raw import Trip as P1Trip
        g = trip.graph
        track = np.column_stack([trip.track_lonlat, trip.track_t])
        t1 = P1Trip(trip.trip_id, "inference", track, g.nodes_lonlat, g.arcs, None)
        df, _ = self._own(t1)
        X = df[self.feature_columns]
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        p = self.model.predict_proba(X)[:, self._pos_col]
        T = len(trip.track_t)
        order = np.argsort(df["obs_idx"].to_numpy(), kind="stable")
        obs = df["obs_idx"].to_numpy()[order]
        arcs_all, dist_all, p = df["arc_id"].to_numpy()[order], df["distance_m"].to_numpy()[order], p[order]
        cuts = np.searchsorted(obs, np.arange(T + 1))
        arcs = [arcs_all[cuts[k]:cuts[k + 1]].astype(np.int64) for k in range(T)]
        dist = [dist_all[cuts[k]:cuts[k + 1]] for k in range(T)]
        scores = [proba_log_emission(p[cuts[k]:cuts[k + 1]], self.temperature) for k in range(T)]
        return arcs, dist, scores

    def score(self, trip, cand_arcs, cand_dist):
        if self.provides_candidates:
            raise RuntimeError("this scorer supplies its own candidates; use candidates_and_scores()")
        feats = self._features(trip, cand_arcs, cand_dist)
        if len(feats) != len(cand_arcs):
            raise ValueError("feature builder returned a different number of steps than candidates")
        counts = [len(a) for a in cand_arcs]
        for k, (f, n) in enumerate(zip(feats, counts)):
            if len(f) != n:
                raise ValueError(f"step {k}: {n} candidates but {len(f)} feature rows")
        if sum(counts) == 0:
            return [np.empty(0) for _ in counts]
        X = np.vstack([np.asarray(f, dtype=float) for f, n in zip(feats, counts) if n])
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        p = self.model.predict_proba(X)[:, self._pos_col]          # one batched call per trip
        return [proba_log_emission(chunk, self.temperature) for chunk in np.split(p, np.cumsum(counts)[:-1])]


def build_scorer(kind: str, artifacts_dir: Path | str) -> Scorer:
    if kind == "distance":
        return DistanceScorer()
    if kind == "classifier":
        return ClassifierScorer(artifacts_dir)
    raise ValueError(f"unknown PATHSNAP_SCORER={kind!r} (use 'distance' or 'classifier')")
