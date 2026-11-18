"""HMM map-matching decoder.

Hidden state at GPS step t = one *candidate road segment (arc)* for that point;
the candidate -> arc_id correspondence is preserved end to end.

Emission   : supplied by the caller as log-scores per candidate. Two helpers are
             provided: `gaussian_log_emission` (distance only; works with no model)
             and `proba_log_emission` (classifier probabilities, optionally tempered).
Transition : Newson & Krumm (2009) style. For consecutive observations a -> b,
                 d_gc    = straight-line distance between the two GPS fixes
                 d_route = network distance between the projections on the two arcs
                 log P   = -|d_gc - d_route| / beta - log beta        (- uturn_penalty)
             Unreachable pairs (d_route > route_cutoff_m) get -inf.
Breaks     : steps with no candidates are skipped (the chain bridges over them);
             a time gap > max_gap_s, or a step none of whose candidates is reachable,
             restarts the chain (new segment) instead of forcing a bogus link.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import numpy as np

from .graph import RoadGraph
from .viterbi import NEG_INF, viterbi


EMISSION_FLOOR = -1e4   # finite stand-in for log(0) in emissions


@dataclass
class HMMConfig:
    beta_m: float = 8.0            # scale of |d_gc - d_route| (m)
    route_cutoff_m: float = 600.0  # max network distance explored between steps
    max_gap_s: float = 15.0        # larger time gap between used fixes -> chain break
    backward_tol_m: float = 25.0   # tolerated backward jitter on the same arc (m)
    uturn_penalty: float = 4.0     # log-penalty for hopping onto the reverse arc
    emission_weight: float = 1.0   # scales the emission log-scores vs transitions
    sigma_m: float = 15.0          # used by gaussian_log_emission only

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HMMConfig":
        """Build from a saved dict, ignoring unknown keys (forward compatible)."""
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in names})


def gaussian_log_emission(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Distance-only emission: log N(dist; 0, sigma)."""
    return -0.5 * (np.asarray(dist_m) / sigma_m) ** 2 - np.log(sigma_m * np.sqrt(2 * np.pi))


def proba_log_emission(p: np.ndarray, temperature: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """Classifier probability of 'this candidate is the true arc' -> log-emission."""
    return np.log(np.clip(np.asarray(p, dtype=float), eps, 1.0)) / temperature


@dataclass
class DecodeResult:
    arc_ids: np.ndarray        # (T,) chosen arc per GPS point, -1 where no candidate existed
    segment_id: np.ndarray     # (T,) chain segment id, -1 where no candidate existed
    n_breaks: int
    n_empty_steps: int
    config: dict


class MapMatchHMM:
    def __init__(self, graph: RoadGraph, config: Optional[HMMConfig] = None):
        self.g = graph
        self.cfg = config or HMMConfig()

    # ----------------------------------------------------------------- decode
    def decode(self, xy: np.ndarray, t: np.ndarray,
               cand_arcs: Sequence[np.ndarray],
               log_emission: Sequence[np.ndarray]) -> DecodeResult:
        """xy (T,2) metres; t (T,) seconds; cand_arcs/log_emission: per step arrays."""
        T = len(xy)
        if not (len(t) == T == len(cand_arcs) == len(log_emission)):
            raise ValueError("xy, t, cand_arcs and log_emission must have the same length")
        for k in range(T):
            if len(cand_arcs[k]) != len(log_emission[k]):
                raise ValueError(f"step {k}: {len(cand_arcs[k])} candidates but "
                                 f"{len(log_emission[k])} emission scores")
        used = [k for k in range(T) if len(cand_arcs[k]) > 0]
        out = np.full(T, -1, dtype=np.int64)
        seg = np.full(T, -1, dtype=np.int64)
        if not used:
            return DecodeResult(out, seg, 0, T, self.cfg.to_dict())

        offs = {}
        for k in used:
            off, _ = self.g.project_on_arcs(xy[k], np.asarray(cand_arcs[k]))
            offs[k] = off
        # NaN / -inf emissions would poison the whole step (every delta = -inf), so they
        # are mapped to a very low but finite log-score; the candidate is not deleted.
        emis = [self.cfg.emission_weight * np.nan_to_num(
                    np.asarray(log_emission[k], dtype=float),
                    nan=EMISSION_FLOOR, neginf=EMISSION_FLOOR, posinf=0.0).clip(min=EMISSION_FLOOR)
                for k in used]
        forced = [False] + [bool(t[used[i]] - t[used[i - 1]] > self.cfg.max_gap_s)
                            for i in range(1, len(used))]

        def trans(i: int) -> np.ndarray:
            return self._transition(xy, used[i - 1], used[i], cand_arcs, offs)

        path, segment = viterbi(emis, trans, forced)
        for i, k in enumerate(used):
            out[k] = cand_arcs[k][path[i]]
            seg[k] = segment[i]
        return DecodeResult(out, seg, int(max(segment)), T - len(used), self.cfg.to_dict())

    # ------------------------------------------------------------- transition
    def _transition(self, xy, ka, kb, cand_arcs, offs) -> np.ndarray:
        g, c = self.g, self.cfg
        A, B = np.asarray(cand_arcs[ka]), np.asarray(cand_arcs[kb])
        oa, ob = offs[ka], offs[kb]
        d_gc = float(np.hypot(*(xy[kb] - xy[ka])))
        M = np.full((len(A), len(B)), NEG_INF)
        for i, ai in enumerate(A):
            dist = g.node_distances(int(g.arc_to[ai]), c.route_cutoff_m)
            rest = g.arc_len[ai] - oa[i]
            for j, bj in enumerate(B):
                if bj == ai and ob[j] >= oa[i] - c.backward_tol_m:
                    d_route = max(ob[j] - oa[i], 0.0)
                else:
                    dn = dist.get(int(g.arc_from[bj]))
                    if dn is None or dn > c.route_cutoff_m:
                        continue
                    d_route = rest + dn + ob[j]
                lp = -abs(d_gc - d_route) / c.beta_m - np.log(c.beta_m)
                if bj != ai and g.reverse_arc_mask(ai, bj):
                    lp -= c.uturn_penalty
                M[i, j] = lp
        return M

    # ------------------------------------------------------------ route output
    def expand_route(self, arc_ids: np.ndarray, segment_id: Optional[np.ndarray] = None) -> List[int]:
        """Ordered distinct arcs of the predicted route; connecting arcs are filled in
        with the shortest path when consecutive decoded arcs are not adjacent
        (and the chain was not broken between them)."""
        g, c = self.g, self.cfg
        route: List[int] = []
        prev_arc, prev_seg = None, None
        for k, a in enumerate(arc_ids):
            if a < 0:
                continue
            s = None if segment_id is None else int(segment_id[k])
            if prev_arc is None:
                route.append(int(a))
            elif a != prev_arc:
                same_chain = segment_id is None or s == prev_seg
                if same_chain and g.arc_to[prev_arc] != g.arc_from[a]:
                    fill = g.path_arcs(int(g.arc_to[prev_arc]), int(g.arc_from[a]), c.route_cutoff_m)
                    if fill:
                        route.extend(int(e) for e in fill)
                route.append(int(a))
            prev_arc, prev_seg = int(a), s
        return route
