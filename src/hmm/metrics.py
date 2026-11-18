"""Route-level / sequence metrics that need only the reference route (arc list).

Pointwise accuracy needs per-GPS-point labels; those come from the shared label
alignment (Person 1). `pointwise_accuracy` is here so the same code is used everywhere.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .graph import RoadGraph


def dedup(arcs: Sequence[int]) -> np.ndarray:
    a = np.asarray([x for x in arcs if x >= 0], dtype=np.int64)
    if len(a) == 0:
        return a
    return a[np.insert(a[1:] != a[:-1], 0, True)]


def route_continuity(arcs: Sequence[int], graph: RoadGraph) -> float:
    """Fraction of consecutive *distinct* arcs that are directly connected
    (head of one = tail of the next). 1.0 means no jumps."""
    a = dedup(arcs)
    if len(a) < 2:
        return 1.0
    return float((graph.arc_to[a[:-1]] == graph.arc_from[a[1:]]).mean())


def n_discontinuities(arcs: Sequence[int], graph: RoadGraph) -> int:
    a = dedup(arcs)
    if len(a) < 2:
        return 0
    return int((graph.arc_to[a[:-1]] != graph.arc_from[a[1:]]).sum())


def route_overlap(pred_arcs: Sequence[int], ref_arcs: Sequence[int], graph: RoadGraph) -> Dict[str, float]:
    """Length-weighted precision/recall/F1 between predicted and reference arc sets."""
    p, r = set(map(int, dedup(pred_arcs))), set(map(int, dedup(ref_arcs)))
    L = graph.arc_len
    inter = sum(L[a] for a in p & r)
    lp, lr = sum(L[a] for a in p), sum(L[a] for a in r)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}


def pointwise_accuracy(pred: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred, labels = np.asarray(pred), np.asarray(labels)
    m = labels >= 0 if mask is None else mask
    return float((pred[m] == labels[m]).mean()) if m.any() else float("nan")
