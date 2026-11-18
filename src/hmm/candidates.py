"""Minimal candidate generator used ONLY so the decoder/API can be developed and
tested before Person 1's generator (src/features) is ready.

TODO(integration): replace with the team's candidate generation / feature table.
Interface the decoder needs: for each GPS point, an array of candidate arc ids.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import shapely
from shapely import STRtree

from .graph import RoadGraph


class ArcIndex:
    """STRtree over arc segments (metres) for radius queries."""

    def __init__(self, graph: RoadGraph):
        self.graph = graph
        a = graph.nodes_xy[graph.arc_from]
        b = graph.nodes_xy[graph.arc_to]
        self.lines = shapely.linestrings(np.stack([a, b], axis=1))
        self.tree = STRtree(self.lines)

    def candidates(self, xy: np.ndarray, radius_m: float = 50.0,
                   max_k: int = 15) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Per point: arc ids (nearest first, at most max_k) and perpendicular distances."""
        pts = shapely.points(xy)
        qi, ti = self.tree.query(pts, predicate="dwithin", distance=radius_m)
        order = np.argsort(qi, kind="stable")
        qi, ti = qi[order], ti[order]
        starts = np.searchsorted(qi, np.arange(len(xy) + 1))
        arcs_out, dist_out = [], []
        for k in range(len(xy)):
            ids = ti[starts[k]:starts[k + 1]]
            if len(ids) == 0:
                arcs_out.append(np.empty(0, dtype=np.int64))
                dist_out.append(np.empty(0))
                continue
            _, d = self.graph.project_on_arcs(xy[k], ids)
            keep = np.argsort(d)[:max_k]
            arcs_out.append(ids[keep].astype(np.int64))
            dist_out.append(d[keep])
        return arcs_out, dist_out
