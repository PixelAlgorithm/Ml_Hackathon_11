"""Road network for one PathSnap trip (Person 4 / decoder side).

Raw files are whitespace separated, headerless:
    .nodes : lon lat             (row index = node id)
    .arcs  : from_node to_node   (row index = arc id, directed)
    .track : lon lat t_seconds
    .route : arc_id per row      (reference route, chronological)

Row i of .track is NOT row i of .nodes / .arcs. Arc geometry is the straight
segment between its two nodes (there are no shape points in the data).

All distance work is done in a local UTM projection (metres), never in degrees.
This module is deliberately self-contained so the decoder can be built and
tested without depending on src/data or src/features.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer


def _read_ws(path: Path, dtype=float) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def make_projector(lon: float, lat: float) -> Transformer:
    """WGS84 lon/lat -> UTM (metres) for the zone containing (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


class RoadGraph:
    """Directed arc graph with bounded shortest-path queries between arcs."""

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray,
                 projector: Optional[Transformer] = None, cache_size: int = 200_000):
        nodes_lonlat = np.asarray(nodes_lonlat, dtype=float)
        arcs = np.asarray(arcs, dtype=np.int64).reshape(-1, 2)
        n = len(nodes_lonlat)
        if len(arcs) and (arcs.min() < 0 or arcs.max() >= n):
            raise ValueError("arc references a node id outside .nodes")
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs
        self.arc_from = arcs[:, 0]
        self.arc_to = arcs[:, 1]
        self.projector = projector or make_projector(*nodes_lonlat.mean(axis=0))
        x, y = self.projector.transform(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.nodes_xy = np.column_stack([x, y])
        a = self.nodes_xy[self.arc_from]
        b = self.nodes_xy[self.arc_to]
        self.arc_vec = b - a
        self.arc_len = np.hypot(self.arc_vec[:, 0], self.arc_vec[:, 1])
        # CSR adjacency: node -> outgoing arc ids
        order = np.argsort(self.arc_from, kind="stable")
        self._out_arcs = order
        self._out_ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(self._out_ptr, self.arc_from[order] + 1, 1)
        self._out_ptr = np.cumsum(self._out_ptr)
        self._cache: Dict[int, Tuple[Dict[int, float], Dict[int, int]]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ basics
    @property
    def n_nodes(self) -> int:
        return len(self.nodes_xy)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float).reshape(-1, 2)
        x, y = self.projector.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        inv = Transformer.from_crs(self.projector.target_crs, "EPSG:4326", always_xy=True)
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        lon, lat = inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])

    def arc_bearing(self, arc_ids: np.ndarray) -> np.ndarray:
        """Compass-style bearing in radians (0 = north, clockwise) of each arc."""
        v = self.arc_vec[arc_ids]
        return np.arctan2(v[:, 0], v[:, 1])

    def reverse_arc_mask(self, i: int, j: int) -> bool:
        return bool(self.arc_from[i] == self.arc_to[j] and self.arc_to[i] == self.arc_from[j])

    # ------------------------------------------------------------- projection
    def project_on_arcs(self, p: np.ndarray, arc_ids: np.ndarray):
        """Project point p (x, y metres) on each arc.

        Returns (offset_m along the arc from its start, perpendicular distance_m).
        """
        a = self.nodes_xy[self.arc_from[arc_ids]]
        v = self.arc_vec[arc_ids]
        L2 = np.maximum((v ** 2).sum(axis=1), 1e-9)
        t = np.clip(((p - a) * v).sum(axis=1) / L2, 0.0, 1.0)
        foot = a + v * t[:, None]
        dist = np.hypot(*(p - foot).T)
        return t * self.arc_len[arc_ids], dist

    # ----------------------------------------------------- bounded Dijkstra
    def _dijkstra(self, src: int, cutoff: float):
        key = src
        hit = self._cache.get(key)
        if hit is not None and hit[2] >= cutoff:
            return hit[0], hit[1]
        dist: Dict[int, float] = {src: 0.0}
        pred: Dict[int, int] = {}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for e in self.out_arcs(u):
                nd = d + self.arc_len[e]
                if nd > cutoff:
                    continue
                v = int(self.arc_to[e])
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = int(e)
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = (dist, pred, cutoff)
        return dist, pred

    def node_distances(self, src_node: int, cutoff: float) -> Dict[int, float]:
        """Network distance (m) from src_node to every node within cutoff.

        The cache may hold a search done with a larger cutoff; entries beyond the
        requested cutoff are filtered out so results never depend on cache history."""
        d = self._dijkstra(int(src_node), cutoff)[0]
        return d if max(d.values()) <= cutoff else {k: v for k, v in d.items() if v <= cutoff}

    def path_arcs(self, src_node: int, dst_node: int, cutoff: float):
        """Arc ids of the shortest path src_node -> dst_node, or None if > cutoff."""
        dist, pred = self._dijkstra(int(src_node), cutoff)
        # the cache may hold a search run with a LARGER cutoff, so re-check the bound
        if dst_node not in dist or dist[dst_node] > cutoff:
            return None
        out, v = [], int(dst_node)
        while v != src_node:
            e = pred[v]
            out.append(e)
            v = int(self.arc_from[e])
        return out[::-1]

    # ------------------------------------------------------------ diagnostics
    def weak_components(self):
        """Number of weakly connected components and per-node labels."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        m = coo_matrix((np.ones(self.n_arcs), (self.arc_from, self.arc_to)),
                       shape=(self.n_nodes, self.n_nodes))
        return connected_components(m, directed=True, connection="weak")

    def isolated_nodes(self) -> int:
        deg = np.bincount(self.arc_from, minlength=self.n_nodes) + \
            np.bincount(self.arc_to, minlength=self.n_nodes)
        return int((deg == 0).sum())


@dataclass
class Trip:
    trip_id: str
    graph: RoadGraph
    track_lonlat: np.ndarray   # (T, 2) lon, lat
    track_t: np.ndarray        # (T,) seconds
    track_xy: np.ndarray       # (T, 2) metres, same projection as graph
    route: Optional[np.ndarray]  # reference arc ids (server-side labels), may be None


def find_trip_dir(root: Path | str, trip_id: str) -> Path:
    """Locate <root>/[partition/]<trip_id>/ regardless of partition folder."""
    root = Path(root)
    for cand in [root / trip_id, *sorted(root.glob(f"*/{trip_id}"))]:
        if (cand / f"{trip_id}.track").exists():
            return cand
    raise FileNotFoundError(f"trip {trip_id} not found under {root}")


def load_trip(root: Path | str, trip_id: str, load_route: bool = True) -> Trip:
    d = find_trip_dir(root, trip_id)
    p = d / trip_id
    nodes = _read_ws(p.with_suffix(".nodes"))
    arcs = _read_ws(p.with_suffix(".arcs"), dtype=np.int64)
    track = _read_ws(p.with_suffix(".track"))
    route = None
    rp = p.with_suffix(".route")
    if load_route and rp.exists():
        route = _read_ws(rp, dtype=np.int64).ravel()
    # one projection per trip, anchored on the track so graph and GPS agree
    proj = make_projector(*track[:, :2].mean(axis=0))
    g = RoadGraph(nodes, arcs, projector=proj)
    return Trip(trip_id, g, track[:, :2], track[:, 2], g.to_xy(track[:, :2]), route)


def list_trip_ids(root: Path | str, partition: Optional[str] = None) -> list[str]:
    """Trip ids that have a .track file under root (optionally one partition folder)."""
    base = Path(root) / partition if partition else Path(root)
    return sorted({t.stem for t in base.rglob("*.track")})
