"""Hand-made grid so the expected route is known exactly."""
import numpy as np
from pyproj import Transformer
from src.hmm.graph import RoadGraph, make_projector
from src.hmm.decoder import MapMatchHMM, HMMConfig, gaussian_log_emission
from src.hmm.candidates import ArcIndex
from src.hmm.metrics import route_continuity


def build_grid(n=6, step_m=100.0):
    """n x n bidirectional grid. Nodes placed with metre offsets around a lon/lat origin."""
    lon0, lat0 = 35.5, 49.5
    proj = make_projector(lon0, lat0)
    x0, y0 = proj.transform(lon0, lat0)
    inv = Transformer.from_crs(proj.target_crs, "EPSG:4326", always_xy=True)
    xs, ys = np.meshgrid(np.arange(n) * step_m + x0, np.arange(n) * step_m + y0)
    lon, lat = inv.transform(xs.ravel(), ys.ravel())
    nodes = np.column_stack([lon, lat])
    arcs = []
    nid = lambda r, c: r * n + c
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                arcs += [(nid(r, c), nid(r, c + 1)), (nid(r, c + 1), nid(r, c))]
            if r + 1 < n:
                arcs += [(nid(r, c), nid(r + 1, c)), (nid(r + 1, c), nid(r, c))]
    return RoadGraph(nodes, np.array(arcs), projector=proj)


def test_decoder_recovers_straight_drive_despite_noise():
    g = build_grid()
    rng = np.random.default_rng(1)
    # drive east along row 2 at 10 m/s, 1 Hz, with 12 m GPS noise
    xs = g.nodes_xy[2 * 6 + 0, 0] + 30.0 + np.arange(0, 420, 10.0)  # start mid-block, not on a junction
    ys = np.full_like(xs, g.nodes_xy[2 * 6, 1])
    truth_xy = np.column_stack([xs, ys])
    obs = truth_xy + rng.normal(scale=12.0, size=truth_xy.shape)
    t = np.arange(len(obs), dtype=float)
    idx = ArcIndex(g)
    arcs, dist = idx.candidates(obs, radius_m=60, max_k=8)
    em = [gaussian_log_emission(d, 15.0) for d in dist]
    dec = MapMatchHMM(g, HMMConfig())
    res = dec.decode(obs, t, arcs, em)
    pred = res.arc_ids
    assert (pred >= 0).all()
    # every predicted arc must lie on row 2 and run eastwards
    assert np.allclose(g.nodes_xy[g.arc_from[pred], 1], g.nodes_xy[2 * 6, 1])
    assert (g.arc_vec[pred, 0] > 0).all()
    assert route_continuity(pred, g) == 1.0


def test_gap_triggers_break_and_empty_steps_are_minus_one():
    g = build_grid()
    xy = g.nodes_xy[[12, 13, 14]] + 3.0
    xy = np.vstack([xy, [[1e6, 1e6]], g.nodes_xy[[15]] + 3.0])   # 4th point: far from everything
    t = np.array([0, 1, 2, 3, 40.0])                            # big gap before last
    idx = ArcIndex(g)
    arcs, dist = idx.candidates(xy, radius_m=40, max_k=5)
    em = [gaussian_log_emission(d, 15.0) for d in dist]
    res = MapMatchHMM(g).decode(xy, t, arcs, em)
    assert res.arc_ids[3] == -1 and res.n_empty_steps == 1
    assert res.n_breaks == 1
    assert res.segment_id[4] == res.segment_id[0] + 1


def test_expand_route_fills_gap_with_shortest_path():
    g = build_grid()
    dec = MapMatchHMM(g)
    # find arc 12->13 and arc 14->15 (skipping 13->14)
    def arc(u, v):
        return int(np.where((g.arc_from == u) & (g.arc_to == v))[0][0])
    a, b = arc(12, 13), arc(14, 15)
    route = dec.expand_route(np.array([a, b]), np.array([0, 0]))
    assert route == [a, arc(13, 14), b]
    # but no fill across a chain break
    route = dec.expand_route(np.array([a, b]), np.array([0, 1]))
    assert route == [a, b]
