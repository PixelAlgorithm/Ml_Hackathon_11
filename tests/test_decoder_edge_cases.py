"""Edge cases for the decoder (task 1): one test per failure mode."""
import numpy as np
import pytest

from src.hmm.decoder import HMMConfig, MapMatchHMM, gaussian_log_emission
from src.hmm.graph import RoadGraph
from src.hmm.candidates import ArcIndex
from tests.test_decoder_toy import build_grid


def _arc(g, u, v):
    return int(np.where((g.arc_from == u) & (g.arc_to == v))[0][0])


def _decode(g, xy, t, radius=40, cfg=None):
    arcs, dist = ArcIndex(g).candidates(xy, radius_m=radius, max_k=8)
    em = [gaussian_log_emission(d, 15.0) for d in dist]
    return MapMatchHMM(g, cfg or HMMConfig()).decode(xy, t, arcs, em), arcs, em


def test_all_points_without_candidates():
    g = build_grid()
    xy = np.array([[1e6, 1e6], [1e6 + 5, 1e6]])
    res, _, _ = _decode(g, xy, np.array([0.0, 1.0]))
    assert (res.arc_ids == -1).all() and (res.segment_id == -1).all()
    assert res.n_empty_steps == 2 and res.n_breaks == 0


def test_empty_step_in_the_middle_is_bridged_without_break():
    g = build_grid()
    xy = np.vstack([g.nodes_xy[12] + [20, 2], [[1e6, 1e6]], g.nodes_xy[12] + [60, 2]])
    res, _, _ = _decode(g, xy, np.array([0.0, 1.0, 2.0]))
    assert res.arc_ids[1] == -1 and res.n_breaks == 0
    assert res.segment_id[0] == res.segment_id[2]


def test_single_point_trip():
    g = build_grid()
    res, _, _ = _decode(g, g.nodes_xy[[12]] + [20, 2], np.array([0.0]))
    assert res.arc_ids[0] >= 0 and res.n_breaks == 0


def test_gps_gap_breaks_chain_but_close_fixes_do_not():
    g = build_grid()
    xy = g.nodes_xy[[12, 13, 14]] + 3.0
    r_ok, _, _ = _decode(g, xy, np.array([0.0, 1.0, 2.0]))
    r_gap, _, _ = _decode(g, xy, np.array([0.0, 1.0, 100.0]))
    assert r_ok.n_breaks == 0 and r_gap.n_breaks == 1


def test_disconnected_components_break_instead_of_bridging():
    """Two separate islands: no path exists between them -> chain must restart."""
    a = build_grid(n=3)
    nodes = np.vstack([a.nodes_lonlat, a.nodes_lonlat + [0.05, 0.0]])   # island B ~3 km east
    arcs = np.vstack([a.arcs, a.arcs + len(a.nodes_lonlat)])
    g = RoadGraph(nodes, arcs, projector=a.projector)
    xy = np.vstack([a.nodes_xy[[3]] + [20, 1], g.nodes_xy[[9 + 3]] + [20, 1]])
    res, _, _ = _decode(g, xy, np.array([0.0, 1.0]))
    assert (res.arc_ids >= 0).all() and res.n_breaks == 1


def test_uturn_penalty_discourages_reverse_arc():
    """Fixes straddle junction 13, so hopping fwd -> reverse costs the same network
    distance as staying on fwd; only the U-turn penalty separates the two options."""
    g = build_grid()
    fwd, back = _arc(g, 12, 13), _arc(g, 13, 12)
    xy = np.array([g.nodes_xy[13] + [-2, 0], g.nodes_xy[13] + [2, 0]])
    arcs = [np.array([fwd, back])] * 2
    em = [np.array([0.0, -5.0]), np.array([-1.0, 0.0])]   # step 1 emission prefers `back`
    t = np.array([0.0, 1.0])
    no_pen = MapMatchHMM(g, HMMConfig(uturn_penalty=0.0)).decode(xy, t, arcs, em).arc_ids
    pen = MapMatchHMM(g, HMMConfig(uturn_penalty=4.0)).decode(xy, t, arcs, em).arc_ids
    assert no_pen[1] == back      # without the penalty the reverse arc wins
    assert pen[1] == fwd          # with it the heading is kept


def test_backward_jitter_on_same_arc_is_not_a_jump():
    g = build_grid()
    a = _arc(g, 12, 13)
    xy = np.array([g.nodes_xy[12] + [50, 1], g.nodes_xy[12] + [40, 1], g.nodes_xy[12] + [55, 1]])
    arcs = [np.array([a])] * 3
    em = [np.zeros(1)] * 3
    res = MapMatchHMM(g).decode(xy, np.arange(3.0), arcs, em)
    assert (res.arc_ids == a).all() and res.n_breaks == 0


def test_nan_and_neginf_emissions_do_not_poison_the_step():
    g = build_grid()
    a, b = _arc(g, 12, 13), _arc(g, 13, 14)
    xy = np.array([g.nodes_xy[12] + [50, 1], g.nodes_xy[13] + [50, 1]])
    arcs = [np.array([a]), np.array([b, a])]
    em = [np.array([0.0]), np.array([np.nan, -np.inf])]
    res = MapMatchHMM(g).decode(xy, np.array([0.0, 1.0]), arcs, em)
    assert (res.arc_ids >= 0).all()


def test_length_mismatch_raises():
    g = build_grid()
    xy = g.nodes_xy[[12]] + 3.0
    with pytest.raises(ValueError):
        MapMatchHMM(g).decode(xy, np.array([0.0]), [np.array([0, 1])], [np.zeros(1)])


def test_route_cutoff_is_respected_even_with_warm_cache():
    """Regression: a Dijkstra cached with a large cutoff must not leak into a small one."""
    g = build_grid()
    far = g.node_distances(12, 1000.0)
    near = g.node_distances(12, 150.0)
    assert max(near.values()) <= 150.0 and len(near) < len(far)
    assert g.path_arcs(12, 17, 150.0) is None and g.path_arcs(12, 17, 1000.0) is not None
