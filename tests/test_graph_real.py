import numpy as np
import pytest
from tests.conftest import DATA_ROOT
from src.hmm.graph import load_trip, list_trip_ids

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="dataset not available")


def test_trip_loads_and_is_consistent():
    tid = "00000005"
    trip = load_trip(DATA_ROOT, tid)
    g = trip.graph
    assert trip.track_xy.shape == (len(trip.track_lonlat), 2)
    assert g.arc_len.min() >= 0
    assert trip.route.max() < g.n_arcs
    # projected GPS and projected network share one frame: track near the network
    gx, gy = trip.track_xy.min(0), trip.track_xy.max(0)
    nx_, ny_ = g.nodes_xy.min(0), g.nodes_xy.max(0)
    assert gx[0] > nx_[0] - 2000 and gy[1] < ny_[1] + 2000


def test_reference_route_is_mostly_connected_and_shortest_path_works():
    trip = load_trip(DATA_ROOT, "00000005")
    g, r = trip.graph, trip.route
    conn = (g.arc_to[r[:-1]] == g.arc_from[r[1:]]).mean()
    assert conn > 0.95
    p = g.path_arcs(int(g.arc_from[r[0]]), int(g.arc_to[r[3]]), 5000)
    assert p is not None and len(p) >= 3


def test_list_trip_ids_counts():
    assert len(list_trip_ids(DATA_ROOT, "train")) == 80
    assert len(list_trip_ids(DATA_ROOT, "student_test")) == 10
