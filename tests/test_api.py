import os

import pytest
from fastapi.testclient import TestClient

from tests.conftest import DATA_ROOT

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="dataset not available")


@pytest.fixture(scope="module")
def client():
    os.environ.update(PATHSNAP_DATA_DIR=str(DATA_ROOT), PATHSNAP_PARTITION="student_test",
                      PATHSNAP_PRELOAD="0", PATHSNAP_SCORER="distance")
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_trips_and_health(client):
    assert client.get("/health").json()["status"] == "ok"
    ids = [t["trip_id"] for t in client.get("/trips").json()]
    assert len(ids) == 10


def test_predict_hides_labels_and_scores_route(client):
    r = client.post("/predict", json={"trip_id": "00000052"})
    assert r.status_code == 200
    body = r.json()
    assert not any("reference" in k or "truth" in k or "label" in k for k in body)
    assert body["metrics_hmm"]["route_f1"] > body["metrics_classifier_only"]["route_f1"]
    assert body["metrics_hmm"]["route_continuity"] >= 0.9
    assert len(body["per_point_arc"]) == body["n_points"]


def test_unknown_trip_404(client):
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404


def test_labels_are_never_loaded_into_the_prediction_path(client):
    import api.main as m
    trip, _ = m.pipeline._load("00000052")
    assert trip.route is None                          # reference route not on the Trip object
    body = client.post("/predict", json={"trip_id": "00000052"}).json()
    assert not any(w in k for k in body for w in ("reference", "truth", "label"))


def test_classifier_scorer_without_artifacts_refuses_to_start(tmp_path):
    from fastapi.testclient import TestClient
    import importlib
    os.environ.update(PATHSNAP_SCORER="classifier", PATHSNAP_ARTIFACTS=str(tmp_path))
    try:
        import api.main as m
        importlib.reload(m)
        with pytest.raises(Exception):
            with TestClient(m.app):
                pass
    finally:
        os.environ.update(PATHSNAP_SCORER="distance")
        os.environ.pop("PATHSNAP_ARTIFACTS", None)
        importlib.reload(m)
