"""ClassifierScorer contract tests. The model here is a throw-away fixture trained on
synthetic numbers inside the test; it is NOT a result and no metric is taken from it."""
import sys
import types

import numpy as np
import pytest

from src.hmm.scoring import ClassifierScorer, build_scorer


def _fake_feature_module(monkeypatch, with_fn=True):
    mod = types.ModuleType("fake_features")
    if with_fn:
        # feature = [-distance]  (closer -> larger feature)
        mod.build_candidate_features = lambda trip, arcs, dist: [np.asarray(d).reshape(-1, 1) * -1.0 for d in dist]
    monkeypatch.setitem(sys.modules, "fake_features", mod)


def _dump_model(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 50, 400)
    y = (d < 15).astype(int)
    joblib.dump(LogisticRegression().fit((-d).reshape(-1, 1), y), tmp_path / "model.joblib")


def test_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClassifierScorer(tmp_path)


def test_missing_feature_builder_fails_loudly(tmp_path, monkeypatch):
    _dump_model(tmp_path)
    _fake_feature_module(monkeypatch, with_fn=False)
    with pytest.raises(RuntimeError, match="build_candidate_features"):
        ClassifierScorer(tmp_path, feature_module="fake_features")


def test_build_scorer_classifier_does_not_fall_back(tmp_path):
    with pytest.raises(Exception):
        build_scorer("classifier", tmp_path)          # no artifacts -> must raise, not return a scorer


def test_scores_are_log_probs_aligned_with_candidates(tmp_path, monkeypatch):
    _dump_model(tmp_path)
    _fake_feature_module(monkeypatch)
    sc = ClassifierScorer(tmp_path, feature_module="fake_features")
    arcs = [np.array([1, 2, 3]), np.empty(0, dtype=int), np.array([4])]
    dist = [np.array([2.0, 20.0, 45.0]), np.empty(0), np.array([5.0])]
    out = sc.score(None, arcs, dist)
    assert [len(o) for o in out] == [3, 0, 1]
    assert all((o <= 0).all() for o in out)           # log of probabilities
    assert out[0][0] > out[0][1] > out[0][2]          # nearer candidate -> higher score
