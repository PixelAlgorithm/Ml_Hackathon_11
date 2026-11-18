import numpy as np
from src.hmm.viterbi import viterbi, NEG_INF


def brute(em, trans):
    import itertools
    best, bp = -np.inf, None
    for p in itertools.product(*[range(len(e)) for e in em]):
        s = em[0][p[0]] + sum(trans(t)[p[t - 1], p[t]] + em[t][p[t]] for t in range(1, len(em)))
        if s > best:
            best, bp = s, p
    return list(bp)


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    em = [rng.normal(size=k) for k in (3, 2, 4, 3, 2)]
    Ms = {t: rng.normal(size=(len(em[t - 1]), len(em[t]))) for t in range(1, 5)}
    path, seg = viterbi(em, lambda t: Ms[t])
    assert path == brute(em, lambda t: Ms[t])
    assert set(seg) == {0}


def test_transition_fixes_emission_noise():
    # state 0 is "on route"; emission prefers 1 at step 2 but transitions forbid 0->1
    em = [np.array([0.0, -5.0])] * 3
    em[1] = np.array([-1.0, 0.0])
    M = np.array([[0.0, NEG_INF], [NEG_INF, 0.0]])
    path, _ = viterbi(em, lambda t: M)
    assert path == [0, 0, 0]


def test_break_on_disconnected_and_forced_gap():
    em = [np.zeros(2)] * 4
    M_ok = np.zeros((2, 2))
    M_dead = np.full((2, 2), NEG_INF)
    mats = {1: M_ok, 2: M_dead, 3: M_ok}
    path, seg = viterbi(em, lambda t: mats[t])
    assert seg == [0, 0, 1, 1]
    path, seg = viterbi(em, lambda t: M_ok, break_before=[False, False, True, False])
    assert seg == [0, 0, 1, 1]


def test_single_step_and_empty():
    assert viterbi([], lambda t: None) == ([], [])
    p, s = viterbi([np.array([0.0, 1.0])], lambda t: None)
    assert p == [1] and s == [0]
