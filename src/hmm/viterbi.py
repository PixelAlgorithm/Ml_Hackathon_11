"""Generic log-space Viterbi with chain breaks.

Kept independent of road networks so it can be unit-tested on toy problems.

Recurrence (per step t, state j):
    delta_t(j) = max_i [ delta_{t-1}(i) + log A_t(i, j) ] + log B_t(j)
    psi_t(j)   = argmax_i [ ... ]
Backtracking starts from argmax_j delta_T(j).

Chain breaks: the chain restarts at step t (delta_t = log B_t, no backpointer) when
  * break_before[t] is True (e.g. a large GPS gap), or
  * every transition into step t is impossible (all -inf), i.e. the candidate
    segments are disconnected from everything the previous step could reach.
Each segment is decoded independently by backtracking from its own best final state.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

NEG_INF = -np.inf


def viterbi(
    log_emission: Sequence[np.ndarray],
    log_transition: Callable[[int], np.ndarray],
    break_before: Optional[Sequence[bool]] = None,
) -> Tuple[List[int], List[int]]:
    """Decode the most likely state index per step.

    Parameters
    ----------
    log_emission : list of arrays, shape (K_t,). K_t >= 1 for every step.
    log_transition : f(t) -> (K_{t-1}, K_t) log transition matrix for t >= 1.
    break_before : optional bools, break_before[t] forces a restart at step t.

    Returns
    -------
    path : chosen state index per step
    segment : segment id per step (increments at every chain break)
    """
    T = len(log_emission)
    if T == 0:
        return [], []
    forced = list(break_before) if break_before is not None else [False] * T

    delta = np.asarray(log_emission[0], dtype=float)
    backptr: List[Optional[np.ndarray]] = [None]
    end_delta: dict[int, np.ndarray] = {}   # delta of the last step of each closed segment
    segment = [0]

    for t in range(1, T):
        emit = np.asarray(log_emission[t], dtype=float)
        restart = bool(forced[t])
        if not restart:
            scores = delta[:, None] + np.asarray(log_transition(t), dtype=float)
            best = scores.argmax(axis=0)
            top = scores[best, np.arange(scores.shape[1])]
            if np.all(np.isneginf(top)):
                restart = True
        if restart:
            end_delta[t - 1] = delta
            delta = emit.copy()
            backptr.append(None)
            segment.append(segment[-1] + 1)
        else:
            delta = top + emit
            backptr.append(best)
            segment.append(segment[-1])

    path = [0] * T
    idx = int(np.argmax(delta))
    for t in range(T - 1, -1, -1):
        path[t] = idx
        if backptr[t] is None:
            if t > 0:
                idx = int(np.argmax(end_delta[t - 1]))
        else:
            idx = int(backptr[t][idx])
    return path, segment
