"""Evaluation metrics for PathSnap (candidate-level scores)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def top1_accuracy(df, score_col="score"):
    """For each GPS point (trip_id, obs_idx), pick the candidate with the
    highest score. Return the fraction of GPS points where that pick has label 1."""
    best = df.loc[df.groupby(["trip_id", "obs_idx"])[score_col].idxmax()]
    return float(best["label"].mean())


def macro_f1(df, score_col="score", threshold=0.5):
    """Macro F1 over the two classes (right road = 1, wrong road = 0).
    A candidate is predicted 1 when its score is >= threshold."""
    pred = (df[score_col] >= threshold).astype(int)
    return float(f1_score(df["label"], pred, average="macro"))


def candidate_log_loss(df, score_col="score"):
    """Log loss of the candidate scores (scores must be probabilities 0 to 1)."""
    return float(log_loss(df["label"], df[score_col], labels=[0, 1]))


def candidate_miss_rate(candidates, truth):
    """Fraction of GPS points whose reference road is NOT in the candidate set.

    candidates: DataFrame with columns trip_id, obs_idx, arc_id
                (one row per GPS point and candidate road)
    truth:      DataFrame with columns trip_id, obs_idx, true_arc_id
                (one row per GPS point)
    """
    keys = ["trip_id", "obs_idx"]
    merged = truth.merge(
        candidates.rename(columns={"arc_id": "true_arc_id"})
        .assign(hit=1)[keys + ["true_arc_id", "hit"]],
        on=keys + ["true_arc_id"],
        how="left",
    )
    hit = merged.groupby(keys)["hit"].max().fillna(0)
    return float(1.0 - hit.mean())