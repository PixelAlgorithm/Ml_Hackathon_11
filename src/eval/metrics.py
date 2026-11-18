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