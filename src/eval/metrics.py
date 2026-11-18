"""Evaluation metrics for PathSnap (candidate-level scores)."""
import numpy as np
import pandas as pd


def top1_accuracy(df, score_col="score"):
    """For each GPS point (trip_id, obs_idx), pick the candidate with the
    highest score. Return the fraction of GPS points where that pick has label 1."""
    best = df.loc[df.groupby(["trip_id", "obs_idx"])[score_col].idxmax()]
    return float(best["label"].mean())