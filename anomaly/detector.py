"""
Anomaly detector: converts raw scores into binary anomaly flags.
Supports threshold-based detection with configurable sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class ThresholdDetector:
    """
    Converts a continuous anomaly score series into binary flags
    using a z-score threshold.

    Args:
        z_threshold: number of std devs above mean to flag as anomaly (default 2.0)
        window:      rolling window for computing mean/std (None = use full series)
    """

    def __init__(self, z_threshold: float = 2.0, window: int | None = None):
        self.z_threshold = z_threshold
        self.window      = window

    def flag(self, scores: pd.Series) -> pd.Series:
        """
        Returns binary series: 1 = anomaly, 0 = normal.
        """
        if self.window is not None:
            mean = scores.rolling(self.window, min_periods=1).mean()
            std  = scores.rolling(self.window, min_periods=1).std().fillna(1e-8)
        else:
            mean = scores.mean()
            std  = scores.std() or 1e-8

        z_scores = (scores - mean) / std
        return (z_scores > self.z_threshold).astype(int).rename("anomaly_flag")

    def z_scores(self, scores: pd.Series) -> pd.Series:
        """Return normalized z-scores instead of binary flags."""
        mean = scores.mean()
        std  = scores.std() or 1e-8
        return ((scores - mean) / std).rename("z_score")


def score_event_window(
    scores:       pd.Series,
    window_start: "date",
    window_end:   "date",
    baseline:     pd.Series | None = None,
) -> dict:
    """
    Evaluate anomaly scores within an event window.

    Returns dict with:
        max_score:      peak score in window
        mean_score:     mean score in window
        z_score:        how many stds above full-series mean
        detected:       bool — did max score exceed 2-sigma threshold
        lead_days:      days before peak_date where score first exceeds threshold
                        (requires baseline for comparison)
    """
    from datetime import date
    window_scores = scores[
        (scores.index >= pd.Timestamp(window_start)) &
        (scores.index <= pd.Timestamp(window_end))
    ]

    if len(window_scores) == 0:
        return {"max_score": np.nan, "mean_score": np.nan,
                "z_score": np.nan, "detected": False}

    global_mean = scores.mean()
    global_std  = scores.std() or 1e-8
    max_score   = window_scores.max()
    z           = (max_score - global_mean) / global_std

    return {
        "max_score":  float(max_score),
        "mean_score": float(window_scores.mean()),
        "z_score":    float(z),
        "detected":   bool(z > 2.0),
    }
