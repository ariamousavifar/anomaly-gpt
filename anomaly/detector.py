"""
Turns a continuous anomaly score series into event-level detections.

Two thresholding rules are provided:
  - score_event_window:         fixed z > 2 on the full-series mean/std
  - score_event_window_at_rate: threshold set to flag a fixed fraction of days

The second is the comparable one. A fixed z is not a common operating point
across detectors, because score distributions differ in skew — see flag_rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def flag_rate(scores: pd.Series, z_threshold: float = 2.0) -> float:
    """
    Fraction of all days the detector flags at a given z-threshold.

    A fixed z-threshold is NOT a common operating point across detectors:
    volatility scores are strongly right-skewed while perplexity scores are
    not, so the same z corresponds to very different false-positive rates.
    """
    z = (scores - scores.mean()) / (scores.std() or 1e-8)
    return float((z > z_threshold).sum()) / len(scores)


def score_event_window_at_rate(
    scores:       pd.Series,
    window_start: "date",
    window_end:   "date",
    rate:         float,
) -> dict:
    """
    Detection with the threshold set so the detector flags exactly `rate`
    of all days. This equalises false-positive rate across detectors and is
    the comparable counterpart to score_event_window's fixed z-threshold.

    Returns dict with threshold, max_score and detected.
    """
    threshold     = scores.quantile(1.0 - rate)
    window_scores = scores[
        (scores.index >= pd.Timestamp(window_start)) &
        (scores.index <= pd.Timestamp(window_end))
    ]

    if len(window_scores) == 0:
        return {"threshold": float(threshold), "max_score": np.nan, "detected": False}

    max_score = window_scores.max()
    return {
        "threshold": float(threshold),
        "max_score": float(max_score),
        "detected":  bool(max_score > threshold),
    }
