"""
Evaluation harness.
Runs all detectors (GPT + baselines) on all 7 known events.
Produces a structured results table for FINDINGS.md.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from anomaly.events import EVENTS, MarketEvent
from anomaly.detector import score_event_window


def evaluate_detector(
    name:        str,
    scores:      pd.Series,
    events:      list[MarketEvent] | None = None,
) -> pd.DataFrame:
    """
    Evaluate one detector's scores against all events.

    Args:
        name:   detector name (for display)
        scores: anomaly score series with DatetimeIndex
        events: list of MarketEvent (default: all 7)

    Returns:
        DataFrame with one row per event.
    """
    events = events or EVENTS
    rows   = []
    for event in events:
        result = score_event_window(scores, event.window_start, event.window_end)
        rows.append({
            "detector":   name,
            "event":      event.name,
            "peak_date":  event.peak_date,
            "max_score":  result["max_score"],
            "mean_score": result["mean_score"],
            "z_score":    result["z_score"],
            "detected":   result["detected"],
        })
    return pd.DataFrame(rows)


def run_full_harness(
    gpt_scores:       pd.Series,
    baseline_scores:  dict[str, pd.Series],
    events:           list[MarketEvent] | None = None,
) -> pd.DataFrame:
    """
    Run all detectors on all events.

    Args:
        gpt_scores:      GPT perplexity scores (primary detector)
        baseline_scores: dict of {detector_name: score_series}
        events:          list of MarketEvent

    Returns:
        Combined DataFrame with results for all detectors x all events.
    """
    events  = events or EVENTS
    results = [evaluate_detector("GPT Perplexity", gpt_scores, events)]

    for name, scores in baseline_scores.items():
        results.append(evaluate_detector(name, scores, events))

    df = pd.concat(results, ignore_index=True)
    return df


def detection_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot table: detectors as rows, events as columns.
    Cell = z_score (bold if detected).
    """
    pivot = results.pivot_table(
        index="detector",
        columns="event",
        values="z_score",
        aggfunc="first",
    ).round(2)
    return pivot


def detection_counts(results: pd.DataFrame) -> pd.DataFrame:
    """Count how many events each detector fires on."""
    return (
        results.groupby("detector")["detected"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "detected", "count": "total"})
        .assign(detection_rate=lambda x: x["detected"] / x["total"])
        .sort_values("detected", ascending=False)
    )
