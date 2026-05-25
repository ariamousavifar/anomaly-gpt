"""Smoke tests for eval harness."""
import numpy as np
import pandas as pd
import pytest
from anomaly.events import EVENTS, get_event
from eval.harness import evaluate_detector, detection_counts


def make_scores(n=3500):
    np.random.seed(0)
    dates  = pd.date_range("2010-01-04", periods=n, freq="B")
    scores = pd.Series(np.random.exponential(1.5, n), index=dates, name="test")
    # Inject spikes at known event dates
    for event in EVENTS:
        peak = pd.Timestamp(event.peak_date)
        if peak in scores.index:
            scores[peak] = scores.mean() + 4 * scores.std()
    return scores


def test_evaluate_detector_shape():
    scores = make_scores()
    result = evaluate_detector("test", scores)
    assert len(result) == len(EVENTS)
    assert "z_score" in result.columns
    assert "detected" in result.columns


def test_evaluate_detector_columns():
    scores = make_scores()
    result = evaluate_detector("test", scores)
    for col in ["detector", "event", "peak_date", "max_score", "detected"]:
        assert col in result.columns


def test_detection_counts():
    scores = make_scores()
    result = evaluate_detector("test", scores)
    counts = detection_counts(result)
    assert "detected" in counts.columns
    assert "detection_rate" in counts.columns


def test_get_event():
    event = get_event("SVB Collapse 2023")
    assert event.peak_date.year == 2023


def test_event_count():
    assert len(EVENTS) == 7
