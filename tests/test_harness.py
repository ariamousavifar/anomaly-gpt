"""Smoke tests for eval harness."""
import numpy as np
import pandas as pd
from anomaly.events import EVENTS, get_event
from eval.harness import evaluate_detector, detection_counts


def make_scores(n=3500, spike_events=None):
    """
    Flat noise with 4-sigma spikes planted at event peak dates.

    spike_events: event names to spike (default: all). Events omitted here
    must NOT be detected — that is what makes this a ground-truth fixture
    rather than a smoke test.
    """
    np.random.seed(0)
    dates = pd.date_range("2010-01-04", periods=n, freq="B")
    # Bounded noise: max achievable z is ~1.7, so no unplanted day can ever
    # cross z=2. Heavy-tailed noise would produce chance detections inside
    # multi-day windows and make the ground truth unrecoverable.
    scores = pd.Series(np.random.uniform(0.0, 1.0, n), index=dates, name="test")
    spike_level = scores.mean() + 40 * scores.std()

    planted = []
    for event in EVENTS:
        if spike_events is not None and event.name not in spike_events:
            continue
        peak = pd.Timestamp(event.peak_date)
        if peak in scores.index:
            scores[peak] = spike_level
            planted.append(event.name)
    return scores, planted


def test_planted_spikes_are_detected():
    """The fixture's ground truth must actually be recovered."""
    scores, planted = make_scores()
    assert planted, "fixture planted no spikes — test would be vacuous"

    result = evaluate_detector("test", scores).set_index("event")
    for name in planted:
        assert bool(result.loc[name, "detected"]), f"missed planted spike: {name}"


def test_unspiked_events_are_not_detected():
    """No spike must mean no detection — guards against a detector that always fires."""
    keep = "COVID Crash 2020"
    scores, planted = make_scores(spike_events=[keep])
    assert planted == [keep]

    result = evaluate_detector("test", scores).set_index("event")
    assert bool(result.loc[keep, "detected"])
    for name in [e.name for e in EVENTS if e.name != keep]:
        assert not bool(result.loc[name, "detected"]), f"false positive on {name}"


def test_detection_counts_matches_planted_total():
    """The published k/7 number must equal the number of true events."""
    keep = ["COVID Crash 2020", "SVB Collapse 2023"]
    scores, planted = make_scores(spike_events=keep)
    result = evaluate_detector("test", scores)
    counts = detection_counts(result)
    assert int(counts.loc["test", "detected"]) == len(planted)
    assert int(counts.loc["test", "total"]) == len(EVENTS)


def test_window_criterion_false_positives_under_heavy_tails():
    """
    Documents a limitation of the z>2 window criterion: with i.i.d.
    heavy-tailed scores, a multi-day window collects chance exceedances, so
    an event can be 'detected' from pure noise.

    Real detector scores are smoothed and therefore autocorrelated, which
    makes chance detection much rarer than this i.i.d. case; the honest null
    for a published claim is the circular-shift test in scripts/run_eval.py,
    not this one. This test only pins the criterion's behaviour.
    """
    np.random.seed(0)
    dates  = pd.date_range("2010-01-04", periods=3500, freq="B")
    noise  = pd.Series(np.random.exponential(1.5, 3500), index=dates, name="noise")

    result = evaluate_detector("noise", noise)
    assert int(result["detected"].sum()) > 0, (
        "expected chance detections from pure noise under the z>2 window rule"
    )


def test_evaluate_detector_shape():
    scores, _ = make_scores()
    result = evaluate_detector("test", scores)
    assert len(result) == len(EVENTS)
    assert "z_score" in result.columns
    assert "detected" in result.columns


def test_evaluate_detector_columns():
    scores, _ = make_scores()
    result = evaluate_detector("test", scores)
    for col in ["detector", "event", "peak_date", "max_score", "detected"]:
        assert col in result.columns


def test_detection_counts():
    scores, _ = make_scores()
    result = evaluate_detector("test", scores)
    counts = detection_counts(result)
    assert "detected" in counts.columns
    assert "detection_rate" in counts.columns


def test_get_event():
    event = get_event("SVB Collapse 2023")
    assert event.peak_date.year == 2023


def test_event_count():
    assert len(EVENTS) == 7
