"""Tests for threshold logic — this code decides every published detection."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from anomaly.detector import (
    flag_rate,
    score_event_window,
    score_event_window_at_rate,
)


def _series(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(rng.exponential(1.0, n), index=idx)


def test_flag_rate_matches_actual_fraction():
    s = _series()
    z = (s - s.mean()) / s.std()
    assert flag_rate(s, 2.0) == pytest.approx((z > 2.0).mean())


def test_flag_rate_differs_across_distribution_shapes():
    """
    The reason a fixed z is not a common operating point: two detectors at
    the same z admit materially different false-positive rates. The direction
    depends on the distributions, so only the magnitude of the gap is asserted.
    """
    idx = pd.date_range("2010-01-01", periods=5000, freq="B")
    rng = np.random.default_rng(1)
    skewed = pd.Series(rng.exponential(1.0, 5000), index=idx)   # right-skewed
    normal = pd.Series(rng.normal(0, 1, 5000), index=idx)       # symmetric
    gap = abs(flag_rate(skewed, 2.0) - flag_rate(normal, 2.0))
    assert gap > 0.01, (
        f"expected materially different FPR at z=2, got gap={gap:.4f}; "
        "if these matched, a fixed z would be a valid common operating point"
    )


def test_matched_rate_flags_requested_fraction():
    """Threshold must admit ~`rate` of all days, whatever the distribution."""
    for rate in (0.01, 0.05, 0.10):
        for seed in (0, 1):
            s   = _series(2000, seed)
            thr = s.quantile(1.0 - rate)
            assert (s > thr).mean() == pytest.approx(rate, abs=0.005)


def test_matched_rate_detects_planted_spike():
    """A spike inside the window must be detected; a flat window must not be."""
    s = _series(1000)
    peak = s.index[500]
    s.loc[peak] = s.max() * 10

    hit = score_event_window_at_rate(s, peak - pd.Timedelta(days=2),
                                     peak + pd.Timedelta(days=2), rate=0.01)
    assert hit["detected"] is True
    assert hit["max_score"] > hit["threshold"]

    quiet_start = s.index[100]
    calm = s.copy()
    calm.loc[quiet_start:s.index[110]] = calm.min()
    miss = score_event_window_at_rate(calm, quiet_start, s.index[110], rate=0.01)
    assert miss["detected"] is False


def test_lower_rate_is_stricter():
    """Monotonicity: a stricter rate can never detect more than a looser one."""
    s = _series(1500)
    w0, w1 = s.index[300], s.index[320]
    strict = score_event_window_at_rate(s, w0, w1, rate=0.01)["detected"]
    loose  = score_event_window_at_rate(s, w0, w1, rate=0.20)["detected"]
    assert not (strict and not loose), "stricter threshold detected what looser did not"


def test_empty_window_is_not_a_detection():
    s = _series()
    r = score_event_window_at_rate(s, date(1990, 1, 1), date(1990, 1, 5), rate=0.05)
    assert r["detected"] is False
    assert np.isnan(r["max_score"])
    z = score_event_window(s, date(1990, 1, 1), date(1990, 1, 5))
    assert z["detected"] is False
