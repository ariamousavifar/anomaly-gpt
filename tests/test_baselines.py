"""Tests for baseline detectors."""
import numpy as np
import pandas as pd
import pytest
from anomaly.baselines import RollingVolatility, EWMAVolatility, IsolationForestDetector, get_all_baselines


def make_returns(n=300):
    np.random.seed(42)
    dates   = pd.date_range("2015-01-01", periods=n, freq="B")
    returns = pd.Series(np.random.normal(0, 0.01, n), index=dates)
    return returns


def test_rolling_vol_length():
    returns = make_returns()
    det     = RollingVolatility(window=20)
    scores  = det.score(returns)
    assert len(scores) == len(returns)


def test_rolling_vol_positive():
    returns = make_returns()
    det     = RollingVolatility(window=20)
    scores  = det.score(returns).dropna()
    assert (scores >= 0).all()


def test_ewma_vol_length():
    returns = make_returns()
    det     = EWMAVolatility(span=20)
    scores  = det.score(returns)
    assert len(scores) == len(returns)


def test_isolation_forest_fit_score():
    returns = make_returns(300)
    det     = IsolationForestDetector(n_lags=20)
    det.fit(returns)
    scores  = det.score(returns)
    assert len(scores) == len(returns) - 20


def test_isolation_forest_requires_fit():
    returns = make_returns()
    det     = IsolationForestDetector()
    with pytest.raises(AssertionError):
        det.score(returns)


def test_get_all_baselines():
    baselines = get_all_baselines()
    assert len(baselines) == 3
