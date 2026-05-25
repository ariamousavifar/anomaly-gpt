"""
Baseline anomaly detectors for comparison against GPT perplexity.

All detectors implement a common interface:
    fit(returns)  -> self
    score(returns) -> pd.Series

Baselines:
    1. RollingVolatility  — 20-day rolling std of returns
    2. EWMAVolatility     — Exponentially weighted moving average std
    3. IsolationForest    — sklearn anomaly detector on lagged features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class RollingVolatility:
    """
    Rolling standard deviation of log-returns.
    Classic volatility proxy — simplest possible baseline.

    Args:
        window: lookback window in trading days (default 20 = ~1 month)
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name   = f"RollingVol(w={window})"

    def fit(self, returns: pd.Series) -> "RollingVolatility":
        return self  # stateless

    def score(self, returns: pd.Series) -> pd.Series:
        """Returns rolling std, normalized to [0,1] range for fair comparison."""
        raw = returns.rolling(window=self.window, min_periods=1).std()
        return raw.rename("rolling_vol")


class EWMAVolatility:
    """
    Exponentially weighted moving average volatility.
    More responsive to recent moves than rolling std.

    Args:
        span: EWMA span parameter (default 20)
    """

    def __init__(self, span: int = 20):
        self.span = span
        self.name = f"EWMA(span={span})"

    def fit(self, returns: pd.Series) -> "EWMAVolatility":
        return self  # stateless

    def score(self, returns: pd.Series) -> pd.Series:
        raw = returns.ewm(span=self.span, adjust=False).std()
        return raw.rename("ewma_vol")


class IsolationForestDetector:
    """
    Isolation Forest on lagged return features.
    ML baseline — no sequence modeling, uses fixed-size feature window.

    Args:
        n_lags:           number of lagged returns as features
        contamination:    expected fraction of anomalies (default 0.05)
        random_state:     for reproducibility
    """

    def __init__(
        self,
        n_lags:        int   = 20,
        contamination: float = 0.05,
        random_state:  int   = 42,
    ):
        self.n_lags        = n_lags
        self.contamination = contamination
        self.random_state  = random_state
        self.name          = f"IsolationForest(lags={n_lags})"
        self._model        = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def _build_features(self, returns: pd.Series) -> tuple[np.ndarray, pd.Index]:
        """Build lagged feature matrix from return series."""
        values = returns.values
        n      = len(values) - self.n_lags
        X      = np.stack([values[i: i + self.n_lags] for i in range(n)])
        idx    = returns.index[self.n_lags:]
        return X, idx

    def fit(self, returns: pd.Series) -> "IsolationForestDetector":
        """Fit on training returns."""
        X, _ = self._build_features(returns)
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        self._fitted = True
        return self

    def score(self, returns: pd.Series) -> pd.Series:
        """
        Anomaly score: negative of Isolation Forest decision function.
        Higher = more anomalous (consistent with other detectors).
        """
        assert self._fitted, "Call fit() before score()"
        X, idx   = self._build_features(returns)
        X_scaled = self._scaler.transform(X)
        # decision_function: lower = more anomalous, so negate
        scores   = -self._model.decision_function(X_scaled)
        return pd.Series(scores, index=idx, name="isolation_forest")


def get_all_baselines() -> list:
    """Return one instance of each baseline detector."""
    return [
        RollingVolatility(window=20),
        EWMAVolatility(span=20),
        IsolationForestDetector(n_lags=20),
    ]
