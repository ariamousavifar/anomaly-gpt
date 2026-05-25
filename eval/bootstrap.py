"""
Bootstrap confidence intervals for anomaly detection metrics.
Kills the 'cherry-picked' objection by showing statistical rigor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(
    data:       np.ndarray,
    statistic:  callable = np.mean,
    n_boot:     int   = 1000,
    ci:         float = 0.95,
    seed:       int   = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng       = np.random.default_rng(seed)
    point_est = statistic(data)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(point_est), float(lower), float(upper)


def z_score_ci(
    scores:       pd.Series,
    window_start: "date",
    window_end:   "date",
    n_boot:       int = 1000,
) -> dict:
    """
    Bootstrap CI for peak z-score in an event window.

    Returns dict with point estimate and 95% CI bounds.
    """
    window_scores = scores[
        (scores.index >= pd.Timestamp(window_start)) &
        (scores.index <= pd.Timestamp(window_end))
    ].values

    global_mean = scores.mean()
    global_std  = scores.std() or 1e-8

    if len(window_scores) == 0:
        return {"z_point": np.nan, "z_lower": np.nan, "z_upper": np.nan}

    def z_max(x):
        return (np.max(x) - global_mean) / global_std

    point, lower, upper = bootstrap_ci(window_scores, statistic=z_max, n_boot=n_boot)
    return {"z_point": point, "z_lower": lower, "z_upper": upper}


def vix_spearman_ci(
    anomaly_scores: pd.Series,
    vix_returns:    pd.Series,
    n_boot:         int = 1000,
) -> dict:
    """
    Bootstrap CI for Spearman correlation between anomaly scores and VIX.
    """
    aligned = pd.concat([anomaly_scores, vix_returns], axis=1).dropna()
    if len(aligned) < 10:
        return {"rho": np.nan, "lower": np.nan, "upper": np.nan}

    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values

    rng  = np.random.default_rng(42)
    rhos = []
    for _ in range(n_boot):
        idx  = rng.choice(len(x), size=len(x), replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(rho)

    rho_point, _ = stats.spearmanr(x, y)
    return {
        "rho":   float(rho_point),
        "lower": float(np.percentile(rhos, 2.5)),
        "upper": float(np.percentile(rhos, 97.5)),
    }
