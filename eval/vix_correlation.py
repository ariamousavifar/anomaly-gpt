"""
VIX correlation analysis.
Compares GPT anomaly scores against VIX as ground-truth stress measure.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def align_with_vix(
    anomaly_scores: pd.Series,
    vix_returns:    pd.Series,
) -> pd.DataFrame:
    """Align anomaly scores with VIX on common dates."""
    df = pd.concat(
        [anomaly_scores.rename("anomaly_score"), vix_returns.rename("vix")],
        axis=1,
    ).dropna()
    return df


def pearson_correlation(anomaly_scores: pd.Series, vix_returns: pd.Series) -> dict:
    """Pearson correlation between anomaly scores and VIX returns."""
    df  = align_with_vix(anomaly_scores, vix_returns)
    r, p = stats.pearsonr(df["anomaly_score"], df["vix"])
    return {"pearson_r": float(r), "p_value": float(p), "n": len(df)}


def spearman_correlation(anomaly_scores: pd.Series, vix_returns: pd.Series) -> dict:
    """Spearman rank correlation (robust to outliers)."""
    df  = align_with_vix(anomaly_scores, vix_returns)
    rho, p = stats.spearmanr(df["anomaly_score"], df["vix"])
    return {"spearman_rho": float(rho), "p_value": float(p), "n": len(df)}


def lead_lag_analysis(
    anomaly_scores: pd.Series,
    vix_returns:    pd.Series,
    max_lag:        int = 10,
) -> pd.DataFrame:
    """
    Compute Spearman correlation at lags -max_lag to +max_lag.
    Negative lag = anomaly score leads VIX (GPT detects early).
    Positive lag = VIX leads anomaly score (GPT reacts late).

    Returns DataFrame with columns [lag, spearman_rho, p_value].
    """
    df   = align_with_vix(anomaly_scores, vix_returns)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_vix = df["vix"].shift(-lag)
        aligned     = pd.concat([df["anomaly_score"], shifted_vix], axis=1).dropna()
        if len(aligned) < 10:
            continue
        rho, p = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        rows.append({"lag": lag, "spearman_rho": rho, "p_value": p})

    result = pd.DataFrame(rows)
    result["interpretation"] = result["lag"].apply(
        lambda l: "GPT leads VIX" if l < 0 else ("simultaneous" if l == 0 else "GPT lags VIX")
    )
    return result


def correlation_report(anomaly_scores: pd.Series, vix_returns: pd.Series) -> str:
    """Generate a text summary of VIX correlation results."""
    pearson  = pearson_correlation(anomaly_scores, vix_returns)
    spearman = spearman_correlation(anomaly_scores, vix_returns)
    lead_lag = lead_lag_analysis(anomaly_scores, vix_returns)
    best_lag = lead_lag.loc[lead_lag["spearman_rho"].idxmax()]

    lines = [
        "=== VIX Correlation Report ===",
        f"Pearson  r   = {pearson['pearson_r']:.3f}  (p={pearson['p_value']:.3e}, n={pearson['n']})",
        f"Spearman rho = {spearman['spearman_rho']:.3f}  (p={spearman['p_value']:.3e})",
        f"Best lag     = {best_lag['lag']} days  (rho={best_lag['spearman_rho']:.3f})",
        f"Interpretation: {best_lag['interpretation']}",
    ]
    return "\n".join(lines)
