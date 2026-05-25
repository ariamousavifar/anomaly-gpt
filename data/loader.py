"""
Financial data loader.
Downloads OHLCV data via yfinance, computes log-returns,
caches to disk to avoid repeated API calls.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/cache")

ASSETS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "VIX": "CBOE Volatility Index",
    "GLD": "Gold ETF",
    "TLT": "20+ Year Treasury Bond ETF",
}

TRAIN_START = "2010-01-01"
TRAIN_END   = "2020-01-01"   # train on pre-COVID only
EVAL_START  = "2010-01-01"   # full range for anomaly evaluation
EVAL_END    = "2024-01-01"


def download_returns(
    ticker:     str,
    start:      str = EVAL_START,
    end:        str = EVAL_END,
    use_cache:  bool = True,
) -> pd.Series:
    """
    Download adjusted close prices and compute log-returns.
    Caches result to data/cache/{ticker}.pkl.

    Returns:
        pd.Series of log-returns indexed by date, NaNs dropped.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.pkl"

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Downloading {ticker} ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    prices  = df["Close"].squeeze()
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = ticker

    with open(cache_path, "wb") as f:
        pickle.dump(returns, f)

    return returns


def load_all_assets(
    tickers:    list[str] | None = None,
    start:      str = EVAL_START,
    end:        str = EVAL_END,
    use_cache:  bool = True,
) -> dict[str, pd.Series]:
    """Load log-returns for all assets. Returns dict {ticker: pd.Series}."""
    tickers = tickers or list(ASSETS.keys())
    return {t: download_returns(t, start, end, use_cache) for t in tickers}


def train_val_split(
    returns:        pd.Series,
    train_end:      str = TRAIN_END,
    val_frac:       float = 0.1,
) -> tuple[pd.Series, pd.Series]:
    """
    Split returns into train (pre-COVID) and validation sets.
    Train: start -> train_end
    Val:   last val_frac of train period
    """
    train_series = returns[returns.index < train_end]
    n_val        = max(1, int(len(train_series) * val_frac))
    train        = train_series.iloc[:-n_val]
    val          = train_series.iloc[-n_val:]
    return train, val


def returns_summary(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Print summary statistics for all assets."""
    rows = []
    for ticker, r in returns.items():
        rows.append({
            "ticker": ticker,
            "n_days": len(r),
            "start":  r.index[0].date(),
            "end":    r.index[-1].date(),
            "mean":   f"{r.mean():.4f}",
            "std":    f"{r.std():.4f}",
            "min":    f"{r.min():.4f}",
            "max":    f"{r.max():.4f}",
        })
    return pd.DataFrame(rows).set_index("ticker")
