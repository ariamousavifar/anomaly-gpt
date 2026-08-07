"""
Financial data loader.
Downloads OHLCV data via yfinance, computes log-returns,
caches to disk to avoid repeated API calls.
"""

from __future__ import annotations

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

# Yahoo Finance symbols. Indices are prefixed with '^' — passing the bare
# label (e.g. "VIX") silently fetches an unrelated instrument.
ASSET_SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "GLD": "GLD",
    "TLT": "TLT",
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
    """Load log-returns for all assets. Returns dict {label: pd.Series}.

    Keys are labels (e.g. "VIX"); the Yahoo symbol (e.g. "^VIX") is resolved
    via ASSET_SYMBOLS. An unrecognised label is passed through unchanged.
    """
    tickers = tickers or list(ASSETS.keys())
    return {
        t: download_returns(ASSET_SYMBOLS.get(t, t), start, end, use_cache)
        for t in tickers
    }


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
