"""
Return discretizer: continuous log-returns -> discrete token IDs.

5-bin alphabet (default):
    0  crash:  r < -0.03
    1  down:  -0.03 <= r < -0.01
    2  flat:  -0.01 <= r < +0.01
    3  up:    +0.01 <= r < +0.03
    4  surge:  r >= +0.03

Thresholds configurable via vocab_size: 3, 5, or 9.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


DEFAULT_BINS = {
    3: [-0.01, 0.01],
    5: [-0.03, -0.01, 0.01, 0.03],
    9: [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05],
}

DEFAULT_LABELS = {
    3: ["down", "flat", "up"],
    5: ["crash", "down", "flat", "up", "surge"],
    9: ["crash5", "crash3", "crash2", "down", "flat", "up", "surge2", "surge3", "surge5"],
}


class ReturnTokenizer:
    """
    Discretizes log-returns into integer token IDs.

    Args:
        vocab_size: 3, 5, or 9
        bins:       custom bin edges (overrides default)
    """

    def __init__(self, vocab_size: int = 5, bins: list[float] | None = None):
        assert vocab_size in DEFAULT_BINS or bins is not None, \
            f"vocab_size must be one of {list(DEFAULT_BINS.keys())} or provide custom bins"
        self.vocab_size = vocab_size
        self.bins       = bins if bins is not None else DEFAULT_BINS[vocab_size]
        self.labels     = DEFAULT_LABELS.get(vocab_size, [str(i) for i in range(vocab_size)])

    def encode(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """Convert log-returns to token IDs in [0, vocab_size)."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        token_ids = np.digitize(returns, bins=self.bins).astype(np.int64)
        assert token_ids.min() >= 0 and token_ids.max() < self.vocab_size, \
            "Token IDs out of range — check bin edges"
        return token_ids

    def decode(self, token_ids: np.ndarray) -> list[str]:
        """Convert token IDs back to label strings."""
        return [self.labels[i] for i in token_ids]

    def bin_distribution(self, returns: np.ndarray | pd.Series) -> dict[str, float]:
        """Fraction of tokens in each bin — useful for sanity checks."""
        token_ids = self.encode(returns)
        total     = len(token_ids)
        return {
            label: float((token_ids == i).sum()) / total
            for i, label in enumerate(self.labels)
        }

    def __repr__(self) -> str:
        return f"ReturnTokenizer(vocab_size={self.vocab_size}, bins={self.bins})"
