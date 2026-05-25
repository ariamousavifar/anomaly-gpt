"""
GPT-based anomaly scorer.
Computes per-timestep perplexity (next-token NLL) as a market surprise score.

Core idea:
    The GPT learns P(r_t | r_{t-k}, ..., r_{t-1}) from normal market data.
    Anomaly score at time t = -log P_theta(r_t | context)
    High score = model's prior violated = potential regime change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from gpt.model import GPT
from data.tokenizer import ReturnTokenizer


class AnomalyScorer:
    """
    Computes rolling perplexity anomaly scores from a trained GPT.

    Args:
        model:          trained GPT instance
        tokenizer:      ReturnTokenizer used during training
        context_length: GPT context window (must match model)
        device:         torch device
    """

    def __init__(
        self,
        model:          GPT,
        tokenizer:      ReturnTokenizer,
        context_length: int,
        device:         torch.device | None = None,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.context_length = context_length
        self.device         = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_sequence(self, returns: pd.Series | np.ndarray) -> pd.Series:
        """
        Compute per-timestep anomaly score for a return series.

        For each timestep t (starting at context_length), the score is the
        negative log-likelihood of the observed token given its context:
            score(t) = -log P_theta(r_t | r_{t-context}, ..., r_{t-1})

        Args:
            returns: pd.Series of log-returns (with DatetimeIndex) or np.ndarray

        Returns:
            pd.Series of anomaly scores, same index as returns[context_length:]
        """
        if isinstance(returns, pd.Series):
            index   = returns.index[self.context_length:]
            values  = returns.values
        else:
            index  = np.arange(self.context_length, len(returns))
            values = returns

        token_ids = self.tokenizer.encode(values)
        scores    = []

        for t in range(self.context_length, len(token_ids)):
            ctx    = token_ids[t - self.context_length: t]
            target = token_ids[t]

            x = torch.tensor(ctx, dtype=torch.long, device=self.device).unsqueeze(0)
            logits, _ = self.model(x)
            log_probs  = torch.log_softmax(logits[0, -1, :], dim=-1)
            nll        = -log_probs[target].item()
            scores.append(nll)

        return pd.Series(scores, index=index, name="anomaly_score")

    def rolling_score(
        self,
        returns:     pd.Series,
        window:      int = 5,
    ) -> pd.Series:
        """
        Smooth raw anomaly scores with a rolling mean.
        Reduces noise from single-day spikes.

        Args:
            returns: log-return series
            window:  rolling window size in days

        Returns:
            Smoothed anomaly score series.
        """
        raw = self.score_sequence(returns)
        return raw.rolling(window=window, min_periods=1).mean().rename("anomaly_score_smooth")

    def score_all_assets(
        self,
        all_returns: dict[str, pd.Series],
        window:      int = 5,
    ) -> pd.DataFrame:
        """
        Score multiple assets. Returns DataFrame with one column per asset.
        """
        scores = {}
        for ticker, returns in all_returns.items():
            scores[ticker] = self.rolling_score(returns, window)
        return pd.DataFrame(scores)
