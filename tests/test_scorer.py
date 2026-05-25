"""Tests for AnomalyScorer."""
import numpy as np
import pandas as pd
import torch
import pytest
from gpt.model import GPT
from data.tokenizer import ReturnTokenizer
from anomaly.scorer import AnomalyScorer


def make_scorer(vocab_size=5, context_length=20):
    model = GPT(vocab_size=vocab_size, context_length=context_length)
    tok   = ReturnTokenizer(vocab_size=vocab_size)
    return AnomalyScorer(model, tok, context_length)


def make_returns(n=200):
    np.random.seed(42)
    dates   = pd.date_range("2018-01-01", periods=n, freq="B")
    returns = pd.Series(np.random.normal(0, 0.01, n), index=dates)
    return returns


def test_score_length():
    scorer  = make_scorer(context_length=20)
    returns = make_returns(200)
    scores  = scorer.score_sequence(returns)
    assert len(scores) == len(returns) - 20


def test_scores_positive():
    scorer  = make_scorer(context_length=20)
    returns = make_returns(200)
    scores  = scorer.score_sequence(returns)
    assert (scores > 0).all()


def test_scores_finite():
    scorer  = make_scorer(context_length=20)
    returns = make_returns(200)
    scores  = scorer.score_sequence(returns)
    assert scores.isna().sum() == 0


def test_rolling_score_length():
    scorer  = make_scorer(context_length=20)
    returns = make_returns(200)
    scores  = scorer.rolling_score(returns, window=5)
    assert len(scores) == len(returns) - 20


def test_score_all_assets():
    scorer  = make_scorer(context_length=20)
    np.random.seed(0)
    dates   = pd.date_range("2018-01-01", periods=200, freq="B")
    assets  = {
        "SPY": pd.Series(np.random.normal(0, 0.01, 200), index=dates),
        "QQQ": pd.Series(np.random.normal(0, 0.012, 200), index=dates),
    }
    df = scorer.score_all_assets(assets)
    assert "SPY" in df.columns
    assert "QQQ" in df.columns
