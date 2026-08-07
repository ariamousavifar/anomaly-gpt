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


def test_score_matches_manual_nll():
    """
    score_sequence(t) must equal -log P(token_t | context), computed by hand.
    Asserts the published equation, not merely that a number came back.
    """
    torch.manual_seed(0)
    scorer  = make_scorer(vocab_size=5, context_length=20)
    returns = make_returns(60)
    scores  = scorer.score_sequence(returns)

    token_ids = scorer.tokenizer.encode(returns.values)
    for t in (20, 35, 59):
        ctx = torch.tensor(token_ids[t - 20:t], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits, _ = scorer.model(ctx)
        expected = -torch.log_softmax(logits[0, -1, :], dim=-1)[token_ids[t]].item()
        assert scores.iloc[t - 20] == pytest.approx(expected, abs=1e-6)


def test_score_is_bounded_by_vocab_entropy():
    """A uniform predictor scores ln(vocab); an untrained model must be near it."""
    torch.manual_seed(0)
    scorer = make_scorer(vocab_size=3, context_length=20)
    scores = scorer.score_sequence(make_returns(200))
    assert scores.mean() == pytest.approx(np.log(3), abs=0.35)
    assert (scores > 0).all()


def test_higher_score_for_out_of_distribution_token():
    """
    A return far outside the training regime must score at least as surprising
    as a typical one. Catches a scorer wired to the wrong token or sign.
    """
    torch.manual_seed(0)
    scorer = make_scorer(vocab_size=3, context_length=20)
    calm   = pd.Series(
        np.full(40, 0.0005),
        index=pd.date_range("2018-01-01", periods=40, freq="B"),
    )
    shocked = calm.copy()
    shocked.iloc[-1] = -0.09      # crash token, unseen in the flat context

    s_calm    = scorer.score_sequence(calm).iloc[-1]
    s_shocked = scorer.score_sequence(shocked).iloc[-1]
    assert s_shocked > s_calm, "anomalous token did not score more surprising"


def test_rolling_score_smooths_isolated_spike():
    """rolling_score must attenuate a one-day spike relative to the raw score."""
    torch.manual_seed(0)
    scorer  = make_scorer(vocab_size=3, context_length=20)
    returns = make_returns(80)
    raw     = scorer.score_sequence(returns)
    smooth  = scorer.rolling_score(returns, window=5)
    assert smooth.std() < raw.std()
    assert len(smooth) == len(raw)


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
