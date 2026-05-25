"""Tests for GPT model."""
import pytest
import torch
from gpt.model import GPT


def test_param_count():
    m = GPT(vocab_size=5, context_length=60)
    assert m.param_count() > 0
    assert m.param_count() < 10_000_000


def test_forward_shape():
    m = GPT(vocab_size=5, context_length=60)
    x = torch.randint(0, 5, (2, 60))
    logits, loss = m(x, x)
    assert logits.shape == (2, 60, 5)
    assert loss is not None
    assert loss.item() > 0


def test_loss_finite():
    m = GPT(vocab_size=5, context_length=60)
    x = torch.randint(0, 5, (4, 60))
    _, loss = m(x, x)
    assert torch.isfinite(loss)


def test_generate():
    m = GPT(vocab_size=5, context_length=60)
    x = torch.randint(0, 5, (1, 10))
    out = m.generate(x, max_new_tokens=5)
    assert out.shape == (1, 15)
    assert out.min() >= 0 and out.max() < 5


def test_token_perplexity_shape():
    m = GPT(vocab_size=5, context_length=60)
    x = torch.randint(0, 5, (2, 61))
    scores = m.token_perplexity(x)
    assert scores.shape == (2, 60)
    assert torch.all(scores > 0)


def test_asset_embedding():
    m = GPT(vocab_size=5, context_length=60, n_assets=3)
    x = torch.randint(0, 5, (2, 60))
    aids = torch.tensor([0, 1])
    logits, loss = m(x, x, asset_ids=aids)
    assert logits.shape == (2, 60, 5)


def test_weight_tying():
    m = GPT(vocab_size=5, context_length=60)
    assert m.head.weight is m.token_emb.weight


def test_context_length_assertion():
    m = GPT(vocab_size=5, context_length=60)
    x = torch.randint(0, 5, (1, 61))
    with pytest.raises(AssertionError):
        m(x)
