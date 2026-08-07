"""Tests for GPT model."""
import math

import pytest
import torch
import torch.nn.functional as F

from gpt.model import GPT


def test_param_count():
    m = GPT(vocab_size=5, context_length=60)
    assert m.param_count() > 0
    assert m.param_count() < 10_000_000


@pytest.mark.parametrize("vocab,ctx,expected", [
    (3, 20, 200_768),   # shipped config (checkpoints/final)
    (3, 60, 203_328),
    (9, 120, 207_552),
])
def test_param_count_is_exact(vocab, ctx, expected):
    """
    Pins the published parameter count. A loose bound cannot catch a doc
    that quotes the wrong config's size.
    """
    assert GPT(vocab_size=vocab, context_length=ctx).param_count() == expected


def test_attention_is_causal():
    """
    Changing a FUTURE token must not change logits at earlier positions.
    This is the core correctness property of a decoder-only transformer;
    a broken mask leaks the target and silently deflates the anomaly score.
    """
    torch.manual_seed(0)
    m = GPT(vocab_size=5, context_length=16)
    m.eval()

    x = torch.randint(0, 5, (1, 16))
    perturbed = x.clone()
    perturbed[0, -1] = (x[0, -1].item() + 1) % 5   # change only the last token

    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(perturbed)

    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6), "future token leaked backwards"
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-6), "last position ignored its input"


def test_untrained_loss_matches_uniform_entropy():
    """
    At init the model is near-uniform, so loss on targets independent of the
    input should sit close to ln(vocab). Catches init and weight-tying bugs
    that a 'loss > 0' assertion cannot.

    Targets must be independent of the input: with weight tying the residual
    stream still carries the input embedding, so scoring x against itself
    beats uniform at init and would mask a real regression.
    """
    for vocab in (3, 5, 9):
        torch.manual_seed(0)
        m = GPT(vocab_size=vocab, context_length=16)
        m.eval()
        x = torch.randint(0, vocab, (32, 16))
        targets = torch.randint(0, vocab, (32, 16))
        with torch.no_grad():
            _, loss = m(x, targets)
        assert loss.item() == pytest.approx(math.log(vocab), abs=0.15), (
            f"vocab={vocab}: expected ~{math.log(vocab):.3f}, got {loss.item():.3f}"
        )


def test_tied_head_favours_input_token_at_init():
    """
    Documents the consequence of weight tying probed above: at init the model
    assigns above-uniform probability to the token it was just shown. Pins the
    behaviour so a future init change does not silently alter anomaly scores.
    """
    torch.manual_seed(0)
    m = GPT(vocab_size=3, context_length=16)
    m.eval()
    x = torch.randint(0, 3, (32, 16))
    with torch.no_grad():
        _, self_loss = m(x, x)
        _, indep_loss = m(x, torch.randint(0, 3, (32, 16)))
    assert self_loss.item() < indep_loss.item()


def test_token_perplexity_matches_cross_entropy():
    """token_perplexity must equal per-token cross-entropy, not merely be positive."""
    torch.manual_seed(0)
    m = GPT(vocab_size=5, context_length=16)
    m.eval()
    idx = torch.randint(0, 5, (2, 17))

    got = m.token_perplexity(idx)

    with torch.no_grad():
        logits, _ = m(idx[:, :-1])
        B, T, C = logits.shape
        want = F.cross_entropy(
            logits.reshape(B * T, C), idx[:, 1:].reshape(B * T), reduction="none"
        ).reshape(B, T)

    assert torch.allclose(got, want, atol=1e-6)


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
