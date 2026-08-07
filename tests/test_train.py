"""Tests for the training loop: seeding, LR schedule, split integrity."""
import tempfile

import pandas as pd
import pytest
import torch

from data.loader import train_val_split
from gpt.model import GPT
from gpt.train import TrainConfig, get_lr, set_seed, train


def _tiny_cfg(seed, ckpt_dir):
    return TrainConfig(
        vocab_size=3, context_length=8, n_embd=16, n_layer=2, n_head=2,
        max_iters=60, batch_size=8, eval_interval=25, eval_iters=3,
        wandb_log=False, ckpt_dir=ckpt_dir, seed=seed,
    )


def _run(seed, ckpt_dir):
    """Mirrors real usage: seed, build model, train."""
    cfg = _tiny_cfg(seed, ckpt_dir)
    # fixed data across runs so only the training seed varies
    data = torch.randint(0, 3, (400,), generator=torch.Generator().manual_seed(999))
    set_seed(cfg.seed)
    model = GPT(vocab_size=cfg.vocab_size, context_length=cfg.context_length,
                n_embd=cfg.n_embd, n_layer=cfg.n_layer, n_head=cfg.n_head)
    return train(model, data[:300], data[300:], cfg, torch.device("cpu"))


def test_training_is_deterministic_under_seed():
    """Same seed must give bit-identical losses. Fails if the repo never seeds."""
    with tempfile.TemporaryDirectory() as d:
        a = _run(123, d)
        b = _run(123, d)
    assert a["val_loss"] == b["val_loss"], "same seed produced different val losses"
    assert a["train_loss"] == b["train_loss"], "same seed produced different train losses"


def test_different_seeds_diverge():
    """Guards against the seed being ignored entirely (e.g. hardcoded)."""
    with tempfile.TemporaryDirectory() as d:
        a = _run(1, d)
        b = _run(2, d)
    assert a["val_loss"] != b["val_loss"], "different seeds gave identical losses"


def test_get_lr_warmup_is_linear():
    cfg = TrainConfig(learning_rate=1e-3, warmup_iters=100, max_iters=1000)
    assert get_lr(0, cfg) == 0.0
    assert get_lr(50, cfg) == pytest.approx(5e-4)
    assert get_lr(100, cfg) == pytest.approx(1e-3)


def test_get_lr_cosine_decays_to_zero():
    cfg = TrainConfig(learning_rate=1e-3, warmup_iters=100, max_iters=1000)
    assert get_lr(550, cfg) == pytest.approx(5e-4, rel=1e-3)
    assert get_lr(1000, cfg) == pytest.approx(0.0, abs=1e-12)
    lrs = [get_lr(s, cfg) for s in range(100, 1000, 50)]
    assert lrs == sorted(lrs, reverse=True), "cosine schedule must be monotone decreasing"


def test_train_val_split_is_disjoint_and_ordered():
    """Leakage guard: no shared dates, and val strictly follows train."""
    idx = pd.date_range("2010-01-01", periods=1000, freq="B")
    returns = pd.Series(range(1000), index=idx, dtype=float)
    tr, va = train_val_split(returns, train_end="2013-01-01", val_frac=0.1)
    assert len(tr) > 0 and len(va) > 0
    assert set(tr.index).isdisjoint(set(va.index)), "train and val share dates"
    assert tr.index.max() < va.index.min(), "validation must come after training"
    assert va.index.max() < pd.Timestamp("2013-01-01"), "val leaked past train_end"
