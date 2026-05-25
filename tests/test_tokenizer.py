"""Tests for ReturnTokenizer."""
import numpy as np
import pytest
from data.tokenizer import ReturnTokenizer


def test_encode_range():
    tok = ReturnTokenizer(vocab_size=5)
    returns = np.random.uniform(-0.1, 0.1, 1000)
    ids = tok.encode(returns)
    assert ids.min() >= 0
    assert ids.max() < 5


def test_decode_length():
    tok = ReturnTokenizer(vocab_size=5)
    returns = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
    ids = tok.encode(returns)
    labels = tok.decode(ids)
    assert len(labels) == 5


def test_known_bins_5():
    tok = ReturnTokenizer(vocab_size=5)
    returns = np.array([-0.05, -0.02, 0.0, 0.02, 0.05])
    ids = tok.encode(returns)
    assert ids[0] == 0  # crash
    assert ids[1] == 1  # down
    assert ids[2] == 2  # flat
    assert ids[3] == 3  # up
    assert ids[4] == 4  # surge


def test_known_bins_3():
    tok = ReturnTokenizer(vocab_size=3)
    returns = np.array([-0.05, 0.0, 0.05])
    ids = tok.encode(returns)
    assert ids[0] == 0  # down
    assert ids[1] == 1  # flat
    assert ids[2] == 2  # up


def test_known_bins_9():
    tok = ReturnTokenizer(vocab_size=9)
    returns = np.array([0.0])
    ids = tok.encode(returns)
    assert ids[0] == 4  # flat center bin


def test_bin_distribution_sums_to_one():
    tok = ReturnTokenizer(vocab_size=5)
    returns = np.random.normal(0, 0.01, 10000)
    dist = tok.bin_distribution(returns)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_invalid_vocab_size():
    with pytest.raises(AssertionError):
        ReturnTokenizer(vocab_size=7)
