"""
Sliding window sequence builder.
Converts token ID arrays into (input, target) tensors for GPT training.
"""

from __future__ import annotations

import numpy as np
import torch

from data.tokenizer import ReturnTokenizer


def build_sequences(
    token_ids:      np.ndarray,
    context_length: int,
) -> torch.Tensor:
    """
    Convert a 1-D token ID array into a tensor of shape (N, context_length+1).
    Each row is a context window; training uses row[:-1] as input, row[1:] as target.

    Args:
        token_ids:      1-D integer array
        context_length: GPT context window size

    Returns:
        torch.Tensor of shape (N, context_length+1), dtype=long
    """
    n      = len(token_ids) - context_length
    assert n > 0, f"Sequence too short: {len(token_ids)} <= {context_length}"
    rows   = np.stack([token_ids[i: i + context_length + 1] for i in range(n)])
    return torch.tensor(rows, dtype=torch.long)


def prepare_dataset(
    returns:        "pd.Series",
    tokenizer:      ReturnTokenizer,
    context_length: int,
    train_end:      str = "2020-01-01",
    val_frac:       float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full pipeline: returns -> tokens -> train/val/full tensors.

    Returns:
        train_data: token tensor for training (pre-COVID, minus val)
        val_data:   token tensor for validation
        full_data:  token tensor for full date range (anomaly evaluation)
    """
    from data.loader import train_val_split

    token_ids_full = tokenizer.encode(returns.values)

    train_ret, val_ret = train_val_split(returns, train_end, val_frac)
    token_ids_train    = tokenizer.encode(train_ret.values)
    token_ids_val      = tokenizer.encode(val_ret.values)

    train_data = build_sequences(token_ids_train, context_length)
    val_data   = build_sequences(token_ids_val,   context_length)
    full_data  = build_sequences(token_ids_full,  context_length)

    return train_data, val_data, full_data


def build_joint_dataset(
    all_returns:    dict[str, "pd.Series"],
    tokenizer:      ReturnTokenizer,
    context_length: int,
    train_end:      str = "2020-01-01",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Build interleaved multi-asset dataset for joint training.
    Returns (train_data, val_data, asset_ids_list).
    """
    from data.loader import train_val_split

    all_train, all_val = [], []
    asset_ids_train, asset_ids_val = [], []

    for asset_id, (ticker, returns) in enumerate(all_returns.items()):
        train_ret, val_ret = train_val_split(returns, train_end)
        t_train = build_sequences(tokenizer.encode(train_ret.values), context_length)
        t_val   = build_sequences(tokenizer.encode(val_ret.values),   context_length)
        all_train.append(t_train)
        all_val.append(t_val)
        asset_ids_train.extend([asset_id] * len(t_train))
        asset_ids_val.extend([asset_id]   * len(t_val))

    train_data = torch.cat(all_train, dim=0)
    val_data   = torch.cat(all_val,   dim=0)

    return train_data, val_data, asset_ids_train
