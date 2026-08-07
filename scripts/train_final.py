"""
Train the final anomaly-detection model.

    python -m scripts.train_final

The hyperparameters below are the ones embedded in the shipped checkpoint
(checkpoints/final/final_model.pt), read back from its own `config` field.

Provenance note: that checkpoint was produced before seeding was added, so
re-running this script will NOT reproduce it bit-for-bit. It reproduces the
configuration, not the original random draw. To regenerate every published
number from the shipped weights instead, use scripts/score_assets.py.

Training is CPU-bound and takes roughly 8x a single sweep run. Prefer a
machine with CUDA; the device is selected automatically below.
"""

from __future__ import annotations

import argparse

import torch

from data.loader import download_returns, train_val_split
from data.tokenizer import ReturnTokenizer
from gpt.model import GPT
from gpt.train import TrainConfig, select_device, set_seed, train

TICKER = "SPY"


def build_config(seed: int = 0, wandb_log: bool = False) -> TrainConfig:
    """Configuration of the shipped checkpoint."""
    return TrainConfig(
        vocab_size     = 3,
        context_length = 20,
        n_embd         = 64,
        n_layer        = 4,
        n_head         = 4,
        dropout        = 0.1,
        n_assets       = 1,
        max_iters      = 8000,
        batch_size     = 64,
        learning_rate  = 3e-4,
        weight_decay   = 0.1,
        grad_clip      = 1.0,
        warmup_iters   = 100,
        eval_interval  = 250,
        eval_iters     = 50,
        wandb_log      = wandb_log,
        wandb_project  = "anomaly-gpt",
        run_name       = "final_v3_c20_SPY",
        ckpt_dir       = "checkpoints/final",
        seed           = seed,
    )


def main(seed: int = 0, wandb_log: bool = False) -> None:
    cfg    = build_config(seed=seed, wandb_log=wandb_log)
    device = select_device()

    returns            = download_returns(TICKER)
    train_ret, val_ret = train_val_split(returns)
    tokenizer          = ReturnTokenizer(vocab_size=cfg.vocab_size)

    train_flat = torch.tensor(tokenizer.encode(train_ret.values), dtype=torch.long)
    val_flat   = torch.tensor(tokenizer.encode(val_ret.values),   dtype=torch.long)

    set_seed(cfg.seed)   # before init, so weights are reproducible too
    model = GPT(
        vocab_size     = cfg.vocab_size,
        context_length = cfg.context_length,
        n_embd         = cfg.n_embd,
        n_layer        = cfg.n_layer,
        n_head         = cfg.n_head,
        dropout        = cfg.dropout,
    ).to(device)

    print(f"Device      : {device}")
    print(f"Parameters  : {model.param_count():,}")
    print(f"Train tokens: {len(train_flat):,} | Val tokens: {len(val_flat):,}")
    print(f"Seed        : {cfg.seed}\n")

    history = train(model, train_flat, val_flat, cfg, device)

    best_i = history["val_loss"].index(min(history["val_loss"]))
    print(f"\nBest val loss : {history['val_loss'][best_i]:.4f}")
    print(f"Train at best : {history['train_loss'][best_i]:.4f}")
    print(f"Checkpoint    : {cfg.ckpt_dir}/{cfg.run_name}_best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true", help="enable W&B logging")
    main(ap.parse_args().seed, ap.parse_args().wandb)
