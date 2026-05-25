"""
Execute the 3x3 ablation sweep.
Run this on GCP or Colab: python -m experiments.run_sweep

Saves results to experiments/results.csv for offline analysis.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from data.loader import load_all_assets, train_val_split
from data.tokenizer import ReturnTokenizer
from experiments.grid import GRID, grid_to_config
from gpt.model import GPT
from gpt.train import train

RESULTS_PATH = "experiments/results.csv"
TICKER       = "SPY"
WANDB_LOG    = True


def run_sweep(wandb_log: bool = WANDB_LOG) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {len(GRID)}-point ablation sweep on {TICKER}\n")

    all_returns = load_all_assets(tickers=[TICKER])
    returns     = all_returns[TICKER]
    train_ret, val_ret = train_val_split(returns)

    results = []

    for i, point in enumerate(GRID):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(GRID)}: {point.run_name}")
        print(f"  vocab_size={point.vocab_size}, context_length={point.context_length}")
        print(f"{'='*60}")

        cfg       = grid_to_config(point, wandb_log=wandb_log)
        tokenizer = ReturnTokenizer(vocab_size=point.vocab_size)

        train_tokens = tokenizer.encode(train_ret.values)
        val_tokens   = tokenizer.encode(val_ret.values)

        min_len = point.context_length + 1
        if len(train_tokens) < min_len or len(val_tokens) < min_len:
            print(f"  SKIP: not enough tokens for context_length={point.context_length}")
            continue

        train_flat = torch.tensor(train_tokens, dtype=torch.long)
        val_flat   = torch.tensor(val_tokens,   dtype=torch.long)

        model = GPT(
            vocab_size     = point.vocab_size,
            context_length = point.context_length,
            n_embd         = cfg.n_embd,
            n_layer        = cfg.n_layer,
            n_head         = cfg.n_head,
            dropout        = cfg.dropout,
        ).to(device)

        print(f"  Parameters: {model.param_count():,}")
        print(f"  Train tokens: {len(train_flat):,} | Val tokens: {len(val_flat):,}")

        t0      = time.time()
        history = train(model, train_flat, val_flat, cfg, device)
        elapsed = time.time() - t0

        best_val   = min(history["val_loss"])
        best_train = min(history["train_loss"])

        results.append({
            "run_name":        point.run_name,
            "vocab_size":      point.vocab_size,
            "context_length":  point.context_length,
            "best_val_loss":   round(best_val,   4),
            "best_train_loss": round(best_train, 4),
            "params":          model.param_count(),
            "elapsed_s":       round(elapsed, 1),
        })

        print(f"  Best val loss: {best_val:.4f} | Time: {elapsed:.1f}s")

    if not results:
        print("No results — all runs skipped.")
        return

    os.makedirs("experiments", exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSweep complete. Results saved to {RESULTS_PATH}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'Run':<12} {'Vocab':>6} {'Context':>8} {'Val Loss':>10} {'Time(s)':>8}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        print(
            f"{r['run_name']:<12} {r['vocab_size']:>6} {r['context_length']:>8} "
            f"{r['best_val_loss']:>10.4f} {r['elapsed_s']:>8.1f}"
        )


if __name__ == "__main__":
    run_sweep()