"""
Execute the 3x3 ablation sweep.
Run this on the GCP VM: python -m experiments.run_sweep

Saves results to experiments/results.csv for offline analysis.
"""

from __future__ import annotations

import csv
import os
import time

import torch

from data.loader import load_all_assets
from data.tokenizer import ReturnTokenizer
from data.sequences import prepare_dataset
from experiments.grid import GRID, grid_to_config
from gpt.model import GPT
from gpt.train import train

RESULTS_PATH = "experiments/results.csv"
TICKER       = "SPY"          # train on SPY for the sweep
WANDB_LOG    = True


def run_sweep(wandb_log: bool = WANDB_LOG) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running {len(GRID)}-point ablation sweep on {TICKER}\n")

    # Download data once
    all_returns = load_all_assets(tickers=[TICKER])
    returns     = all_returns[TICKER]

    results = []

    for i, point in enumerate(GRID):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(GRID)}: {point.run_name}")
        print(f"  vocab_size={point.vocab_size}, context_length={point.context_length}")
        print(f"{'='*60}")

        cfg       = grid_to_config(point, wandb_log=wandb_log)
        tokenizer = ReturnTokenizer(vocab_size=point.vocab_size)

        train_data, val_data, _ = prepare_dataset(
            returns, tokenizer, point.context_length
        )

        # Put data on flat tensors (not windowed) for get_batch
        train_flat = train_data[:, 0]  # just the first token per window
        # Actually use full flat token array
        from data.loader import train_val_split
        train_ret, val_ret = train_val_split(returns)
        import numpy as np
        train_flat = torch.tensor(tokenizer.encode(train_ret.values), dtype=torch.long)
        val_flat   = torch.tensor(tokenizer.encode(val_ret.values),   dtype=torch.long)

        model = GPT(
            vocab_size     = point.vocab_size,
            context_length = point.context_length,
            n_embd         = cfg.n_embd,
            n_layer        = cfg.n_layer,
            n_head         = cfg.n_head,
            dropout        = cfg.dropout,
        ).to(device)

        print(f"  Parameters: {model.param_count():,}")

        t0      = time.time()
        history = train(model, train_flat, val_flat, cfg, device)
        elapsed = time.time() - t0

        best_val = min(history["val_loss"])
        best_train = min(history["train_loss"])

        results.append({
            "run_name":       point.run_name,
            "vocab_size":     point.vocab_size,
            "context_length": point.context_length,
            "best_val_loss":  round(best_val, 4),
            "best_train_loss": round(best_train, 4),
            "params":         model.param_count(),
            "elapsed_s":      round(elapsed, 1),
        })

        print(f"  Best val loss: {best_val:.4f} | Time: {elapsed:.1f}s")

    # Save results
    os.makedirs("experiments", exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSweep complete. Results saved to {RESULTS_PATH}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'Run':<12} {'Vocab':>6} {'Context':>8} {'Val Loss':>10} {'Train Loss':>11}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        print(
            f"{r['run_name']:<12} {r['vocab_size']:>6} {r['context_length']:>8} "
            f"{r['best_val_loss']:>10.4f} {r['best_train_loss']:>11.4f}"
        )


if __name__ == "__main__":
    run_sweep()
