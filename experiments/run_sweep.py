"""
Execute the 3x3 ablation sweep.
Run this on GCP or Colab: python -m experiments.run_sweep

Saves results to experiments/results.csv for offline analysis.
"""

from __future__ import annotations

import csv
import os
import time

import torch

from data.loader import load_all_assets, train_val_split
from data.tokenizer import ReturnTokenizer
from experiments.grid import GRID, grid_to_config
from gpt.model import GPT
from gpt.train import select_device, set_seed, train

RESULTS_PATH = "experiments/results.csv"
TICKER       = "SPY"
WANDB_LOG    = True
SEEDS        = (0, 1, 2)   # every grid point is run once per seed


def run_sweep(wandb_log: bool = WANDB_LOG, seeds: tuple[int, ...] = SEEDS) -> None:
    device = select_device()
    print(f"Device: {device}")
    print(f"Running {len(GRID)}-point ablation sweep x {len(seeds)} seeds on {TICKER}\n")

    all_returns = load_all_assets(tickers=[TICKER])
    returns     = all_returns[TICKER]
    train_ret, val_ret = train_val_split(returns)

    results = []

    total = len(GRID) * len(seeds)
    n     = 0

    for point in GRID:
        for seed in seeds:
            n += 1
            print(f"\n{'='*60}")
            print(f"Run {n}/{total}: {point.run_name} seed={seed}")
            print(f"  vocab_size={point.vocab_size}, context_length={point.context_length}")
            print(f"{'='*60}")

            cfg          = grid_to_config(point, wandb_log=wandb_log)
            cfg.seed     = seed
            cfg.run_name = f"{point.run_name}_s{seed}"
            cfg.ckpt_dir = f"checkpoints/{cfg.run_name}"
            tokenizer    = ReturnTokenizer(vocab_size=point.vocab_size)

            train_tokens = tokenizer.encode(train_ret.values)
            val_tokens   = tokenizer.encode(val_ret.values)

            min_len = point.context_length + 1
            if len(train_tokens) < min_len or len(val_tokens) < min_len:
                print(f"  SKIP: not enough tokens for context_length={point.context_length}")
                continue

            train_flat = torch.tensor(train_tokens, dtype=torch.long)
            val_flat   = torch.tensor(val_tokens,   dtype=torch.long)

            set_seed(seed)   # before init, so weights are reproducible too
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

            # Pair train and val at the SAME step; a min() over each series
            # independently yields a gap between two different points in training.
            best_i     = history["val_loss"].index(min(history["val_loss"]))
            best_val   = history["val_loss"][best_i]
            train_at   = history["train_loss"][best_i]

            results.append({
                "run_name":            point.run_name,
                "seed":                seed,
                "vocab_size":          point.vocab_size,
                "context_length":      point.context_length,
                "best_val_loss":       round(best_val, 4),
                "train_loss_at_best":  round(train_at, 4),
                "gap_at_best":         round(train_at - best_val, 4),
                "params":              model.param_count(),
                "elapsed_s":           round(elapsed, 1),
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
    """Aggregate across seeds. Differences smaller than the seed spread are noise."""
    import pandas as pd

    df = pd.DataFrame(results)
    agg = (
        df.groupby(["run_name", "vocab_size", "context_length"])["best_val_loss"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values("mean")
    )

    print("\n=== SWEEP SUMMARY (mean +/- std across seeds) ===")
    print(f"{'Run':<12} {'Vocab':>6} {'Context':>8} {'Val Loss':>18} {'Seeds':>6}")
    print("-" * 56)
    for _, r in agg.iterrows():
        cell = f"{r['mean']:.4f} +/- {r['std']:.4f}"
        print(
            f"{r['run_name']:<12} {int(r['vocab_size']):>6} {int(r['context_length']):>8} "
            f"{cell:>18} {int(r['count']):>6}"
        )

    print(f"\nMean seed-to-seed std: {agg['std'].mean():.4f}")
    print("Any claimed difference smaller than this is not resolvable by this sweep.")


if __name__ == "__main__":
    run_sweep()