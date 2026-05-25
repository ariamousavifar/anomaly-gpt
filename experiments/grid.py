"""
3x3 ablation grid definition.
Axes: vocab_size (3, 5, 9) x context_length (20, 60, 120).
9 training runs total — each a separate W&B run.
"""

from __future__ import annotations
from dataclasses import dataclass
from gpt.train import TrainConfig


@dataclass
class GridPoint:
    vocab_size:     int
    context_length: int
    run_name:       str


def build_grid() -> list[GridPoint]:
    """Return all 9 grid points."""
    vocab_sizes     = [3, 5, 9]
    context_lengths = [20, 60, 120]
    grid = []
    for v in vocab_sizes:
        for c in context_lengths:
            grid.append(GridPoint(
                vocab_size     = v,
                context_length = c,
                run_name       = f"v{v}_c{c}",
            ))
    return grid


def grid_to_config(point: GridPoint, wandb_log: bool = True) -> TrainConfig:
    """Convert a grid point to a TrainConfig."""
    return TrainConfig(
        vocab_size     = point.vocab_size,
        context_length = point.context_length,
        n_embd         = 64,
        n_layer        = 4,
        n_head         = 4,
        dropout        = 0.1,
        max_iters      = 3000,
        batch_size     = 64,
        learning_rate  = 3e-4,
        wandb_log      = wandb_log,
        wandb_project  = "anomaly-gpt",
        run_name       = point.run_name,
        ckpt_dir       = f"checkpoints/{point.run_name}",
    )


GRID = build_grid()
