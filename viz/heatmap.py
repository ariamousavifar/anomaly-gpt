"""
Ablation heatmap: vocab_size x context_length -> val_loss.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ablation_heatmap(
    results_csv: str = "experiments/results.csv",
    metric:      str = "best_val_loss",
    save_path:   str | None = None,
    figsize:     tuple = (8, 6),
) -> plt.Figure:
    """
    Plot 3x3 heatmap of ablation results.

    Args:
        results_csv: path to sweep results CSV
        metric:      column to display (default: best_val_loss)
        save_path:   if provided, save figure
        figsize:     figure size
    """
    df = pd.read_csv(results_csv)

    pivot = df.pivot_table(
        index="context_length",
        columns="vocab_size",
        values=metric,
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd_r",
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )

    # Mark best cell
    best_idx = pivot.stack().idxmin()
    best_row = list(pivot.index).index(best_idx[0])
    best_col = list(pivot.columns).index(best_idx[1])
    ax.add_patch(plt.Rectangle(
        (best_col, best_row), 1, 1,
        fill=False, edgecolor="blue", lw=3, label="Best config"
    ))

    ax.set_title(
        f"Ablation Grid: Val Loss vs Vocab Size x Context Length\n"
        f"Best: vocab={best_idx[1]}, context={best_idx[0]} ({pivot.loc[best_idx[0], best_idx[1]]:.4f})",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Vocab Size (number of return bins)", fontsize=11)
    ax.set_ylabel("Context Length (trading days)", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_detection_heatmap(
    results:   pd.DataFrame,
    save_path: str | None = None,
    figsize:   tuple = (12, 5),
) -> plt.Figure:
    """
    Heatmap of z-scores: detectors x events.
    """
    pivot = results.pivot_table(
        index="detector",
        columns="event",
        values="z_score",
        aggfunc="first",
    ).round(2)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=2.0,
        linewidths=0.5,
        cbar_kws={"label": "Z-Score (>2 = detected)"},
    )
    ax.set_title(
        "Anomaly Detection Z-Scores: All Detectors x All Events",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
