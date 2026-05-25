"""
4-detector comparison plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_detector_comparison(
    gpt_scores:      pd.Series,
    baseline_scores: dict[str, pd.Series],
    event_name:      str,
    window_start:    "date",
    window_end:      "date",
    save_path:       str | None = None,
    figsize:         tuple = (14, 8),
) -> plt.Figure:
    """
    4-panel comparison: one subplot per detector, same time axis.
    """
    all_detectors = {"GPT Perplexity": gpt_scores, **baseline_scores}
    n = len(all_detectors)

    start = pd.Timestamp(window_start) - pd.Timedelta(days=45)
    end   = pd.Timestamp(window_end)   + pd.Timedelta(days=45)
    peak  = pd.Timestamp(window_end)   - pd.Timedelta(days=(
        (pd.Timestamp(window_end) - pd.Timestamp(window_start)).days // 2
    ))

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    for ax, (name, scores), color in zip(axes, all_detectors.items(), colors):
        window = scores[(scores.index >= start) & (scores.index <= end)]
        if len(window) == 0:
            continue

        # Normalize scores to z-scores for fair visual comparison
        z = (window - scores.mean()) / (scores.std() or 1)
        ax.plot(z.index, z.values, color=color, linewidth=1.5, label=name)
        ax.axhline(2.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvspan(pd.Timestamp(window_start), pd.Timestamp(window_end),
                   alpha=0.12, color="red")
        ax.set_ylabel("Z-Score", fontsize=9)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle(
        f"4-Detector Comparison: {event_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
