"""
Perplexity timeline plots with event markers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from anomaly.events import EVENTS, MarketEvent


def plot_anomaly_timeline(
    scores:      pd.Series,
    title:       str = "GPT Perplexity Anomaly Score",
    events:      list[MarketEvent] | None = None,
    vix:         pd.Series | None = None,
    figsize:     tuple = (16, 6),
    save_path:   str | None = None,
) -> plt.Figure:
    """
    Plot anomaly scores over time with event markers.

    Args:
        scores:     anomaly score series with DatetimeIndex
        title:      plot title
        events:     list of MarketEvent to mark (default: all 7)
        vix:        optional VIX series to overlay
        figsize:    figure size
        save_path:  if provided, save to this path
    """
    events = events or EVENTS
    fig, ax = plt.subplots(figsize=figsize)

    # Plot anomaly score
    ax.plot(scores.index, scores.values, color="#2196F3", linewidth=1.2,
            alpha=0.8, label="GPT Perplexity Score", zorder=2)

    # 2-sigma threshold
    mean, std = scores.mean(), scores.std()
    ax.axhline(mean + 2 * std, color="#FF5722", linestyle="--",
               linewidth=1.0, alpha=0.7, label="2σ threshold", zorder=1)

    # Event markers
    colors = plt.cm.Set1(np.linspace(0, 1, len(events)))
    for event, color in zip(events, colors):
        peak = pd.Timestamp(event.peak_date)
        if peak in scores.index or (scores.index.min() <= peak <= scores.index.max()):
            ax.axvline(peak, color=color, linewidth=1.5, alpha=0.8, zorder=3)
            y_pos = scores.max() * 0.95
            ax.annotate(
                event.name.replace(" ", "\n"),
                xy=(peak, y_pos),
                fontsize=7,
                color=color,
                rotation=0,
                ha="center",
                va="top",
            )

    # Optional VIX overlay
    if vix is not None:
        ax2 = ax.twinx()
        ax2.plot(vix.index, vix.values, color="#4CAF50", linewidth=0.8,
                 alpha=0.5, label="VIX", zorder=0)
        ax2.set_ylabel("VIX", color="#4CAF50", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#4CAF50")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Anomaly Score (NLL)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_event_zoom(
    scores:      pd.Series,
    event:       MarketEvent,
    all_scores:  dict[str, pd.Series] | None = None,
    figsize:     tuple = (12, 5),
    save_path:   str | None = None,
) -> plt.Figure:
    """
    Zoomed plot for a single event window with all detector scores overlaid.
    """
    # Add 30-day buffer around event window
    start = pd.Timestamp(event.window_start) - pd.Timedelta(days=30)
    end   = pd.Timestamp(event.window_end)   + pd.Timedelta(days=30)

    fig, ax = plt.subplots(figsize=figsize)

    # GPT score
    window = scores[(scores.index >= start) & (scores.index <= end)]
    ax.plot(window.index, window.values, color="#2196F3", linewidth=2,
            label="GPT Perplexity", zorder=3)

    # Baseline scores
    if all_scores:
        colors = ["#FF9800", "#9C27B0", "#4CAF50"]
        for (name, s), color in zip(all_scores.items(), colors):
            w = s[(s.index >= start) & (s.index <= end)]
            if len(w) > 0:
                # Normalize to same scale as GPT scores for visual comparison
                s_norm = (w - w.mean()) / (w.std() or 1) * window.std() + window.mean()
                ax.plot(w.index, s_norm.values, linewidth=1.5,
                        color=color, alpha=0.7, label=name)

    # Event window shading
    ax.axvspan(
        pd.Timestamp(event.window_start),
        pd.Timestamp(event.window_end),
        alpha=0.15, color="red", label="Event window"
    )
    ax.axvline(pd.Timestamp(event.peak_date), color="red",
               linewidth=2, linestyle="--", label="Peak date")

    ax.set_title(f"Event Deep-Dive: {event.name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Anomaly Score (normalized)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
