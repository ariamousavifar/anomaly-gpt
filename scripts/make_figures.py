"""
Regenerate every committed figure from committed artifacts.

    python -m scripts.make_figures

Reads experiments/results.csv and anomaly_scores.csv, writes the PNGs
referenced by README.md and FINDINGS.md. Run this after make sweep or
make score so the figures never drift from the numbers.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from viz.heatmap import plot_ablation_heatmap
from viz.plot import plot_anomaly_timeline

RESULTS_CSV = "experiments/results.csv"
SCORES_CSV  = "anomaly_scores.csv"
HEATMAP_PNG = "notebooks/ablation_heatmap.png"
TIMELINE_PNG = "notebooks/anomaly_timeline_SPY.png"


def main() -> None:
    # Only plot the ablation once it has been produced by the seeded runner.
    # A 'seed' column is the marker; without it the CSV is a pre-seeding
    # artifact whose numbers are withdrawn, and plotting it would republish them.
    results = pd.read_csv(RESULTS_CSV)
    if "seed" in results.columns:
        plot_ablation_heatmap(RESULTS_CSV, save_path=HEATMAP_PNG, figsize=(8, 5))
    else:
        print(f"SKIP {HEATMAP_PNG}: {RESULTS_CSV} has no 'seed' column "
              "(pre-seeding artifact). Run `make sweep` first.")

    scores = pd.read_csv(SCORES_CSV, index_col="Date", parse_dates=True)
    fig = plot_anomaly_timeline(
        scores["SPY"],
        title="GPT Perplexity Anomaly Score — SPY 2010–2024",
        vix=scores["VIX"],
        figsize=(16, 6),
    )
    fig.savefig(TIMELINE_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {TIMELINE_PNG}")


if __name__ == "__main__":
    main()
