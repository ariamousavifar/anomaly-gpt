"""
Produce every published detection number.

    python -m scripts.run_eval

Reports each event at two operating points:

  1. Fixed z > 2 threshold (the repo's original criterion).
  2. Matched flag rate — all detectors thresholded so they flag the same
     fraction of days.

(1) is not a common operating point: perplexity and volatility scores have
very different skew, so the same z admits very different false-positive
rates. Both are printed so the comparison's threshold-sensitivity is visible.
"""

from __future__ import annotations

import argparse

import pandas as pd

from anomaly.baselines import get_all_baselines
from anomaly.detector import flag_rate, score_event_window, score_event_window_at_rate
from anomaly.events import EVENTS
from data.loader import download_returns

SCORES_PATH = "anomaly_scores.csv"
TICKER      = "SPY"
TRAIN_END   = "2020-01-01"


def build_detectors(scores_path: str = SCORES_PATH) -> dict[str, pd.Series]:
    """GPT scores from disk plus the three baselines, all on SPY."""
    gpt     = pd.read_csv(scores_path, index_col="Date", parse_dates=True)[TICKER]
    returns = download_returns(TICKER)

    detectors = {"GPT Perplexity": gpt}
    for b in get_all_baselines():
        if hasattr(b, "_fitted"):
            # fit unsupervised baselines on pre-COVID data only
            b.fit(returns[returns.index < TRAIN_END])
        detectors[b.name] = b.score(returns).dropna()
    return detectors


def table_fixed_z(detectors: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for event in EVENTS:
        row = {"event": event.name}
        for name, s in detectors.items():
            row[name] = score_event_window(s, event.window_start, event.window_end)["detected"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def table_matched_rate(detectors: dict[str, pd.Series], rate: float) -> pd.DataFrame:
    rows = []
    for event in EVENTS:
        row = {"event": event.name}
        for name, s in detectors.items():
            row[name] = score_event_window_at_rate(
                s, event.window_start, event.window_end, rate
            )["detected"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def _render(df: pd.DataFrame) -> str:
    shown = df.replace({True: "YES", False: "-"})
    totals = pd.DataFrame(
        [{c: f"{int(df[c].sum())}/{len(df)}" for c in df.columns}], index=["TOTAL"]
    )
    return pd.concat([shown, totals]).to_string()


def exclusive_to_gpt(df: pd.DataFrame, gpt_col: str = "GPT Perplexity") -> list[str]:
    others = [c for c in df.columns if c != gpt_col]
    return [e for e in df.index if df.loc[e, gpt_col] and not df.loc[e, others].any()]


def circular_shift_null(
    scores:   pd.Series,
    n_shifts: int = 1000,
    seed:     int = 0,
) -> dict:
    """
    Significance of a detector's event-detection count.

    Null: the event dates carry no information. Realised by circularly
    shifting the score series, which preserves its distribution AND its
    autocorrelation (both matter — a shuffle would destroy the smoothing and
    badly understate chance detections) while randomising alignment with the
    event calendar.

    Returns observed count, null mean, and P(null >= observed).
    """
    import numpy as np

    def count(s: pd.Series) -> int:
        return sum(
            score_event_window(s, e.window_start, e.window_end)["detected"]
            for e in EVENTS
        )

    observed = count(scores)
    rng      = np.random.default_rng(seed)
    n        = len(scores)
    shifts   = rng.choice(np.arange(50, n - 50), size=n_shifts, replace=False)
    null     = np.array([
        count(pd.Series(np.roll(scores.values, int(k)), index=scores.index))
        for k in shifts
    ])
    return {
        "observed":  observed,
        "null_mean": float(null.mean()),
        "p_value":   float((null >= observed).mean()),
    }


def main(scores_path: str = SCORES_PATH) -> None:
    detectors = build_detectors(scores_path)

    print("=" * 78)
    print("IMPLIED FALSE-POSITIVE RATE AT z > 2")
    print("=" * 78)
    for name, s in detectors.items():
        print(f"  {name:<26} flags {flag_rate(s):6.2%} of {len(s):,} days   "
              f"(skew {s.skew():.2f})")

    gpt_rate = flag_rate(detectors["GPT Perplexity"])

    print()
    print("=" * 78)
    print("OPERATING POINT 1 — fixed z > 2 (NOT rate-matched)")
    print("=" * 78)
    t1 = table_fixed_z(detectors)
    print(_render(t1))
    print(f"\n  GPT-exclusive: {exclusive_to_gpt(t1) or 'none'}")

    print()
    print("=" * 78)
    print(f"OPERATING POINT 2 — all detectors matched to {gpt_rate:.2%} of days flagged")
    print("=" * 78)
    t2 = table_matched_rate(detectors, gpt_rate)
    print(_render(t2))
    print(f"\n  GPT-exclusive: {exclusive_to_gpt(t2) or 'none'}")

    print()
    print("=" * 78)
    print("SIGNIFICANCE — circular-shift null, 1000 shifts")
    print("=" * 78)
    for name, s in detectors.items():
        r = circular_shift_null(s)
        print(f"  {name:<26} observed {r['observed']}/7   "
              f"null mean {r['null_mean']:.2f}   p = {r['p_value']:.3f}")

    print()
    print("=" * 78)
    print("SENSITIVITY — detections vs flag rate")
    print("=" * 78)
    print(f"{'rate':>8} " + " ".join(f"{n.split('(')[0][:14]:>15}" for n in detectors))
    for rate in (0.01, 0.02, 0.03, 0.05, gpt_rate):
        t = table_matched_rate(detectors, rate)
        print(f"{rate:>8.2%} " + " ".join(f"{int(t[c].sum()):>13}/7" for c in t.columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default=SCORES_PATH)
    main(ap.parse_args().scores)
