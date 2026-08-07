# Findings

Every number below is produced by code in this repo. The command that
generates each one is given with it. Numbers that could not be regenerated
have been removed rather than carried over.

Reproduce everything on this page with:

```bash
make score   # regenerate anomaly_scores.csv from the checkpoint
make eval    # print every number below
```

---

## Finding 1 — GPT perplexity detects more documented crises than any baseline, and the margin is not a threshold artifact

`make eval` — `scripts/run_eval.py`

### Detection results, SPY 2010–2024

Fixed `z > 2` threshold on each detector's own score distribution:

| Event                      | GPT | Rolling Vol | EWMA | Isolation Forest |
|----------------------------|-----|-------------|------|------------------|
| Flash Crash 2010           | —   | —           | —    | —                |
| China Devaluation 2015     | YES | —           | YES  | —                |
| XIV Vol Cascade 2018       | YES | —           | —    | —                |
| COVID Crash 2020           | YES | YES         | YES  | YES              |
| Meme Stock Squeeze 2021    | YES | —           | —    | —                |
| Fed Rate Shock 2022        | —   | —           | —    | YES              |
| SVB Collapse 2023          | —   | —           | —    | —                |
| **Total**                  | **4/7** | **1/7** | **2/7** | **2/7**      |

### The threshold control

A fixed `z > 2` is *not* a common operating point. Perplexity and volatility
scores have very different skew, so the same z admits different
false-positive rates:

| Detector | Flags at z>2 | Skew |
|----------|--------------|------|
| GPT Perplexity | 2.11% of days | 0.26 |
| Rolling Volatility | 2.05% | 3.54 |
| EWMA | 2.61% | 3.41 |
| Isolation Forest | 5.46% | 1.64 |

Re-thresholding every detector to flag the same 2.11% of days:

| Event                      | GPT | Rolling Vol | EWMA | Isolation Forest |
|----------------------------|-----|-------------|------|------------------|
| China Devaluation 2015     | YES | —           | —    | —                |
| XIV Vol Cascade 2018       | YES | —           | —    | —                |
| COVID Crash 2020           | YES | YES         | YES  | YES              |
| Meme Stock Squeeze 2021    | YES | —           | —    | —                |
| **Total**                  | **4/7** | **1/7** | **1/7** | **1/7**      |

The margin holds. Across the full range of operating points:

| Flag rate | GPT | Rolling Vol | EWMA | Isolation Forest |
|-----------|-----|-------------|------|------------------|
| 1.0% | 3/7 | 1/7 | 1/7 | 1/7 |
| 2.0% | 4/7 | 1/7 | 1/7 | 1/7 |
| 3.0% | 4/7 | 1/7 | 3/7 | 1/7 |
| 5.0% | 6/7 | 2/7 | 4/7 | 2/7 |

### Significance

Null hypothesis: the event dates carry no information. Realised by
circularly shifting each score series 1000 times, which preserves its
distribution *and* its autocorrelation while randomising alignment with the
event calendar.

| Detector | Observed | Null mean | P(null ≥ observed) |
|----------|----------|-----------|--------------------|
| GPT Perplexity | 4/7 | 0.78 | **0.004** |
| EWMA | 2/7 | 0.34 | 0.049 |
| Isolation Forest | 2/7 | 0.62 | 0.134 |
| Rolling Volatility | 1/7 | 0.17 | 0.148 |

GPT's detection count is the only one significant at the 1% level.

### Events detected only by GPT

At the matched 2.11% flag rate: China Devaluation 2015, XIV Vol Cascade
2018, Meme Stock Squeeze 2021.

This repo makes **no claim about why** those events and not others. The
detector was not designed against a mechanism, and with n=7 events there is
no power to test one.

---

## Ablation — not currently reported

The 3×3 grid (vocab_size × context_length) was originally run without
seeding, so its numbers were not reproducible and every claim resting on
them has been removed.

`experiments/run_sweep.py` is now seeded and runs each grid point across
multiple seeds, reporting mean ± standard deviation. `experiments/results.csv`
is a **pre-seeding artifact and should not be cited**; it will be replaced
when the seeded sweep is run.

Two measurements motivated removing the previous findings:

- Re-running `v3_c20` under three seeds gave validation losses of 0.4684,
  0.4693 and 0.4758 — a spread of ~0.004, which is larger than several of
  the differences the removed findings were built on.
- Validation loss is not comparable across vocabulary sizes. Different
  tokenizations have different marginal entropies, so a lower loss at
  vocab=3 is substantially a property of the alphabet rather than of
  generalisation. Any future cross-vocab claim needs a marginal baseline.

No replacement findings have been written.

---

## Scope and limitations

- **In-sample.** Detection thresholds are computed from the full 2010–2024
  score series, which contains the event windows being tested. This is not a
  forward-looking backtest.
- **n = 7.** Seven events cannot support fine distinctions between detectors.
  The circular-shift test above is the only significance claim made.
- **One asset, daily frequency.** SPY only. Intraday events such as the Flash
  Crash are invisible by construction.
- **Ternary tokenization** discards within-bin magnitude.
- **Checkpoint provenance.** The shipped checkpoint predates the seeding fix,
  so its exact training run cannot be reproduced from this repo.
  `scripts/score_assets.py` reproduces every number above *from that
  checkpoint*; `scripts/train_final.py` documents the configuration.
