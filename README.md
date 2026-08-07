# anomaly-gpt

**GPT-based financial anomaly detector**: next-token perplexity as a market surprise score, evaluated on 7 documented crises against rolling volatility, EWMA, and Isolation Forest baselines.

[![CI](https://github.com/AriaMF/anomaly-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/AriaMF/anomaly-gpt/actions)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The Idea

A language model trained on normal market sequences learns an implicit prior over return dynamics. Anomalies are violations of that prior, and cross-entropy loss is a measure of surprise:

```
Anomaly Score(t) = -log P_θ(r_t | r_{t-19}, ..., r_{t-1})
```

High score = the model did not expect this token given its context.

---

## Key Result

**GPT perplexity detects 4/7 documented crises vs 1/7 for rolling volatility, and the margin survives a false-positive-rate control and a permutation test.**

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

A fixed `z > 2` threshold is not a common operating point — perplexity and
volatility scores have very different skew. Re-thresholding all four
detectors to flag the same 2.11% of days keeps GPT at 4/7 and drops every
baseline to 1/7. The margin holds at every flag rate from 1% to 5%.

Against a circular-shift null (1000 shifts, preserving each series'
distribution and autocorrelation), GPT's count is the only one significant
at the 1% level:

| Detector | Observed | Null mean | p |
|----------|----------|-----------|---|
| GPT Perplexity | 4/7 | 0.78 | **0.004** |
| EWMA | 2/7 | 0.34 | 0.049 |
| Isolation Forest | 2/7 | 0.62 | 0.134 |
| Rolling Volatility | 1/7 | 0.17 | 0.148 |

Full tables and the commands that generate them: [FINDINGS.md](FINDINGS.md).

---

## Anomaly Score Timeline

![Anomaly Timeline](notebooks/anomaly_timeline_SPY.png)

---

## Architecture

Decoder-only GPT trained on discretized log-returns:

```
Daily returns → log transform → 3-bin discretization (down/flat/up)
    → token sequence → GPT (4 layers, 4 heads, 64 dim) → next-token NLL
```

- **200,768 parameters** (`GPT(vocab_size=3, context_length=20)`)
- **Weight tying**: token embedding ↔ output head
- **AdamW + cosine LR schedule** with linear warmup
- **Trained on SPY 2010–2019** (pre-COVID), evaluated on 2010–2024

The model is heavily over-parameterised relative to its ~2,264 training
tokens. This is a deliberate simplicity choice, not a tuned one; a capacity
ablation has not been run.

---

## Quickstart

```bash
git clone https://github.com/AriaMF/anomaly-gpt.git
cd anomaly-gpt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
```

Outside an activated venv, pass the interpreter explicitly:
`make test PYTHON=.venv/bin/python`.

The trained weights are **not** in this repo. Download them first:

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('checkpoints/final', exist_ok=True)
p = hf_hub_download(repo_id='AriaMF/anomaly-gpt', filename='final_model.pt')
shutil.copy(p, 'checkpoints/final/final_model.pt')
"
```

Then reproduce every published number:

```bash
make score   # regenerate anomaly_scores.csv from the checkpoint
make eval    # print every detection table, FPR control and p-value
```

`scripts/score_assets.py` falls back to downloading the checkpoint
automatically if `checkpoints/final/final_model.pt` is absent.

---

## Repo Structure

```
anomaly-gpt/
├── gpt/              # GPT architecture + training loop
├── data/             # yfinance loader, return tokenizer
├── anomaly/          # perplexity scorer, baselines, event registry, thresholds
├── eval/             # detection harness, VIX correlation
├── experiments/      # 3×3 ablation grid + seeded sweep runner
├── scripts/          # train_final, score_assets, run_eval, make_figures
├── viz/              # timeline plots, ablation heatmap
├── notebooks/        # end-to-end demo
├── tests/            # 57 tests
├── FINDINGS.md       # results, each with its generating command
└── .github/          # CI
```

---

## Baselines

All four detectors are evaluated on identical event windows and compared at
a matched false-positive rate:

| Detector | Type | Signal |
|----------|------|--------|
| GPT Perplexity | Sequential model | Next-token NLL, 5-day smoothed |
| Rolling Volatility | Classical | 20-day std of returns |
| EWMA Volatility | Classical | Exponentially weighted std |
| Isolation Forest | ML | Anomaly score on 20 lagged returns |

---

## Ablation

The 3×3 grid over vocabulary size and context length was originally run
without seeding, so its results were not reproducible and the claims built
on them have been withdrawn. `experiments/run_sweep.py` is now seeded and
runs each point across multiple seeds; `experiments/results.csv` is a
pre-seeding artifact and should not be cited until the sweep is re-run.

See [FINDINGS.md](FINDINGS.md#ablation--not-currently-reported).

---

## Scope and Limitations

This is a **research platform**, not a trading signal.

- Thresholds are computed in-sample over the full 2010–2024 series, which
  includes the event windows being tested. Not a forward-looking backtest.
- 7 events is too few to support fine distinctions; the permutation test
  above is the only significance claim made.
- SPY only, daily frequency — intraday events (Flash Crash) are invisible
  by construction.
- Ternary tokenization discards within-bin magnitude.
- The shipped checkpoint predates the seeding fix, so its exact training run
  cannot be reproduced. `scripts/score_assets.py` reproduces all published
  numbers from that checkpoint; `scripts/train_final.py` documents its
  configuration.

---

## Model Weights

Pretrained checkpoint: [AriaMF/anomaly-gpt](https://huggingface.co/AriaMF/anomaly-gpt)

## Experiment Tracking

[Weights & Biases](https://api.wandb.ai/links/ariamosavefar-universit-de-gen-ve/icnkshoa)

## Attribution

The transformer in `gpt/model.py` is adapted from coursework for
**Deep Learning, University of Geneva (2024)**, cleaned and extended with
`param_count()` and asset-embedding support.

The return tokenizer, perplexity scorer, baseline detectors, event registry
and evaluation harness are original to this repo.
