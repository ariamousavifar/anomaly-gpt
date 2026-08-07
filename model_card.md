---
language: en
license: mit
tags:
- pytorch
- finance
- anomaly-detection
- time-series
- transformer
- perplexity
datasets:
- yahoo-finance
metrics:
- loss
---

# anomaly-gpt

GPT-based financial anomaly detector. Uses next-token perplexity as a market surprise score, evaluated on 7 documented crises against rolling volatility, EWMA, and Isolation Forest baselines.

## Model Description

Decoder-only GPT (200,768 parameters) trained on discretized SPY log-returns (2010–2019).
Anomaly score = negative log-likelihood of the observed return token given a 20-day context:

```
Anomaly Score(t) = -log P_θ(r_t | r_{t-19}, ..., r_{t-1})
```

Scores are smoothed with a 5-day rolling mean before thresholding.

## Results

**GPT perplexity detects 4/7 documented crises vs 1/7 for rolling volatility.**

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

A fixed `z > 2` threshold is not a common operating point across detectors
with different score skew. Matched to a common 2.11% flag rate, GPT stays at
4/7 while every baseline drops to 1/7. Against a circular-shift null
(1000 shifts), GPT's count is the only one significant at the 1% level
(p = 0.004; rolling volatility p = 0.148).

Events detected only by GPT at the matched flag rate: China Devaluation 2015,
XIV Vol Cascade 2018, Meme Stock Squeeze 2021. No mechanism is claimed for
why these and not others — with n=7 there is no power to test one.

## Architecture

- **vocab_size:** 3 (down / flat / up)
- **context_length:** 20 trading days
- **n_layer:** 4, **n_head:** 4, **n_embd:** 64
- **Parameters:** 200,768
- **Weight tying:** token embedding ↔ output head
- **Optimizer:** AdamW (β=0.9, 0.95) + cosine LR schedule with linear warmup
- **Training data:** SPY log-returns 2010–2019, 2,264 tokens (251 held out for validation)
- **Evaluation:** SPY 2010–2024

The model is heavily over-parameterised relative to its training set. This is
a simplicity choice, not a tuned one; no capacity ablation has been run.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="AriaMF/anomaly-gpt", filename="final_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Clone the repo first: github.com/AriaMF/anomaly-gpt
from gpt.model import GPT
model = GPT(vocab_size=3, context_length=20)
model.load_state_dict(ckpt["model"])
model.eval()

from data.tokenizer import ReturnTokenizer
from anomaly.scorer import AnomalyScorer
from data.loader import download_returns

tok     = ReturnTokenizer(vocab_size=3)
scorer  = AnomalyScorer(model, tok, context_length=20)
returns = download_returns("SPY")
scores  = scorer.rolling_score(returns)
print(scores.tail())
```

## Limitations

- Thresholds are computed in-sample over the full evaluation series, which
  contains the event windows being tested. Not a forward-looking backtest.
- 7 events is too few to support fine distinctions between detectors.
- SPY only, daily frequency — intraday events are undetectable by construction.
- Ternary tokenization discards within-bin magnitude.
- This checkpoint predates the repo's seeding fix, so its exact training run
  is not reproducible. `scripts/score_assets.py` reproduces every published
  number from these weights; `scripts/train_final.py` documents the config.
- Not a trading signal — research platform only.

## Attribution

The transformer implementation is adapted from coursework for **Deep Learning,
University of Geneva (2024)**. The return tokenizer, perplexity scorer,
baseline detectors, event registry and evaluation harness are original.

## Links

- GitHub: https://github.com/AriaMF/anomaly-gpt
- Weights & Biases: https://api.wandb.ai/links/ariamosavefar-universit-de-gen-ve/icnkshoa
