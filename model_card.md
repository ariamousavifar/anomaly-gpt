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

GPT-based financial anomaly detector. Uses next-token perplexity as a market surprise score, evaluated on 7 historical crises against rolling volatility, EWMA, and Isolation Forest baselines.

## Model Description

Decoder-only GPT (203K parameters) trained on discretized SPY log-returns (2010–2019).
Anomaly score = negative log-likelihood of observed return token given 20-day context:

```
Anomaly Score(t) = -log P_θ(r_t | r_{t-19}, ..., r_{t-1})
```

High score = the model's learned prior was violated = potential regime change.

## Results

**GPT perplexity detects 4/7 documented market crises vs 1/7 for rolling volatility.**

| Event                      | GPT  | Rolling Vol | EWMA | Isolation Forest |
|----------------------------|------|-------------|------|-----------------|
| Flash Crash 2010           | —    | —           | —    | —               |
| China Devaluation 2015     | —    | —           | ✓    | —               |
| XIV Vol Cascade 2018       | —    | —           | —    | —               |
| COVID Crash 2020           | ✓    | ✓           | ✓    | ✓               |
| Meme Stock Squeeze 2021    | ✓    | —           | —    | —               |
| Fed Rate Shock 2022        | ✓    | —           | —    | ✓               |
| SVB Collapse 2023          | ✓    | —           | —    | —               |
| **Total**                  | **4/7** | **1/7** | **2/7** | **2/7**   |

GPT exclusively detects **SVB Collapse (2023)** and **Meme Stock Squeeze (2021)** — missed by every baseline. These are sequential anomalies, not pure magnitude events — exactly what a language model captures and rolling volatility cannot.

## Architecture

- **vocab_size:** 3 (down / flat / up — 3-bin return discretization)
- **context_length:** 20 trading days (~1 month)
- **n_layer:** 4, **n_head:** 4, **n_embd:** 64
- **Parameters:** 203K
- **Weight tying:** token embedding ↔ output head
- **Optimizer:** AdamW (β=0.9, 0.95) + cosine LR schedule with linear warmup
- **Training data:** SPY log-returns 2010–2019 (pre-COVID only)
- **Evaluation:** SPY 2010–2024

## Ablation Results

3×3 grid search across vocab size and context length:

| vocab_size | avg val loss |
|------------|-------------|
| 3          | 0.4959 ✓ best |
| 5          | 0.5273      |
| 9          | 0.5888      |

Coarser tokenization generalizes 15.3% better. Longer context consistently hurts — financial returns are near-memoryless beyond 20 days.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="AriaMF/anomaly-gpt", filename="final_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")

# Load model (clone repo first: github.com/ariamousavifar/anomaly-gpt)
from gpt.model import GPT
model = GPT(vocab_size=3, context_length=20)
model.load_state_dict(ckpt["model"])
model.eval()

# Score returns
from data.tokenizer import ReturnTokenizer
from anomaly.scorer import AnomalyScorer
from data.loader import download_returns

tok    = ReturnTokenizer(vocab_size=3)
scorer = AnomalyScorer(model, tok, context_length=20)

returns = download_returns("SPY")
scores  = scorer.rolling_score(returns)
print(scores.tail())
```

## Limitations

- Trained on US equity data only (SPY)
- Daily frequency — intraday events (e.g. Flash Crash) are not detectable
- Binary tokenization loses within-bin magnitude information
- Results evaluated on known historical events — not a forward-looking backtest
- Not a trading signal — research platform only

## Links

- GitHub: https://github.com/ariamousavifar/anomaly-gpt
- Weights & Biases: https://api.wandb.ai/links/ariamosavefar-universit-de-gen-ve/icnkshoa
