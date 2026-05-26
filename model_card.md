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

GPT-based financial anomaly detector. Uses next-token perplexity as a market surprise score.

## Model Description

Decoder-only GPT (203K parameters) trained on discretized SPY log-returns (2010–2019).
Anomaly score = negative log-likelihood of observed return token given 20-day context.

## Results

| Detector | Events Detected (out of 7) |
|----------|---------------------------|
| **GPT Perplexity** | **4/7** |
| Rolling Volatility | 1/7 |
| EWMA | 2/7 |
| Isolation Forest | 2/7 |

GPT exclusively detects SVB Collapse (2023) and Meme Stock Squeeze (2021) — missed by all baselines.

## Architecture

- vocab_size: 3 (down / flat / up)
- context_length: 20 trading days
- n_layer: 4, n_head: 4, n_embd: 64
- Parameters: 203K
- Weight tying: token embedding ↔ output head
- Optimizer: AdamW + cosine LR schedule

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="AriaMF/anomaly-gpt", filename="final_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")

# Load model
from gpt.model import GPT
model = GPT(vocab_size=3, context_length=20)
model.load_state_dict(ckpt["model"])
model.eval()

# Score returns
from data.tokenizer import ReturnTokenizer
from anomaly.scorer import AnomalyScorer
from data.loader import download_returns

tok = ReturnTokenizer(vocab_size=3)
scorer = AnomalyScorer(model, tok, context_length=20)
returns = download_returns("SPY")
scores = scorer.rolling_score(returns)
print(scores.tail())
```

## Limitations

- Trained on US equity data only (SPY)
- Daily frequency — intraday events not detectable
- Not a trading signal — research platform only

## Links

- GitHub: https://github.com/ariamousavifar/anomaly-gpt
- W&B: https://api.wandb.ai/links/ariamosavefar-universit-de-gen-ve/icnkshoa
