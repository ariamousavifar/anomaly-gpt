# Model Card — anomaly-gpt

## Model Description
Decoder-only GPT (203K parameters) trained on discretized financial log-returns.
Uses next-token perplexity as a market anomaly score.

## Training Data
SPY log-returns, 2010–2019 (pre-COVID), ~2,500 trading days.

## Architecture
- vocab_size: 5 (crash / down / flat / up / surge)
- context_length: 60 trading days
- n_layer: 4, n_head: 4, n_embd: 64
- Weight tying: token embedding ↔ output head

## Intended Use
Financial anomaly detection research. Not a trading signal.

## Limitations
- Trained on US equity data only
- Binary tokenization loses magnitude information within bins
- Not suitable for production trading systems

## Results
See FINDINGS.md and notebooks/01_anomaly_detection.ipynb.
