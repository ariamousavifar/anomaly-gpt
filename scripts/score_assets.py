"""
Regenerate anomaly_scores.csv from the trained checkpoint.

This is the script that produces every anomaly number published in
README.md and FINDINGS.md.

    python -m scripts.score_assets

The checkpoint is not committed (see checkpoints/README.md); it is either
downloaded from HuggingFace or read from checkpoints/final/final_model.pt.
Scoring is deterministic: the model is in eval() mode under torch.no_grad(),
so no seed is required.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from anomaly.scorer import AnomalyScorer
from data.loader import ASSETS, load_all_assets
from data.tokenizer import ReturnTokenizer
from gpt.model import GPT

CKPT_LOCAL   = "checkpoints/final/final_model.pt"
HF_REPO      = "AriaMF/anomaly-gpt"
HF_FILENAME  = "final_model.pt"
OUTPUT_PATH  = "anomaly_scores.csv"
SMOOTH_WINDOW = 5


def resolve_checkpoint(path: str = CKPT_LOCAL) -> str:
    """Use the local checkpoint if present, else pull it from HuggingFace."""
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    print(f"{path} not found — downloading from {HF_REPO}")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def load_model(ckpt_path: str) -> tuple[GPT, dict]:
    """Rebuild the model from the config embedded in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    model = GPT(
        vocab_size     = cfg["vocab_size"],
        context_length = cfg["context_length"],
        n_embd         = cfg["n_embd"],
        n_layer        = cfg["n_layer"],
        n_head         = cfg["n_head"],
        dropout        = cfg["dropout"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main(output: str = OUTPUT_PATH) -> pd.DataFrame:
    ckpt_path   = resolve_checkpoint()
    model, cfg  = load_model(ckpt_path)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Config     : vocab={cfg['vocab_size']} context={cfg['context_length']} "
          f"run={cfg.get('run_name')} val_loss={cfg.get('val_loss', 'n/a')}")
    print(f"Parameters : {model.param_count():,}")

    returns = load_all_assets(tickers=list(ASSETS.keys()))
    tok     = ReturnTokenizer(vocab_size=cfg["vocab_size"])
    scorer  = AnomalyScorer(model, tok, context_length=cfg["context_length"])

    scores = scorer.score_all_assets(returns, window=SMOOTH_WINDOW)
    scores.index.name = "Date"
    scores.to_csv(output)

    print(f"\nWrote {output}  shape={scores.shape}")
    print(f"Date range: {scores.index[0].date()} -> {scores.index[-1].date()}")
    return scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=OUTPUT_PATH)
    main(ap.parse_args().output)
