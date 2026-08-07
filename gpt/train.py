"""
Training loop for GPT financial sequence model.
- AdamW optimizer with weight decay
- Cosine LR schedule with linear warmup
- W&B logging
- Checkpoint saving
"""

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import wandb



def select_device() -> torch.device:
    """CUDA if present, then Apple MPS, else CPU.

    Checking only for CUDA silently falls back to CPU on Apple silicon,
    which is by far the slowest option available on those machines.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training (init, batch sampling, dropout)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    # Model
    vocab_size:     int   = 5
    context_length: int   = 60
    n_embd:         int   = 64
    n_layer:        int   = 4
    n_head:         int   = 4
    dropout:        float = 0.1
    n_assets:       int   = 1

    # Training
    max_iters:      int   = 5000
    batch_size:     int   = 64
    learning_rate:  float = 3e-4
    weight_decay:   float = 0.1
    grad_clip:      float = 1.0
    warmup_iters:   int   = 100
    eval_interval:  int   = 250
    eval_iters:     int   = 50

    # Logging
    wandb_log:      bool  = True
    wandb_project:  str   = "anomaly-gpt"
    run_name:       str   = "default"

    # Checkpointing
    ckpt_dir:       str   = "checkpoints"

    # Reproducibility
    seed:           int   = 0


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_batch(data, context_length, batch_size, device, asset_id=None):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x  = torch.stack([data[i:     i + context_length] for i in ix]).to(device)
    y  = torch.stack([data[i + 1: i + context_length + 1] for i in ix]).to(device)
    asset_ids = None
    if asset_id is not None:
        asset_ids = torch.full((batch_size,), asset_id, dtype=torch.long, device=device)
    return x, y, asset_ids


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, asset_id=None):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y, aids = get_batch(data, cfg.context_length, cfg.batch_size, device, asset_id)
            _, loss     = model(x, y, aids)
            losses[k]   = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, cfg, device, asset_id=None):
    """Full training loop. Returns history dict with train/val losses.

    Seeds from cfg.seed so batch sampling and dropout are reproducible.
    Callers must also call set_seed(cfg.seed) before constructing the model
    if they need weight initialisation to be reproducible too.
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    if cfg.wandb_log:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=cfg.__dict__,
            reinit=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    history = {"train_loss": [], "val_loss": [], "lr": [], "step": []}
    best_val_loss = float("inf")

    model.train()
    for step in range(cfg.max_iters):

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, device, asset_id)
            print(
                f"step {step:5d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f} | lr {lr:.2e}"
            )
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["lr"].append(lr)
            history["step"].append(step)

            if cfg.wandb_log:
                wandb.log({
                    "train/loss": losses["train"],
                    "val/loss":   losses["val"],
                    "lr":         lr,
                    "step":       step,
                })

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")
                torch.save({
                    "step":     step,
                    "model":    model.state_dict(),
                    "val_loss": best_val_loss,
                    "config":   cfg.__dict__,
                }, ckpt_path)

        x, y, aids = get_batch(train_data, cfg.context_length, cfg.batch_size, device, asset_id)
        _, loss = model(x, y, aids)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    if cfg.wandb_log:
        wandb.finish()

    return history
