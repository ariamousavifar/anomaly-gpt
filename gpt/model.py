"""
GPT decoder-only transformer for financial sequence modeling.
Architecture: causal self-attention, multi-head, feed-forward blocks.
Adapted from course implementation — cleaned, extended with param_count()
and asset embedding support for multi-asset joint training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, head_size: int, context_length: int, n_embd: int, dropout: float):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) / self.scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_head: int, head_size: int, context_length: int,
                 n_embd: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, context_length, n_embd, dropout)
            for _ in range(n_head)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward block (expand 4x, GELU, contract)."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> LayerNorm -> FFN (pre-norm)."""

    def __init__(self, n_head: int, context_length: int, n_embd: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, head_size, context_length, n_embd, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffn  = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Decoder-only GPT for financial return sequence modeling.

    Args:
        vocab_size:      number of discrete return tokens (e.g. 5)
        context_length:  number of timesteps in context window
        n_embd:          embedding dimension
        n_layer:         number of transformer blocks
        n_head:          number of attention heads
        dropout:         dropout probability
        n_assets:        if > 1, adds a learned asset embedding
    """

    def __init__(
        self,
        vocab_size:     int,
        context_length: int,
        n_embd:         int   = 64,
        n_layer:        int   = 4,
        n_head:         int   = 4,
        dropout:        float = 0.1,
        n_assets:       int   = 1,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size     = vocab_size

        self.token_emb    = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(context_length, n_embd)
        self.asset_emb    = nn.Embedding(n_assets, n_embd) if n_assets > 1 else None

        self.blocks = nn.Sequential(*[
            Block(n_head, context_length, n_embd, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f  = nn.LayerNorm(n_embd)
        self.head  = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None, asset_ids=None):
        B, T = idx.shape
        assert T <= self.context_length

        tok = self.token_emb(idx)
        pos = self.position_emb(torch.arange(T, device=idx.device))
        x   = tok + pos

        if self.asset_emb is not None and asset_ids is not None:
            x = x + self.asset_emb(asset_ids).unsqueeze(1)

        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)
        return idx

    @torch.no_grad()
    def token_perplexity(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Per-token NLL for anomaly scoring.
        Returns shape (B, T-1).
        """
        self.eval()
        inputs  = idx[:, :-1]
        targets = idx[:, 1:]
        logits, _ = self(inputs)
        B, T, C   = logits.shape
        nll = F.cross_entropy(
            logits.reshape(B * T, C),
            targets.reshape(B * T),
            reduction="none",
        )
        return nll.reshape(B, T)
