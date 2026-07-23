"""From-scratch GPT decoder for the 28-token math vocabulary.

Standard pre-norm transformer: token + learned positional embeddings, causal
self-attention (via ``F.scaled_dot_product_attention``), GELU MLP, RMSNorm,
and a weight-tied LM head. Architecture is fully described by
:class:`GPTConfig`, so a model variation (see ``variations.py``) plus a seed
reproduces the network exactly.

The model trains in pure bf16 on CUDA (see ``model_dtype()``): RMSNorm is
hand-written and deliberately does NOT upcast to float32 — everything runs in
the model's native dtype.

Shapes use the convention B = batch, T = sequence length, V = vocab size,
C = embedding dim (``n_embd``). Shape asserts guard the boundaries where a
silent broadcasting bug would otherwise propagate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class GPTConfig(BaseModel):
    """Architecture hyperparameters. Frozen per model variation."""

    vocab_size: int = 28
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


def model_dtype(device: torch.device) -> torch.dtype:
    """Project precision policy: pure bf16 on CUDA (L40S), fp32 elsewhere
    (CPU/MPS smoke runs). Scripts call ``model.to(device, model_dtype(device))``."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class RMSNorm(nn.Module):
    """Hand-written RMSNorm, computed in the input's native dtype.

    y = x / sqrt(mean(x^2) + eps) * weight — no bias, and intentionally no
    float32 upcast (the usual ``x.float()`` dance) since we train in pure bf16.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled_dot_product_attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # combined projection for query, key, value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape to (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the LM head shares the token embedding matrix
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 init
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 convention)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count. With ``non_embedding``, subtract the
        positional embedding (the token embedding is tied to the LM head and
        so is genuinely part of the model's parameters)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) token ids
        assert idx.dim() == 2, f"expected idx (B, T), got shape {tuple(idx.shape)}"
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, C)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        assert logits.shape == (B, T, self.config.vocab_size), (
            f"expected logits (B, T, V)=({B}, {T}, {self.config.vocab_size}), "
            f"got {tuple(logits.shape)}"
        )
        return logits
