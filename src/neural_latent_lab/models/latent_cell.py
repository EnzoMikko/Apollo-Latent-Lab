"""Blocs Transformer réutilisables.

``TransformerBlock`` est l'unité de calcul partagée par tous les modèles :
- empilée ``n_layers`` fois pour le backbone (baseline / compute-matched),
- réutilisée **k fois avec les mêmes poids** comme cellule de réflexion latente
  (``latent_scratchpad``).

Pre-norm + résidus, attention causale. La "latent cell" du projet est simplement
une instance de ``TransformerBlock`` appliquée plusieurs fois : ``h <- block(h)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_len: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_len: int
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
