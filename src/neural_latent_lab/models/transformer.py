"""Backbone Transformer decoder-only + encodage positionnel configurable.

Le choix de ``pos_encoding`` est un levier scientifique majeur pour la
généralisation OOD en longueur ; il DOIT être identique entre les modèles
comparés. ``nope`` (pas d'encodage positionnel) est souvent le plus robuste
pour la généralisation en longueur sur l'addition.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neural_latent_lab.models.latent_cell import TransformerBlock


class PositionalEncoding(nn.Module):
    def __init__(self, kind: str, d_model: int, max_len: int) -> None:
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pos = nn.Embedding(max_len, d_model)
        elif kind == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        elif kind == "nope":
            pass
        else:
            raise ValueError(f"pos_encoding inconnu: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "learned":
            pos = torch.arange(x.size(1), device=x.device)
            return x + self.pos(pos).unsqueeze(0)
        if self.kind == "sinusoidal":
            return x + self.pe[:, : x.size(1)]
        return x  # nope


class BaselineTransformer(nn.Module):
    """Decoder-only standard : embedding + ``depth`` blocs + LM head.

    Sert de baseline (``depth = n_layers``) et de modèle compute-matched
    (``depth = n_layers + k``, couches distinctes).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        depth: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pos_encoding: str,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(pos_encoding, d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, max_len) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.pos_enc(self.tok_emb(input_ids)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(self.backbone(input_ids)))
