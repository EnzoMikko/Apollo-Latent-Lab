"""Fabrique de modèles à partir de la config.

Triangle de comparaison scientifique (cf. plan) :
- ``baseline``        : n_layers couches, k=0.
- ``latent``          : n_layers + cellule partagée ×k  -> +FLOPs, ~mêmes params.
- ``compute_matched`` : n_layers + k couches distinctes  -> +FLOPs ET +params.
"""

from __future__ import annotations

import torch.nn as nn

from neural_latent_lab.models.latent_scratchpad import LatentScratchpad
from neural_latent_lab.models.transformer import BaselineTransformer
from neural_latent_lab.utils.config import ModelConfig


def build_model(cfg: ModelConfig, vocab_size: int) -> nn.Module:
    if cfg.kind == "baseline":
        return BaselineTransformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            depth=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pos_encoding=cfg.pos_encoding,
        )
    if cfg.kind == "compute_matched":
        return BaselineTransformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            depth=cfg.n_layers + cfg.k,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pos_encoding=cfg.pos_encoding,
        )
    if cfg.kind == "latent":
        return LatentScratchpad(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pos_encoding=cfg.pos_encoding,
            k=cfg.k,
        )
    raise ValueError(f"model.kind inconnu: {cfg.kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
