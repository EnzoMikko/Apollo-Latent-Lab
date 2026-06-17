"""Latent Scratchpad : backbone + cellule latente partagée appliquée k fois.

Flux : embedding -> backbone (n_layers) -> état latent h0 -> cellule partagée
répétée k fois (h_{t+1} = cell(h_t)) -> LM head.

Propriétés clés :
* La cellule a des **poids partagés** sur les k applications (récurrence en
  profondeur, style Universal Transformer) : on ajoute du *calcul* sans ajouter
  de *paramètres* -> isole l'effet structurel (H1/H2).
* ``k = 0`` saute entièrement la cellule -> sortie **identique** au baseline
  (vérifié par ``tests/test_model_shapes.py``).
* ``forward(..., return_latents=True)`` renvoie ``[h0, h1, ..., hk]`` pour l'analyse.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_latent_lab.models.latent_cell import TransformerBlock
from neural_latent_lab.models.transformer import BaselineTransformer


class LatentScratchpad(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pos_encoding: str,
        k: int,
    ) -> None:
        super().__init__()
        self.k = k
        self.backbone_model = BaselineTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            depth=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pos_encoding=pos_encoding,
        )
        # Cellule latente partagée (None si k == 0 pour égaler exactement le baseline).
        self.latent_cell = (
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_len) if k > 0 else None
        )

    def _latents(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        h = self.backbone_model.backbone(input_ids)
        latents = [h]
        if self.latent_cell is not None:
            for _ in range(self.k):
                h = self.latent_cell(h)
                latents.append(h)
        return latents

    def forward(
        self, input_ids: torch.Tensor, return_latents: bool = False
    ):
        latents = self._latents(input_ids)
        h = latents[-1]
        logits = self.backbone_model.head(self.backbone_model.ln_f(h))
        if return_latents:
            return logits, latents
        return logits
