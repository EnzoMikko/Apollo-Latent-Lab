"""Estimation analytique des FLOPs d'un Transformer (forward).

On utilise une formule par couche, indépendante de l'implémentation, afin de
pouvoir comparer équitablement des modèles de profondeurs effectives différentes :
- baseline        : profondeur effective = n_layers
- latent (k)      : profondeur effective = n_layers + k  (cellule partagée ×k)
- compute_matched : profondeur effective = n_layers + k  (couches distinctes)

Les FLOPs d'une couche Transformer (forward) pour une séquence de longueur ``L``
et une dimension ``d`` sont dominés par :
  - projections QKV + sortie : 4 * L * d^2
  - scores d'attention + agrégation : 2 * L^2 * d
  - MLP (deux couches, largeur d_ff) : 2 * L * d * d_ff
On compte une multiply-add comme 2 FLOPs.
"""

from __future__ import annotations


def transformer_layer_flops(seq_len: int, d_model: int, d_ff: int) -> float:
    attn_proj = 4 * seq_len * d_model * d_model
    attn_scores = 2 * seq_len * seq_len * d_model
    mlp = 2 * seq_len * d_model * d_ff
    return 2.0 * (attn_proj + attn_scores + mlp)


def estimate_forward_flops(
    *,
    seq_len: int,
    d_model: int,
    d_ff: int,
    n_layers: int,
    k: int = 0,
    vocab_size: int | None = None,
) -> float:
    """FLOPs d'un forward sur une séquence (profondeur effective = n_layers + k)."""
    effective_depth = n_layers + max(k, 0)
    total = effective_depth * transformer_layer_flops(seq_len, d_model, d_ff)
    if vocab_size is not None:
        total += 2.0 * seq_len * d_model * vocab_size  # LM head
    return total
