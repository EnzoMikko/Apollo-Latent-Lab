import torch

from neural_latent_lab.models.builder import build_model, count_parameters
from neural_latent_lab.utils.config import ModelConfig


def _cfg(kind, k=0):
    return ModelConfig(
        kind=kind, d_model=32, n_layers=2, n_heads=4, d_ff=64,
        dropout=0.0, max_len=16, k=k, pos_encoding="nope",
    )


def test_forward_shapes():
    vocab_size = 14
    x = torch.randint(0, vocab_size, (4, 12))
    for model in (
        build_model(_cfg("baseline"), vocab_size),
        build_model(_cfg("latent", k=4), vocab_size),
        build_model(_cfg("compute_matched", k=4), vocab_size),
    ):
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        assert logits.shape == (4, 12, vocab_size)


def test_latent_k0_equals_backbone():
    """latent(k=0) ne doit appliquer aucune cellule -> sortie = backbone+head."""
    vocab_size = 14
    model = build_model(_cfg("latent", k=0), vocab_size)
    assert model.latent_cell is None
    model.eval()
    x = torch.randint(0, vocab_size, (2, 10))
    logits = model(x)
    h = model.backbone_model.backbone(x)
    expected = model.backbone_model.head(model.backbone_model.ln_f(h))
    assert torch.allclose(logits, expected, atol=1e-6)


def test_latent_returns_k_plus_one_states():
    model = build_model(_cfg("latent", k=4), 14)
    model.eval()
    x = torch.randint(0, 14, (2, 10))
    _, latents = model(x, return_latents=True)
    assert len(latents) == 5  # h0 + 4 cycles


def test_compute_matched_has_more_params_than_latent():
    """À k égal, le compute-matched (couches distinctes) a plus de params."""
    latent = build_model(_cfg("latent", k=4), 14)
    matched = build_model(_cfg("compute_matched", k=4), 14)
    assert count_parameters(matched) > count_parameters(latent)
