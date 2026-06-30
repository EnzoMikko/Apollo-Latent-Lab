import torch

from neural_latent_lab.models.builder import build_model
from neural_latent_lab.utils.config import ModelConfig
from neural_latent_lab.utils.seed import seed_everything


def _build():
    cfg = ModelConfig(
        kind="latent", d_model=32, n_layers=2, n_heads=4, d_ff=64,
        dropout=0.0, max_len=16, k=2, pos_encoding="nope",
    )
    return build_model(cfg, vocab_size=14)


def test_same_seed_same_init():
    seed_everything(123)
    m1 = _build()
    seed_everything(123)
    m2 = _build()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_same_seed_same_forward():
    seed_everything(7)
    m1 = _build()
    seed_everything(7)
    m2 = _build()
    x = torch.randint(0, 14, (3, 12))
    m1.eval()
    m2.eval()
    assert torch.allclose(m1(x), m2(x), atol=1e-7)
