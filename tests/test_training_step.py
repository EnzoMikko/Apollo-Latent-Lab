import torch
from torch.utils.data import DataLoader

from neural_latent_lab.data.addition import make_addition_splits
from neural_latent_lab.models.builder import build_model
from neural_latent_lab.training.evaluate import compute_loss
from neural_latent_lab.utils.config import ModelConfig
from neural_latent_lab.utils.seed import seed_everything


def test_single_training_step_reduces_loss():
    seed_everything(0)
    device = torch.device("cpu")
    vocab, splits = make_addition_splits(
        train_digits=(1, 2), ood_digits=(3, 4),
        n_train=128, n_val=8, n_test=8, n_ood=8,
        reverse_output=True, max_len=24, seed=0,
    )
    cfg = ModelConfig(
        kind="latent", d_model=32, n_layers=2, n_heads=4, d_ff=64,
        dropout=0.0, max_len=24, k=2, pos_encoding="nope",
    )
    model = build_model(cfg, vocab_size=len(vocab)).to(device)
    loader = DataLoader(splits["train"], batch_size=64, shuffle=False)
    batch = next(iter(loader))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    loss0 = compute_loss(model, batch, device).item()
    for _ in range(20):
        opt.zero_grad()
        loss = compute_loss(model, batch, device)
        loss.backward()
        opt.step()
    loss1 = compute_loss(model, batch, device).item()
    assert loss1 < loss0
