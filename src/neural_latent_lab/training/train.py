"""Boucle d'entraînement complète et reproductible pour une config donnée.

Gère : seed -> dataset -> modèle -> optimisation (AdamW + warmup) -> validation
-> checkpoint -> métriques JSON. Retourne le chemin du fichier de métriques.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from neural_latent_lab.data.addition import make_addition_splits
from neural_latent_lab.logging.experiment_logger import ExperimentLogger
from neural_latent_lab.models.builder import build_model, count_parameters
from neural_latent_lab.training.evaluate import (
    compute_loss,
    evaluate_loss,
    exact_match_accuracy,
)
from neural_latent_lab.training.metrics import RunMetrics
from neural_latent_lab.utils.config import ExperimentConfig, config_to_dict
from neural_latent_lab.utils.device import get_device
from neural_latent_lab.utils.flops import estimate_forward_flops
from neural_latent_lab.utils.seed import seed_everything


def _effective_k(cfg: ExperimentConfig) -> int:
    """Profondeur supplémentaire pour le calcul des FLOPs."""
    if cfg.model.kind in ("latent", "compute_matched"):
        return cfg.model.k
    return 0


def _make_experiment_id(cfg: ExperimentConfig, timestamp: str) -> str:
    return f"{cfg.name}_seed{cfg.seed}_{timestamp}"


def train(cfg: ExperimentConfig) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = _make_experiment_id(cfg, timestamp)
    logger = ExperimentLogger(cfg.output_dir, experiment_id)
    logger.info(f"=== {experiment_id} | kind={cfg.model.kind} k={cfg.model.k} ===")

    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    vocab, splits = make_addition_splits(
        train_digits=cfg.data.train_digits,
        ood_digits=cfg.data.ood_digits,
        n_train=cfg.data.n_train,
        n_val=cfg.data.n_val,
        n_test=cfg.data.n_test,
        n_ood=cfg.data.n_ood,
        reverse_output=cfg.data.reverse_output,
        max_len=cfg.model.max_len,
        seed=cfg.seed,
    )
    train_loader = DataLoader(splits["train"], batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=cfg.train.batch_size)

    model = build_model(cfg.model, vocab_size=len(vocab)).to(device)
    n_params = count_parameters(model)
    logger.info(f"Paramètres entraînables : {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.train.warmup_steps:
            return (step + 1) / max(cfg.train.warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, last_train_loss = 0, 0.0
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            last_train_loss = loss.item()
        if (epoch + 1) % cfg.train.eval_every == 0:
            val_loss = evaluate_loss(model, val_loader, device)
            logger.info(
                f"epoch {epoch + 1}/{cfg.train.epochs} "
                f"train_loss={last_train_loss:.4f} val_loss={val_loss:.4f}"
            )

    # --- Évaluation finale ---
    logger.info("Évaluation finale (exact-match)...")
    train_acc, _, _ = exact_match_accuracy(model, splits["train"], vocab, device)
    val_acc, _, _ = exact_match_accuracy(model, splits["val"], vocab, device)
    test_acc, test_by_len, test_tokens = exact_match_accuracy(
        model, splits["test"], vocab, device
    )
    ood_acc, ood_by_len, _ = exact_match_accuracy(model, splits["ood"], vocab, device)
    val_loss = evaluate_loss(model, val_loader, device)

    # --- Coût : latence (forward) + FLOPs analytiques ---
    model.eval()
    sample = next(iter(val_loader))["input_ids"][:32].to(device)
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(sample)
        t0 = time.perf_counter()
        for _ in range(5):
            model(sample)
        latency_ms = (time.perf_counter() - t0) / 5 / sample.size(0) * 1000.0

    flops = estimate_forward_flops(
        seq_len=cfg.model.max_len,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        n_layers=cfg.model.n_layers,
        k=_effective_k(cfg),
        vocab_size=len(vocab),
    )

    accuracy_by_length = {f"test_{d}": v for d, v in test_by_len.items()}
    accuracy_by_length.update({f"ood_{d}": v for d, v in ood_by_len.items()})

    metrics = RunMetrics(
        experiment_id=experiment_id,
        config_name=cfg.name,
        model_type=cfg.model.kind,
        dataset=cfg.data.name,
        seed=cfg.seed,
        latent_depth_k=_effective_k(cfg),
        num_parameters=n_params,
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=test_acc,
        ood_accuracy=ood_acc,
        train_loss=last_train_loss,
        validation_loss=val_loss,
        latency_ms=latency_ms,
        estimated_flops=flops,
        tokens_generated=test_tokens,
        checkpoint_path=str(logger.checkpoint_path),
        timestamp=timestamp,
        accuracy_by_length=accuracy_by_length,
    )
    metrics.finalize_ratios()

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_to_dict(cfg),
            "metrics": metrics.to_dict(),
            "seed": cfg.seed,
            "experiment_id": experiment_id,
        },
        logger.checkpoint_path,
    )
    logger.info(f"Checkpoint : {logger.checkpoint_path}")
    logger.info(
        f"RESULT test_acc={test_acc:.3f} ood_acc={ood_acc:.3f} params={n_params:,} "
        f"flops={flops:.2e}"
    )
    metrics_path = logger.save_metrics(metrics, config_to_dict(cfg))
    return str(metrics_path)
