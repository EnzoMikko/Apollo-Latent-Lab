"""Extraction et analyse des états latents d'un modèle entraîné.

Pipeline (cf. README §E) :
1. Recharger le checkpoint + reconstruire données/modèle.
2. Extraire l'état latent final à la position ``=`` (juste avant de générer la
   réponse) pour chaque exemple du split test, avec ses métadonnées.
3. Dump ``.npz`` dans ``outputs/latent_dumps/``.
4. Probe linéaire : une variable de tâche (présence de retenue) est-elle décodable ?
5. Score de clustering (silhouette) + figure PCA colorée par métadonnée.

Un latent qui encode une structure utile -> probe au-dessus du hasard + clusters
cohérents. C'est le test de H4.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from neural_latent_lab.data.addition import make_addition_splits
from neural_latent_lab.logging.experiment_logger import ExperimentLogger
from neural_latent_lab.models.builder import build_model
from neural_latent_lab.utils.config import _coerce, ExperimentConfig
from neural_latent_lab.utils.device import get_device


@torch.no_grad()
def _extract_latents(model, dataset, vocab, device):
    """Retourne (latents [N, d], has_carry [N], n_digits [N]) à la position '='."""
    model.eval()
    feats, carries, digits = [], [], []
    batch_size = 256
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        prompt_len = torch.stack([it["prompt_len"] for it in items])
        out = model(input_ids, return_latents=True) if _supports_latents(model) else None
        if out is not None:
            _, latents = out
            h = latents[-1]  # état latent final
        else:
            h = model.backbone(input_ids) if hasattr(model, "backbone") else model.backbone_model.backbone(input_ids)
        # Position '=' = dernier token du prompt (index prompt_len - 1).
        idx = (prompt_len - 1).clamp(min=0)
        gathered = h[torch.arange(h.size(0)), idx].cpu().numpy()
        feats.append(gathered)
        carries.append(torch.stack([it["has_carry"] for it in items]).numpy())
        digits.append(torch.stack([it["n_digits"] for it in items]).numpy())
    return np.concatenate(feats), np.concatenate(carries), np.concatenate(digits)


def _supports_latents(model) -> bool:
    return hasattr(model, "latent_cell")


def _linear_probe(feats: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy d'un classifieur linéaire (sur un split held-out des latents)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    x_tr, x_te, y_tr, y_te = train_test_split(
        feats, labels, test_size=0.3, random_state=0, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))


def analyze_checkpoint(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = _coerce(ExperimentConfig, ckpt["config"])
    experiment_id = ckpt["experiment_id"]
    device = get_device(cfg.device)
    logger = ExperimentLogger(cfg.output_dir, experiment_id)
    logger.info(f"Analyse des latents pour {experiment_id}")

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
    model = build_model(cfg.model, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])

    feats, carries, digits = _extract_latents(model, splits["test"], vocab, device)

    norms = np.linalg.norm(feats, axis=1)
    probe_acc = _linear_probe(feats, carries)
    try:
        cluster = float(silhouette_score(feats, carries)) if len(np.unique(carries)) > 1 else float("nan")
    except Exception:
        cluster = float("nan")

    np.savez(
        logger.latent_dump_path,
        latents=feats,
        has_carry=carries,
        n_digits=digits,
    )
    logger.info(f"Latents dumpés : {logger.latent_dump_path}")

    # Figure PCA colorée par présence de retenue.
    pca = PCA(n_components=2).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pca[:, 0], pca[:, 1], c=carries, cmap="coolwarm", s=8, alpha=0.6)
    ax.set_title(f"PCA latents — {experiment_id}\ncouleur = retenue")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="has_carry")
    fig.tight_layout()
    fig.savefig(logger.figure_path("latent_pca.png"), dpi=120)
    plt.close(fig)

    result = {
        "experiment_id": experiment_id,
        "latent_state_norm_mean": float(norms.mean()),
        "latent_state_norm_std": float(norms.std()),
        "latent_state_variance": float(feats.var()),
        "linear_probe_accuracy": probe_acc,
        "cluster_score": cluster,
    }
    out_path = Path(cfg.output_dir) / "metrics" / f"{experiment_id}_analysis.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Analyse latente : probe_carry={probe_acc:.3f} silhouette={cluster:.3f}")
    return result
