"""Logger d'expérience : crée l'arborescence ``outputs/`` et écrit logs + métriques.

Chaque run est traçable via son ``experiment_id`` :
- ``outputs/logs/{id}.log``       : journal texte
- ``outputs/metrics/{id}.json``   : métriques + config (reproductibilité)
- ``outputs/checkpoints/{id}.pt`` : poids (chemin retourné, écrit par train)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_latent_lab.training.metrics import RunMetrics


class ExperimentLogger:
    def __init__(self, output_dir: str | Path, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.root = Path(output_dir)
        self.dirs = {
            name: self.root / name
            for name in ("logs", "metrics", "checkpoints", "figures", "latent_dumps")
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"nll.{experiment_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh = logging.FileHandler(self.dirs["logs"] / f"{experiment_id}.log")
            fh.setFormatter(fmt)
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    @property
    def checkpoint_path(self) -> Path:
        return self.dirs["checkpoints"] / f"{self.experiment_id}.pt"

    @property
    def latent_dump_path(self) -> Path:
        return self.dirs["latent_dumps"] / f"{self.experiment_id}_latents.npz"

    def figure_path(self, name: str) -> Path:
        return self.dirs["figures"] / name

    def save_metrics(self, metrics: RunMetrics, config: dict[str, Any]) -> Path:
        path = self.dirs["metrics"] / f"{self.experiment_id}.json"
        payload = {"metrics": metrics.to_dict(), "config": config}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.info(f"Métriques écrites dans {path}")
        return path
