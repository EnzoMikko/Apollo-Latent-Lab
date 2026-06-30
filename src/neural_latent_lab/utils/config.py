"""Chargement et validation des configs d'expérience (YAML -> dataclass typée).

Une config décrit entièrement une expérience reproductible : dataset, modèle,
profondeur latente ``k``, optimisation, seed, device et sortie.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "addition"
    train_digits: tuple[int, int] = (1, 4)
    ood_digits: tuple[int, int] = (5, 8)
    n_train: int = 20000
    n_val: int = 2000
    n_test: int = 2000
    n_ood: int = 2000
    reverse_output: bool = True  # résultat écrit LSB en premier (trick addition)


@dataclass
class ModelConfig:
    # "baseline" | "latent" | "compute_matched"
    kind: str = "latent"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64
    # nombre de cycles latents partagés (latent) ; ignoré pour baseline.
    k: int = 4
    # "nope" | "sinusoidal" | "learned" -- DOIT être identique entre modèles
    # comparés, sinon la généralisation OOD est biaisée.
    pos_encoding: str = "nope"


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_every: int = 1  # en epochs


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    device: str = "auto"
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Sous-configs imbriquées, résolues par nom de champ (les annotations sont des
# chaînes à cause de ``from __future__ import annotations``).
_NESTED = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig}


def _coerce(dc_type, payload: dict[str, Any]):
    """Construit une dataclass à partir d'un dict, en coercant les sous-dataclasses
    et en convertissant les listes YAML en tuples quand le champ est un tuple."""
    if payload is None:
        return dc_type()
    kwargs: dict[str, Any] = {}
    valid = {f.name for f in dataclasses.fields(dc_type)}
    for key, value in payload.items():
        if key not in valid:
            raise ValueError(f"Champ inconnu '{key}' pour {dc_type.__name__}")
        if key in _NESTED and isinstance(value, dict):
            kwargs[key] = _coerce(_NESTED[key], value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return dc_type(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Charge une config YAML et la valide via la dataclass ``ExperimentConfig``."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _coerce(ExperimentConfig, raw)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Sérialise une config en dict (pour le logging / les checkpoints)."""
    return dataclasses.asdict(cfg)
