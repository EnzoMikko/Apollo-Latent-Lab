"""Schéma des métriques d'un run + helpers de calcul (cf. README §12).

``RunMetrics`` regroupe tous les champs obligatoires d'un run. Les ratios
dérivés (accuracy_per_flop / accuracy_per_token) sont calculés à partir des
mesures brutes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunMetrics:
    # Identité
    experiment_id: str
    config_name: str
    model_type: str
    dataset: str
    seed: int
    latent_depth_k: int
    num_parameters: int

    # Performance
    train_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    test_accuracy: float = 0.0
    ood_accuracy: float = 0.0
    train_loss: float = 0.0
    validation_loss: float = 0.0

    # Coût
    latency_ms: float = 0.0
    estimated_flops: float = 0.0
    tokens_generated: float = 0.0
    accuracy_per_flop: float = 0.0
    accuracy_per_token: float = 0.0

    # Latent (optionnel, rempli par l'analyse)
    latent_state_norm_mean: Optional[float] = None
    latent_state_norm_std: Optional[float] = None
    latent_state_variance: Optional[float] = None
    linear_probe_accuracy: Optional[float] = None
    cluster_score: Optional[float] = None
    representation_stability_score: Optional[float] = None

    # Traçabilité
    checkpoint_path: str = ""
    timestamp: str = ""

    # Accuracy détaillée par longueur (diagnostic OOD)
    accuracy_by_length: dict[str, float] = field(default_factory=dict)

    def finalize_ratios(self) -> None:
        """Calcule les ratios dérivés à partir des mesures brutes."""
        if self.estimated_flops > 0:
            self.accuracy_per_flop = self.test_accuracy / self.estimated_flops
        if self.tokens_generated > 0:
            self.accuracy_per_token = self.test_accuracy / self.tokens_generated

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
