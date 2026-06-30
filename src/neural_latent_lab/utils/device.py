"""Sélection du device (CPU/CUDA)."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Retourne le device à utiliser.

    ``prefer`` peut valoir ``"auto"``, ``"cpu"`` ou ``"cuda"``. En mode auto on
    prend le GPU s'il est disponible, sinon le CPU.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
