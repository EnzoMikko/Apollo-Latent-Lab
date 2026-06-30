"""Seeding global pour la reproductibilité."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Fixe les graines de Python, NumPy et PyTorch.

    Avec ``deterministic=True`` on active les flags cuDNN déterministes afin que
    deux runs avec la même seed produisent des poids d'initialisation et des
    pertes identiques (vérifié par ``tests/test_reproducibility.py``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
