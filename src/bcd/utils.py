# ---------------------------------------------------------------------------
# Reproducibility helpers.
#
# Called by the pipeline before EVERY model is trained, so each of the four
# runs (2 models x 2 stages) starts from the same random state and the
# results do not depend on the order in which the runs happen.
# ---------------------------------------------------------------------------
"""Seeding and determinism."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator the project uses.

    Args:
        seed: The seed for `random`, numpy and torch.
        deterministic: If True, ask PyTorch to use only deterministic
            algorithms, so two runs give bit-identical results.

    Notes:
        Weight initialisation, dropout masks and DataLoader shuffling all
        draw from torch's generator; seeding it makes them repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        # Required by some CUDA kernels for determinism; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True: if an operation has no deterministic version,
        # PyTorch warns instead of crashing ("where possible").
        torch.use_deterministic_algorithms(True, warn_only=True)
