# ---------------------------------------------------------------------------
# Step 5 of the pipeline: wrap numpy arrays into a PyTorch Dataset and build
# DataLoaders that feed mini-batches to the shared training loop.
# ---------------------------------------------------------------------------
"""PyTorch Dataset and DataLoader factory."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BreastCancerDataset(Dataset):
    """Tabular dataset of (features, label) pairs as float32 tensors.

    Args:
        X: Feature matrix, shape (n_samples, n_features), already scaled.
        y: Labels, shape (n_samples,), 1 = malignant.

    Notes:
        Labels are stored as float32 because BCEWithLogitsLoss compares
        a real-valued logit with a real-valued target in {0.0, 1.0}.
        The whole dataset (a few hundred rows) is converted to tensors once,
        up front, so __getitem__ is just an index lookup.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (features, label) pair."""
        return self.X[idx], self.y[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a DataLoader over (X, y).

    Args:
        X: Features.
        y: Labels.
        batch_size: Mini-batch size.
        shuffle: True for the training set only.
        seed: Seed of the generator that drives the shuffling order.

    Returns:
        A DataLoader yielding (features, labels) mini-batches.

    Notes:
        NOTE: Only the training loader is shuffled. Shuffling every epoch
        gives the optimiser differently composed mini-batches, which makes
        the gradient estimates less correlated from one step to the next.
        Validation and test are only evaluated, so their order is
        irrelevant and is kept fixed.

        A dedicated, seeded torch.Generator makes the shuffling order
        reproducible, and num_workers=0 avoids extra worker processes
        (the data is tiny, and workers would add non-determinism on Windows).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        BreastCancerDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )
