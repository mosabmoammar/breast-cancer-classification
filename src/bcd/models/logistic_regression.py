# ---------------------------------------------------------------------------
# Baseline model: logistic regression written as a PyTorch nn.Module.
#
# It is trained by exactly the same loop, loss and optimiser as the ANN, so
# any difference in results comes from the model itself (linear vs
# non-linear), not from a different training procedure.
# ---------------------------------------------------------------------------
"""Logistic regression as a single linear layer."""

from __future__ import annotations

import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    """Logistic regression: logit = w . x + b, P(malignant) = sigmoid(logit).

    Args:
        n_features: Number of input features (30).

    Notes:
        NOTE: Logistic regression is a single nn.Linear(30, 1) layer: one
        weight per feature plus a bias, with no hidden layer. The forward
        pass returns the raw logit (w . x + b). The sigmoid that turns it
        into a probability is NOT applied here: the loss
        (BCEWithLogitsLoss) applies it internally in a numerically stable
        way, and we apply it explicitly only when we need probabilities
        for metrics. Because the model is linear in x, its decision
        boundary (logit = 0, i.e. probability = 0.5) is a hyperplane.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sample, shape (batch,)."""
        # squeeze(-1) turns shape (batch, 1) into (batch,) to match the labels.
        return self.linear(x).squeeze(-1)
