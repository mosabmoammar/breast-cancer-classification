# ---------------------------------------------------------------------------
# Artificial Neural Network: a small Multilayer Perceptron (MLP).
#
# Architecture (from config/config.yaml): 30 -> 16 -> 8 -> 1
#   input (30 features) -> Linear -> ReLU -> Dropout
#                       -> Linear -> ReLU -> Dropout
#                       -> Linear -> one logit
# ---------------------------------------------------------------------------
"""Multilayer Perceptron classifier."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ANN(nn.Module):
    """Small MLP with ReLU hidden layers, dropout and a single output logit.

    Args:
        n_features: Number of input features (30).
        hidden_sizes: Width of each hidden layer, e.g. (16, 8).
        dropout: Probability of zeroing a hidden activation during training.

    Notes:
        NOTE: Why this size: with 30 -> 16 -> 8 -> 1 the network has
        30*16+16 + 16*8+8 + 8*1+1 = 641 trainable parameters, against 364
        training samples. A wider or deeper network would have far more
        parameters than samples and could simply memorise the training set.
        Two hidden layers are enough to learn non-linear combinations of
        the features (which logistic regression cannot), and the
        decreasing widths (16 then 8) progressively compress the 30 inputs
        towards a single decision.

        NOTE: Why ReLU: ReLU(z) = max(0, z) has gradient 1 for every
        positive input. The sigmoid's gradient is at most 0.25 and close to
        0 for large |z|, so stacking sigmoid layers multiplies small numbers
        during backpropagation and the gradient reaching the first layer
        vanishes. ReLU avoids this and is also cheap to compute.

        NOTE: Why dropout 0.2: during training, each hidden activation is
        set to 0 with probability 0.2 (and the rest are scaled up by
        1/0.8). The network cannot rely on any single neuron, which acts as
        a regulariser against overfitting. "Light" (0.2 rather than 0.5)
        because the network is already small; heavy dropout on 16 and 8
        units would cause underfitting. Dropout is automatically disabled
        in model.eval() mode, so evaluation is deterministic.

        NOTE: Single output unit: for a binary problem one logit is enough;
        sigmoid(logit) is P(malignant) and P(benign) = 1 - P(malignant).
    """

    def __init__(self, n_features: int, hidden_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = n_features
        for width in hidden_sizes:
            layers += [nn.Linear(in_size, width), nn.ReLU(), nn.Dropout(dropout)]
            in_size = width
        # Output layer: no activation. It returns a raw logit; the sigmoid is
        # applied inside the loss during training and explicitly at prediction.
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per sample, shape (batch,)."""
        return self.network(x).squeeze(-1)
