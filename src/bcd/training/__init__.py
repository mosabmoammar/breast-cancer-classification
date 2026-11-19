# ---------------------------------------------------------------------------
# Training layer: the single training loop shared by both models.
# ---------------------------------------------------------------------------
"""Shared gradient-descent training loop with early stopping."""

from bcd.training.trainer import TrainingHistory, predict_proba, train_model

__all__ = ["TrainingHistory", "predict_proba", "train_model"]
