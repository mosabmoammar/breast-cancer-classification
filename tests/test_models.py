# ---------------------------------------------------------------------------
# Tests for the model contract: one output per sample and probabilities in
# [0, 1], for both models. Also a quick smoke test of the training loop.
# ---------------------------------------------------------------------------
"""Both models honour the same input/output contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bcd.data.dataset import make_loader
from bcd.pipeline.experiment import MODELS, build_model
from bcd.training.trainer import predict_proba, train_model
from bcd.utils import set_seed


@pytest.mark.parametrize("model_name", MODELS)
def test_one_output_per_sample(model_name, cfg) -> None:
    """A batch of 7 samples gives 7 logits."""
    set_seed(cfg.seed)
    model, _ = build_model(model_name, cfg, n_features=30)
    out = model(torch.randn(7, 30))
    assert out.shape == (7,)


@pytest.mark.parametrize("model_name", MODELS)
def test_probabilities_between_zero_and_one(model_name, cfg, data) -> None:
    """predict_proba returns one probability per test sample, all in [0, 1]."""
    set_seed(cfg.seed)
    model, _ = build_model(model_name, cfg, n_features=30)
    prob = predict_proba(model, data.scaled.X_test)
    assert prob.shape == (len(data.scaled.y_test),)
    assert np.all((prob >= 0) & (prob <= 1))


def test_logistic_regression_is_single_linear_layer(cfg) -> None:
    """Logistic regression has exactly 30 weights + 1 bias."""
    model, wd = build_model("Logistic Regression", cfg, n_features=30)
    assert sum(p.numel() for p in model.parameters()) == 31
    assert wd == 0.0


def test_training_reduces_loss_and_restores_best(cfg, data) -> None:
    """A short run lowers the training loss and keeps the best-validation weights."""
    set_seed(cfg.seed)
    model, wd = build_model("ANN", cfg, n_features=30)
    tr = make_loader(data.scaled.X_train, data.scaled.y_train, 32, shuffle=True, seed=cfg.seed)
    va = make_loader(data.scaled.X_val, data.scaled.y_val, 32, shuffle=False, seed=cfg.seed)
    h = train_model(model, tr, va, learning_rate=1e-3, weight_decay=wd, max_epochs=20, patience=5)
    assert h.train_loss[-1] < h.train_loss[0]
    assert h.val_loss[h.best_epoch - 1] == min(h.val_loss)
