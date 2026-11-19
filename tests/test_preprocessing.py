# ---------------------------------------------------------------------------
# Tests for feature scaling: the scaler must learn from training data only.
# ---------------------------------------------------------------------------
"""The scaler is fitted on the training split only."""

from __future__ import annotations

import numpy as np


def test_scaler_statistics_come_from_train_only(data) -> None:
    """The scaler's mean/std equal the TRAINING split's statistics."""
    np.testing.assert_allclose(data.scaler.mean_, data.raw.X_train.mean(axis=0))
    np.testing.assert_allclose(data.scaler.scale_, data.raw.X_train.std(axis=0))
    # And they differ from statistics computed on all the data (no leakage).
    X_all = np.vstack([data.raw.X_train, data.raw.X_val, data.raw.X_test])
    assert not np.allclose(data.scaler.mean_, X_all.mean(axis=0))


def test_scaled_train_is_standardised_but_test_is_not_exactly(data) -> None:
    """Train is exactly zero-mean / unit-std; test is only approximately so."""
    np.testing.assert_allclose(data.scaled.X_train.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(data.scaled.X_train.std(axis=0), 1, atol=1e-10)
    # Test was transformed with TRAIN statistics, so its mean is not exactly 0.
    assert not np.allclose(data.scaled.X_test.mean(axis=0), 0, atol=1e-10)


def test_val_and_test_use_train_statistics(data) -> None:
    """Validation and test are transformed with the training mean/std."""
    expected_val = (data.raw.X_val - data.scaler.mean_) / data.scaler.scale_
    expected_test = (data.raw.X_test - data.scaler.mean_) / data.scaler.scale_
    np.testing.assert_allclose(data.scaled.X_val, expected_val)
    np.testing.assert_allclose(data.scaled.X_test, expected_test)
