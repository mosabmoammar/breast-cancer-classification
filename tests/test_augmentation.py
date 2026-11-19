# ---------------------------------------------------------------------------
# Tests for SMOTE: it must only enlarge the training set, balance its
# classes, and leave validation and test untouched.
# ---------------------------------------------------------------------------
"""SMOTE changes only the training set."""

from __future__ import annotations

import numpy as np

from bcd.data.augmentation import apply_smote, class_counts
from bcd.data.loader import MALIGNANT, load_data
from bcd.data.preprocessing import scale_splits
from bcd.data.splitter import split_data


def test_smote_balances_training_set(data) -> None:
    """After SMOTE both classes have as many samples as the original majority."""
    before = class_counts(data.scaled.y_train)
    after = class_counts(data.y_train_smote)
    assert after["benign"] == before["benign"]
    assert after["malignant"] == before["benign"]  # 1:1 ratio
    # Only malignant samples are added, and the original rows are kept first.
    n = len(data.scaled.y_train)
    np.testing.assert_array_equal(data.X_train_smote[:n], data.scaled.X_train)
    assert np.all(data.y_train_smote[n:] == MALIGNANT)


def test_smote_does_not_touch_val_or_test(cfg) -> None:
    """Validation and test are byte-identical with and without SMOTE."""
    X, y, _ = load_data()
    scaled, _ = scale_splits(split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed))
    val_X, val_y = scaled.X_val.copy(), scaled.y_val.copy()
    test_X, test_y = scaled.X_test.copy(), scaled.y_test.copy()

    apply_smote(scaled.X_train, scaled.y_train, cfg.augmentation.k_neighbors,
                cfg.augmentation.sampling_strategy, cfg.seed)

    np.testing.assert_array_equal(scaled.X_val, val_X)
    np.testing.assert_array_equal(scaled.y_val, val_y)
    np.testing.assert_array_equal(scaled.X_test, test_X)
    np.testing.assert_array_equal(scaled.y_test, test_y)


def test_prepared_data_keeps_original_val_and_test(data) -> None:
    """The pipeline's val/test sets are the plain scaled splits (no synthetic rows)."""
    assert len(data.scaled.y_val) == 91
    assert len(data.scaled.y_test) == 114
    np.testing.assert_allclose(
        data.scaled.X_test, (data.raw.X_test - data.scaler.mean_) / data.scaler.scale_
    )


def test_smote_is_reproducible(data, cfg) -> None:
    """Same seed, same synthetic samples."""
    X_a, y_a = apply_smote(data.scaled.X_train, data.scaled.y_train,
                           cfg.augmentation.k_neighbors, cfg.augmentation.sampling_strategy, cfg.seed)
    np.testing.assert_array_equal(X_a, data.X_train_smote)
    np.testing.assert_array_equal(y_a, data.y_train_smote)
