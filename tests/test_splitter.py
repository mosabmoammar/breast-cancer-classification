# ---------------------------------------------------------------------------
# Tests for the train / validation / test split: stratified, disjoint and
# identical on every run.
# ---------------------------------------------------------------------------
"""The split is stratified and reproducible."""

from __future__ import annotations

import numpy as np

from bcd.data.loader import load_data
from bcd.data.splitter import split_data


def test_split_sizes_and_disjoint(cfg) -> None:
    """Sizes match the configured ratios and no sample appears twice."""
    X, y, _ = load_data()
    s = split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed)
    assert (len(s.y_train), len(s.y_val), len(s.y_test)) == (364, 91, 114)
    # Every row is a distinct patient, so rows must not be shared across splits.
    rows = [tuple(r) for r in np.vstack([s.X_train, s.X_val, s.X_test])]
    assert len(set(rows)) == len(X)


def test_split_is_stratified(cfg) -> None:
    """Each subset keeps the overall malignant share (within one sample)."""
    X, y, _ = load_data()
    s = split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed)
    overall = y.mean()
    for y_sub in (s.y_train, s.y_val, s.y_test):
        assert abs(y_sub.mean() - overall) <= 1 / len(y_sub)


def test_split_is_reproducible(cfg) -> None:
    """Two calls with the same seed give exactly the same subsets."""
    X, y, _ = load_data()
    a = split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed)
    b = split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed)
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
