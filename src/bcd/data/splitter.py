# ---------------------------------------------------------------------------
# Step 2 of the pipeline: split the data ONCE into train / validation / test.
#
# The same split (same seed) is reused by both models and both stages, so
# every comparison in the project is made on exactly the same patients.
# ---------------------------------------------------------------------------
"""Stratified train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    """The three disjoint subsets of the data.

    Attributes:
        X_train, y_train: Used to fit the scaler, SMOTE (Stage 2) and the model weights.
        X_val, y_val: Used only for early stopping and the learning curves.
        X_test, y_test: Used only once, for the final evaluation.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> DataSplits:
    """Split into stratified train / validation / test subsets.

    Args:
        X: Feature matrix.
        y: Labels (1 = malignant).
        test_size: Fraction of ALL samples held out as the test set.
        val_size: Fraction of the REMAINING (non-test) samples used for validation.
        seed: random_state for both splits, so the split is identical on every run.

    Returns:
        A DataSplits with the three subsets.

    Notes:
        NOTE: Both splits are stratified on y. The dataset is imbalanced
        (about 37% malignant), and with only 114 test and 91 validation
        samples a purely random split could leave noticeably fewer
        malignant cases in one subset by chance. Stratification keeps the
        malignant share the same in every subset, so validation loss and
        test metrics reflect the real class mix.

        The test split is taken FIRST, from the untouched data. Nothing
        that happens afterwards (scaling, SMOTE, early stopping) can see it,
        which prevents data leakage into the final evaluation.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, stratify=y_rest, random_state=seed
    )
    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)
