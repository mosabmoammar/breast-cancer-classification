# ---------------------------------------------------------------------------
# Step 4 of the pipeline (Stage 2 only): rebalance the training split with
# SMOTE (Synthetic Minority Over-sampling Technique).
#
# Input: the SCALED training split. Output: a larger training set in which
# malignant and benign are equally frequent. Validation and test are never
# passed to this module.
# ---------------------------------------------------------------------------
"""SMOTE oversampling of the minority (malignant) class."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE

from bcd.data.loader import CLASS_NAMES


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int,
    sampling_strategy: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic malignant samples until the classes are balanced.

    Args:
        X_train: Scaled training features.
        y_train: Training labels (1 = malignant, the minority class).
        k_neighbors: Number of malignant neighbours SMOTE interpolates towards.
        sampling_strategy: Desired minority/majority ratio after resampling
            (1.0 = as many malignant as benign samples).
        seed: random_state, so the synthetic samples are identical on every run.

    Returns:
        X_resampled, y_resampled: The original training samples followed by
        the synthetic malignant samples.

    Notes:
        NOTE: How SMOTE works: for a real malignant sample it picks one of
        its k nearest malignant neighbours and creates a new point at a
        random position on the line segment between the two. The new
        samples are plausible "in-between" tumours rather than exact copies,
        so the model sees a denser malignant region instead of memorising
        duplicated points (the risk of plain random oversampling).

        NOTE: Why AFTER the split and on the training set ONLY: synthetic
        points are built from real ones. If SMOTE ran before the split, a
        synthetic training point could be interpolated from a test patient,
        leaking test information into training. Validation and test must
        also keep the real class mix, otherwise early stopping and the
        final metrics would describe an artificial population.

        NOTE: Why AFTER scaling: SMOTE uses Euclidean distances to find
        neighbours. On unscaled data the distance is dominated by large
        features such as "area", so the neighbours would be chosen almost
        on one feature alone.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return np.asarray(X_res), np.asarray(y_res)


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Count the samples of each class.

    Args:
        y: Labels (1 = malignant, 0 = benign).

    Returns:
        Dictionary {"benign": n0, "malignant": n1}.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(CLASS_NAMES))
    return {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)}
