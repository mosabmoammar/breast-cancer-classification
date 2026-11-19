# ---------------------------------------------------------------------------
# Step 1 of the pipeline: load the Breast Cancer Wisconsin (Diagnostic)
# dataset from scikit-learn and remap the labels so that 1 = malignant.
#
# This is the ONLY place in the project where labels are touched. Every
# later module (splits, SMOTE, metrics, plots) relies on 1 = malignant.
# ---------------------------------------------------------------------------
"""Load the dataset with malignant as the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

# Label constants used everywhere else, so no module hard-codes 0 or 1.
BENIGN: int = 0
MALIGNANT: int = 1
# Index i of this tuple is the name of label i (used in reports and plots).
CLASS_NAMES: tuple[str, str] = ("benign", "malignant")


def remap_labels(y_sklearn: np.ndarray) -> np.ndarray:
    """Convert scikit-learn's encoding (0 = malignant) to ours (1 = malignant).

    Args:
        y_sklearn: Labels as shipped by scikit-learn: 0 = malignant, 1 = benign.

    Returns:
        Labels with 1 = malignant, 0 = benign (dtype int64).

    Notes:
        NOTE: We remap because metric functions such as precision, recall
        and F1 treat label 1 as the "positive" class by default. The class
        we care about detecting is malignant: a missed cancer (false
        negative) is the costly error. With 1 = malignant, "recall" directly
        means "share of cancers we caught", and a false negative directly
        means "a cancer we missed". Keeping scikit-learn's encoding would
        silently report everything from the benign point of view.
    """
    return (1 - y_sklearn).astype(np.int64)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the 569 x 30 feature matrix and the remapped labels.

    Returns:
        X: Feature matrix, shape (569, 30), float64.
        y: Labels, shape (569,), 1 = malignant, 0 = benign.
        feature_names: The 30 feature names, in column order.
    """
    bunch = load_breast_cancer()
    X = bunch.data.astype(np.float64)
    y = remap_labels(bunch.target)
    return X, y, list(bunch.feature_names)
