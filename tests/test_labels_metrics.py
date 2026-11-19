# ---------------------------------------------------------------------------
# Tests for the label convention: after remapping, 1 = malignant, and the
# metrics treat malignant as the positive class.
# ---------------------------------------------------------------------------
"""Label remapping and positive-class metrics."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

from bcd.data.loader import BENIGN, CLASS_NAMES, MALIGNANT, load_data
from bcd.evaluation.metrics import compute_metrics


def test_labels_are_remapped_to_malignant_one() -> None:
    """Every sample scikit-learn calls 'malignant' now has label 1."""
    bunch = load_breast_cancer()
    _, y, _ = load_data()
    sk_malignant = bunch.target == list(bunch.target_names).index("malignant")
    assert np.all(y[sk_malignant] == MALIGNANT)
    assert np.all(y[~sk_malignant] == BENIGN)
    assert CLASS_NAMES[MALIGNANT] == "malignant"
    # The dataset has 212 malignant and 357 benign tumours.
    assert (y == MALIGNANT).sum() == 212 and (y == BENIGN).sum() == 357


def test_metrics_use_malignant_as_positive() -> None:
    """Recall/precision are computed for label 1 (malignant)."""
    # 4 malignant, 6 benign. Prediction misses 1 malignant (FN) and raises
    # 2 false alarms on benign cases (FP).
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.55, 0.1, 0.1, 0.1, 0.1])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 2, 4)
    assert m["recall"] == 3 / 4        # malignant recall, not benign recall (4/6)
    assert m["precision"] == 3 / 5     # malignant precision
    # Confusion matrix layout: rows = true [benign, malignant].
    np.testing.assert_array_equal(m["confusion_matrix"], [[4, 2], [1, 3]])


def test_threshold_is_inclusive() -> None:
    """A probability exactly equal to the threshold is predicted malignant."""
    m = compute_metrics(np.array([1, 0]), np.array([0.5, 0.49]), threshold=0.5)
    assert m["tp"] == 1 and m["tn"] == 1
