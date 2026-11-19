# ---------------------------------------------------------------------------
# Classification metrics, always computed with malignant (label 1) as the
# positive class. Used for the train and test splits of every model/stage.
# ---------------------------------------------------------------------------
"""Metrics for a binary classifier where 1 = malignant."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bcd.data.loader import BENIGN, CLASS_NAMES, MALIGNANT

# Order in which scalar metrics appear in every table and chart.
METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


def predict_labels(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Turn probabilities into hard labels.

    Args:
        y_prob: P(malignant) per sample.
        threshold: Predict malignant when P(malignant) >= threshold.

    Returns:
        Labels (1 = malignant, 0 = benign).

    Notes:
        NOTE: The threshold is fixed at 0.5 for both models and both stages.
        Tuning it would change the precision/recall trade-off on its own and
        hide the effect of the model or of SMOTE, so the comparison would
        no longer be fair.
    """
    return (y_prob >= threshold).astype(np.int64)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute every metric the project reports.

    Args:
        y_true: True labels (1 = malignant).
        y_prob: Predicted P(malignant).
        threshold: Decision threshold.

    Returns:
        Dictionary with:
            accuracy, precision, recall, f1, roc_auc (floats);
            tn, fp, fn, tp (ints);
            confusion_matrix (2x2 array, rows = true, cols = predicted,
                order [benign, malignant]);
            report (scikit-learn classification report, text).

    Notes:
        What each metric means for cancer diagnosis (positive = malignant):
        - accuracy: share of all patients classified correctly. Misleading
          on imbalanced data: always predicting "benign" already scores ~63%.
        - recall (sensitivity) = TP / (TP + FN): share of real cancers that
          are detected. NOTE: our MAIN metric, because a false negative is a
          missed cancer, the most dangerous error.
        - precision = TP / (TP + FP): share of "malignant" alarms that are
          real cancers. Shows the cost in false alarms (extra biopsies, anxiety).
        - F1: harmonic mean of precision and recall; high only if both are high.
        - ROC-AUC: probability that a random malignant case gets a higher
          score than a random benign case. It uses the probabilities, not
          the 0.5 cut, so it measures ranking quality independent of the threshold.
    """
    y_pred = predict_labels(y_prob, threshold)
    # pos_label / labels are passed explicitly so that malignant is the
    # positive class no matter what scikit-learn's defaults are.
    cm = confusion_matrix(y_true, y_pred, labels=[BENIGN, MALIGNANT])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=MALIGNANT, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": cm,
        "report": classification_report(
            y_true,
            y_pred,
            labels=[BENIGN, MALIGNANT],
            target_names=list(CLASS_NAMES),
            digits=4,
            zero_division=0,
        ),
    }
