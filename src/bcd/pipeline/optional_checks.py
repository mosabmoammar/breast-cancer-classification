# ---------------------------------------------------------------------------
# OPTIONAL checks, run only after the required experiment works. Their
# results are reported separately and never mixed into the main tables.
#
#   1. Sanity check: scikit-learn's LogisticRegression vs our PyTorch one.
#      If both reach similar test metrics, the PyTorch implementation and
#      training loop are behaving as expected.
#   2. Stability check: stratified k-fold cross-validation on the training
#      split. One test set of 114 patients gives one number; k-fold shows
#      how much that number would move with a different subset of patients.
# ---------------------------------------------------------------------------
"""Optional sanity (scikit-learn LR) and stability (k-fold CV) checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bcd.config import Config
from bcd.data.augmentation import apply_smote
from bcd.data.dataset import make_loader
from bcd.evaluation.metrics import METRIC_NAMES, compute_metrics, predict_labels
from bcd.pipeline.experiment import (
    MODEL_LR,
    MODELS,
    STAGE_2,
    STAGES,
    ExperimentResults,
    build_model,
)
from bcd.training.trainer import predict_proba, train_model
from bcd.utils import set_seed


def sklearn_sanity_check(results: ExperimentResults, cfg: Config) -> pd.DataFrame:
    """Compare scikit-learn's LogisticRegression with the PyTorch model on the test set.

    Args:
        results: The main experiment results (for the PyTorch LR and the data).
        cfg: Project configuration.

    Returns:
        DataFrame with one row per (implementation, stage): test metrics,
        plus `agreement`, the share of test patients both implementations
        label the same way.

    Notes:
        scikit-learn fits the same model (a linear logit + sigmoid, trained
        on cross-entropy) with a different optimiser (L-BFGS, run to
        convergence) and a mild L2 penalty by default (C = 1.0). The PyTorch
        version uses Adam with early stopping instead. Similar, not
        identical, numbers are therefore the expected outcome.
    """
    data = results.data
    thr = cfg.training.threshold
    rows = []
    for stage in STAGES:
        X_fit, y_fit = data.training_set(stage)
        sk = LogisticRegression(
            C=cfg.optional.sklearn_lr_C,
            max_iter=cfg.optional.sklearn_lr_max_iter,
            random_state=cfg.seed,
        )
        sk.fit(X_fit, y_fit)
        # predict_proba column 1 is P(label 1) = P(malignant).
        sk_prob = sk.predict_proba(data.scaled.X_test)[:, 1]
        torch_prob = results.run(MODEL_LR, stage).test_prob
        agreement = float(np.mean(predict_labels(sk_prob, thr) == predict_labels(torch_prob, thr)))
        for impl, prob in (("PyTorch LR", torch_prob), ("scikit-learn LR", sk_prob)):
            m = compute_metrics(data.scaled.y_test, prob, thr)
            rows.append(
                {
                    "implementation": impl,
                    "stage": stage,
                    **{k: m[k] for k in (*METRIC_NAMES, "fn", "fp")},
                    "agreement": agreement,
                }
            )
    return pd.DataFrame(rows)


def cross_validation_check(results: ExperimentResults, cfg: Config) -> pd.DataFrame:
    """Stratified k-fold CV on the training split, for both models and stages.

    Args:
        results: The main experiment results (for the data splits).
        cfg: Project configuration.

    Returns:
        DataFrame with one row per (model, stage, fold) and the metrics on
        the held-out fold.

    Notes:
        For each fold, everything that learns from data is refitted on the
        fold's training part only, exactly as in the main pipeline:
        the scaler, SMOTE (Stage 2) and the model. Early stopping keeps
        using the fixed validation split (scaled with the fold's scaler), so
        the held-out fold is used only for scoring. The test set is not
        touched at all.
    """
    raw = results.data.raw
    tr = cfg.training
    skf = StratifiedKFold(n_splits=cfg.optional.cv_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(raw.X_train, raw.y_train), start=1):
        scaler = StandardScaler().fit(raw.X_train[fit_idx])
        X_fit = scaler.transform(raw.X_train[fit_idx])
        y_fit = raw.y_train[fit_idx]
        X_hold, y_hold = scaler.transform(raw.X_train[hold_idx]), raw.y_train[hold_idx]
        X_val = scaler.transform(raw.X_val)
        for stage in STAGES:
            if stage == STAGE_2:
                X_stage, y_stage = apply_smote(
                    X_fit, y_fit, cfg.augmentation.k_neighbors,
                    cfg.augmentation.sampling_strategy, cfg.seed,
                )
            else:
                X_stage, y_stage = X_fit, y_fit
            for model_name in MODELS:
                set_seed(cfg.seed, cfg.deterministic)
                model, weight_decay = build_model(model_name, cfg, X_stage.shape[1])
                train_model(
                    model,
                    make_loader(X_stage, y_stage, tr.batch_size, shuffle=True, seed=cfg.seed),
                    make_loader(X_val, raw.y_val, tr.batch_size, shuffle=False, seed=cfg.seed),
                    learning_rate=tr.learning_rate,
                    weight_decay=weight_decay,
                    max_epochs=tr.max_epochs,
                    patience=tr.patience,
                )
                m = compute_metrics(y_hold, predict_proba(model, X_hold), tr.threshold)
                rows.append(
                    {"model": model_name, "stage": stage, "fold": fold,
                     **{k: m[k] for k in METRIC_NAMES}}
                )
    return pd.DataFrame(rows)


def summarise_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds.

    Args:
        cv: Output of `cross_validation_check`.

    Returns:
        DataFrame indexed by (model, stage) with "<metric>_mean" / "<metric>_std".
    """
    agg = cv.groupby(["model", "stage"], sort=False)[list(METRIC_NAMES)].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg
