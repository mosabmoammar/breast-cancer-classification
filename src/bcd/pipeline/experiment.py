# ---------------------------------------------------------------------------
# The experiment required by the brief:
#   Stage 1: train both models on the original (scaled) training split.
#   Stage 2: train both models on the SMOTE-rebalanced training split.
#
# Both stages share the same split, scaler, architectures, hyperparameters
# and seed. The ONLY difference is the training set handed to the trainer.
# Both stages are evaluated on the same untouched test set.
# ---------------------------------------------------------------------------
"""Run Stage 1 and Stage 2 for both models and collect the results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler
from torch import nn

from bcd.config import Config
from bcd.data.augmentation import apply_smote, class_counts
from bcd.data.dataset import make_loader
from bcd.data.loader import load_data
from bcd.data.preprocessing import scale_splits
from bcd.data.splitter import DataSplits, split_data
from bcd.evaluation import plots
from bcd.evaluation.metrics import METRIC_NAMES, compute_metrics
from bcd.models import ANN, LogisticRegressionModel
from bcd.training.trainer import TrainingHistory, predict_proba, train_model
from bcd.utils import set_seed

MODEL_LR = "Logistic Regression"
MODEL_ANN = "ANN"
MODELS: tuple[str, ...] = (MODEL_LR, MODEL_ANN)

STAGE_1 = "Stage 1 (original)"
STAGE_2 = "Stage 2 (SMOTE)"
STAGES: tuple[str, ...] = (STAGE_1, STAGE_2)


@dataclass(frozen=True)
class PreparedData:
    """All data needed by both stages, prepared once.

    Attributes:
        feature_names: The 30 feature names.
        raw: Unscaled splits (kept for exploration and the optional CV check).
        scaled: Standardised splits (scaler fitted on the training split only).
        scaler: The fitted StandardScaler.
        X_train_smote, y_train_smote: Scaled training split after SMOTE (Stage 2).
    """

    feature_names: list[str]
    raw: DataSplits
    scaled: DataSplits
    scaler: StandardScaler
    X_train_smote: np.ndarray
    y_train_smote: np.ndarray

    def training_set(self, stage: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the training data the model is FITTED on in `stage`."""
        if stage == STAGE_1:
            return self.scaled.X_train, self.scaled.y_train
        if stage == STAGE_2:
            return self.X_train_smote, self.y_train_smote
        raise ValueError(f"Unknown stage: {stage}")


@dataclass
class RunResult:
    """Outcome of training and evaluating one model in one stage.

    Attributes:
        model_name, stage: Which run this is.
        model: The trained nn.Module (best-validation weights).
        history: Loss curves and early-stopping information.
        train_metrics: Metrics on the REAL training samples (no synthetic ones).
        test_metrics: Metrics on the untouched test set.
        test_prob: P(malignant) for each test sample (for the ROC curve).
    """

    model_name: str
    stage: str
    model: nn.Module
    history: TrainingHistory
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    test_prob: np.ndarray


@dataclass
class ExperimentResults:
    """Everything produced by `run_experiment`."""

    data: PreparedData
    runs: dict[tuple[str, str], RunResult]

    def run(self, model_name: str, stage: str) -> RunResult:
        """Look up one run by model and stage."""
        return self.runs[(model_name, stage)]


def prepare_data(cfg: Config) -> PreparedData:
    """Load, split, scale and (for Stage 2) oversample the data.

    Args:
        cfg: Project configuration.

    Returns:
        A PreparedData shared by both stages.

    Notes:
        Order matters and is what prevents data leakage:
        split -> fit scaler on train -> SMOTE on scaled train only.
    """
    X, y, feature_names = load_data()
    raw = split_data(X, y, cfg.data.test_size, cfg.data.val_size, cfg.seed)
    scaled, scaler = scale_splits(raw)
    X_smote, y_smote = apply_smote(
        scaled.X_train,
        scaled.y_train,
        k_neighbors=cfg.augmentation.k_neighbors,
        sampling_strategy=cfg.augmentation.sampling_strategy,
        seed=cfg.seed,
    )
    return PreparedData(feature_names, raw, scaled, scaler, X_smote, y_smote)


def build_model(model_name: str, cfg: Config, n_features: int) -> tuple[nn.Module, float]:
    """Instantiate a model and return it with its weight-decay setting.

    Args:
        model_name: MODEL_LR or MODEL_ANN.
        cfg: Project configuration.
        n_features: Number of input features.

    Returns:
        (model, weight_decay).
    """
    if model_name == MODEL_LR:
        return LogisticRegressionModel(n_features), cfg.models.logistic_regression.weight_decay
    if model_name == MODEL_ANN:
        ann_cfg = cfg.models.ann
        return ANN(n_features, ann_cfg.hidden_sizes, ann_cfg.dropout), ann_cfg.weight_decay
    raise ValueError(f"Unknown model: {model_name}")


def train_and_evaluate(model_name: str, stage: str, data: PreparedData, cfg: Config) -> RunResult:
    """Train one model in one stage and evaluate it on train and test.

    Args:
        model_name: MODEL_LR or MODEL_ANN.
        stage: STAGE_1 or STAGE_2.
        data: Prepared data.
        cfg: Project configuration.

    Returns:
        The RunResult.

    Notes:
        NOTE: The seed is reset before EVERY run, so the four runs start from
        the same random state (initial weights, dropout masks, shuffling) and
        no run depends on the ones before it.

        NOTE: Train metrics are computed on the real training samples, even
        in Stage 2. The synthetic SMOTE points are a training device, not
        patients; measuring the train/test gap on real samples keeps the
        overfitting diagnosis comparable between the two stages.
    """
    set_seed(cfg.seed, cfg.deterministic)
    tr = cfg.training
    X_fit, y_fit = data.training_set(stage)
    model, weight_decay = build_model(model_name, cfg, n_features=X_fit.shape[1])

    train_loader = make_loader(X_fit, y_fit, tr.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(
        data.scaled.X_val, data.scaled.y_val, tr.batch_size, shuffle=False, seed=cfg.seed
    )
    history = train_model(
        model,
        train_loader,
        val_loader,
        learning_rate=tr.learning_rate,
        weight_decay=weight_decay,
        max_epochs=tr.max_epochs,
        patience=tr.patience,
    )

    train_prob = predict_proba(model, data.scaled.X_train)
    test_prob = predict_proba(model, data.scaled.X_test)
    return RunResult(
        model_name=model_name,
        stage=stage,
        model=model,
        history=history,
        train_metrics=compute_metrics(data.scaled.y_train, train_prob, tr.threshold),
        test_metrics=compute_metrics(data.scaled.y_test, test_prob, tr.threshold),
        test_prob=test_prob,
    )


def run_experiment(cfg: Config, data: PreparedData | None = None) -> ExperimentResults:
    """Run both stages for both models.

    Args:
        cfg: Project configuration.
        data: Optionally, already prepared data (otherwise prepared here).

    Returns:
        ExperimentResults with the four runs.
    """
    data = data if data is not None else prepare_data(cfg)
    runs = {
        (model_name, stage): train_and_evaluate(model_name, stage, data, cfg)
        for stage in STAGES
        for model_name in MODELS
    }
    return ExperimentResults(data, runs)


# ----------------------------- result tables -------------------------------


def class_distribution_table(data: PreparedData) -> pd.DataFrame:
    """Class counts for the full data and every subset, before and after SMOTE.

    Args:
        data: Prepared data.

    Returns:
        DataFrame indexed by subset with columns benign, malignant, total,
        malignant_pct.
    """
    y_full = np.concatenate([data.raw.y_train, data.raw.y_val, data.raw.y_test])
    subsets = {
        "Full dataset": y_full,
        "Train (Stage 1)": data.scaled.y_train,
        "Train after SMOTE (Stage 2)": data.y_train_smote,
        "Validation": data.scaled.y_val,
        "Test": data.scaled.y_test,
    }
    rows = []
    for name, y in subsets.items():
        c = class_counts(y)
        total = c["benign"] + c["malignant"]
        rows.append({"subset": name, **c, "total": total, "malignant_pct": 100 * c["malignant"] / total})
    return pd.DataFrame(rows).set_index("subset")


def metrics_table(results: ExperimentResults) -> pd.DataFrame:
    """One row per (model, stage, split) with every scalar metric.

    Args:
        results: Experiment results.

    Returns:
        Long-format DataFrame: model, stage, split, accuracy, precision,
        recall, f1, roc_auc, tn, fp, fn, tp, best_epoch, epochs_run.
    """
    rows = []
    for (model_name, stage), run in results.runs.items():
        for split, m in (("train", run.train_metrics), ("test", run.test_metrics)):
            rows.append(
                {
                    "model": model_name,
                    "stage": stage,
                    "split": split,
                    **{k: m[k] for k in (*METRIC_NAMES, "tn", "fp", "fn", "tp")},
                    "best_epoch": run.history.best_epoch,
                    "epochs_run": run.history.epochs_run,
                }
            )
    return pd.DataFrame(rows)


def comparison_table(results: ExperimentResults) -> pd.DataFrame:
    """Stage 1 vs Stage 2 on the TEST set, per model and metric.

    Args:
        results: Experiment results.

    Returns:
        DataFrame with columns model, metric, <STAGE_1>, <STAGE_2>, delta
        (delta = Stage 2 - Stage 1). Includes the FN and FP counts.
    """
    rows = []
    for model_name in MODELS:
        s1 = results.run(model_name, STAGE_1).test_metrics
        s2 = results.run(model_name, STAGE_2).test_metrics
        for metric in (*METRIC_NAMES, "fn", "fp"):
            delta = s2[metric] - s1[metric]
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    STAGE_1: s1[metric],
                    STAGE_2: s2[metric],
                    # Subtracting two equal floats can leave noise like -1e-16;
                    # report it as an exact 0 so the table reads "no change".
                    "delta": 0.0 if abs(delta) < 1e-12 else delta,
                }
            )
    return pd.DataFrame(rows)


def generalisation_gap_table(results: ExperimentResults) -> pd.DataFrame:
    """Train vs test scores and their gap, used for the overfitting diagnosis.

    Args:
        results: Experiment results.

    Returns:
        DataFrame with columns model, stage, metric, train, test, gap
        (gap = train - test; a large positive gap suggests overfitting).
    """
    rows = []
    for (model_name, stage), run in results.runs.items():
        for metric in METRIC_NAMES:
            train, test = run.train_metrics[metric], run.test_metrics[metric]
            rows.append(
                {"model": model_name, "stage": stage, "metric": metric,
                 "train": train, "test": test, "gap": train - test}
            )
    return pd.DataFrame(rows)


def training_summary_table(results: ExperimentResults) -> pd.DataFrame:
    """Early-stopping summary and final losses per run.

    Args:
        results: Experiment results.

    Returns:
        DataFrame with model, stage, n_train_samples, epochs_run, best_epoch,
        and the train / validation loss at the best epoch.
    """
    rows = []
    for (model_name, stage), run in results.runs.items():
        h = run.history
        rows.append(
            {
                "model": model_name,
                "stage": stage,
                "n_train_samples": len(results.data.training_set(stage)[1]),
                "epochs_run": h.epochs_run,
                "best_epoch": h.best_epoch,
                "train_loss_at_best": h.train_loss[h.best_epoch - 1],
                "val_loss_at_best": h.val_loss[h.best_epoch - 1],
            }
        )
    return pd.DataFrame(rows)


def misclassified_table(results: ExperimentResults, threshold: float) -> pd.DataFrame:
    """Every test patient misclassified by at least one run, with all runs' P(malignant).

    Args:
        results: Experiment results.
        threshold: Decision threshold.

    Returns:
        DataFrame indexed by test-set position with the true class and one
        P(malignant) column per (model, stage). Shows WHICH patients are
        missed (FN) or falsely flagged (FP), and how close they are to the
        threshold, so changes between stages can be traced to individual cases.
    """
    y = results.data.scaled.y_test
    probs = {f"{m} | {s}": results.run(m, s).test_prob for s in STAGES for m in MODELS}
    wrong = np.zeros(len(y), dtype=bool)
    for p in probs.values():
        wrong |= (p >= threshold).astype(np.int64) != y
    idx = np.flatnonzero(wrong)
    table = pd.DataFrame(
        {"true_class": np.where(y[idx] == 1, "malignant", "benign"),
         **{name: p[idx] for name, p in probs.items()}},
        index=pd.Index(idx, name="test_index"),
    )
    return table


# ------------------------------- figures -----------------------------------


def make_figures(results: ExperimentResults, figures_dir: Path) -> dict[str, Figure]:
    """Create and save every required figure.

    Args:
        results: Experiment results.
        figures_dir: Output folder.

    Returns:
        {figure name: Figure}, so the notebook can display them.
    """
    data = results.data
    dist = class_distribution_table(data)
    counts = {
        name: {"benign": int(row["benign"]), "malignant": int(row["malignant"])}
        for name, row in dist.iterrows()
    }
    matrices = {k: r.test_metrics["confusion_matrix"] for k, r in results.runs.items()}
    roc = {
        stage: {
            m: (data.scaled.y_test, results.run(m, stage).test_prob,
                results.run(m, stage).test_metrics["roc_auc"])
            for m in MODELS
        }
        for stage in STAGES
    }
    histories = {k: r.history for k, r in results.runs.items()}
    # Per-stage versions, shown in the notebook's Stage 1 / Stage 2 sections.
    per_stage: dict[str, Figure] = {}
    for n, stage in enumerate(STAGES, start=1):
        per_stage[f"confusion_matrices_stage{n}"] = plots.plot_confusion_matrices(
            matrices, MODELS, [stage], figures_dir / f"confusion_matrices_stage{n}.png"
        )
        per_stage[f"loss_curves_stage{n}"] = plots.plot_loss_curves(
            histories, MODELS, [stage], figures_dir / f"loss_curves_stage{n}.png"
        )
    return per_stage | {
        "class_distribution": plots.plot_class_distribution(
            counts, figures_dir / "class_distribution.png"
        ),
        "confusion_matrices": plots.plot_confusion_matrices(
            matrices, MODELS, STAGES, figures_dir / "confusion_matrices.png"
        ),
        "roc_curves": plots.plot_roc_curves(roc, figures_dir / "roc_curves.png"),
        "loss_curves": plots.plot_loss_curves(
            histories, MODELS, STAGES, figures_dir / "loss_curves.png"
        ),
        "stage_comparison": plots.plot_stage_comparison(
            comparison_table(results), figures_dir / "stage_comparison.png"
        ),
    }


def save_tables(results: ExperimentResults, results_dir: Path, threshold: float) -> dict[str, Path]:
    """Write every result table (CSV) and the classification reports (text).

    Args:
        results: Experiment results.
        results_dir: Output folder.
        threshold: Decision threshold (for the misclassified-patients table).

    Returns:
        {table name: path written}.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "all_metrics": metrics_table(results),
        "stage_comparison": comparison_table(results),
        "generalisation_gap": generalisation_gap_table(results),
        "training_summary": training_summary_table(results),
    }
    paths: dict[str, Path] = {}
    for name, df in tables.items():
        paths[name] = results_dir / f"{name}.csv"
        df.to_csv(paths[name], index=False, float_format="%.6f")
    paths["class_distribution"] = results_dir / "class_distribution.csv"
    class_distribution_table(results.data).to_csv(paths["class_distribution"], float_format="%.2f")
    paths["misclassified_test_patients"] = results_dir / "misclassified_test_patients.csv"
    misclassified_table(results, threshold).to_csv(
        paths["misclassified_test_patients"], float_format="%.4f"
    )

    report_lines = []
    for (model_name, stage), run in results.runs.items():
        for split, m in (("TRAIN", run.train_metrics), ("TEST", run.test_metrics)):
            report_lines += [f"=== {model_name} | {stage} | {split} ===", m["report"], ""]
    paths["classification_reports"] = results_dir / "classification_reports.txt"
    paths["classification_reports"].write_text("\n".join(report_lines), encoding="utf-8")
    return paths
