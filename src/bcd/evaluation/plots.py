# ---------------------------------------------------------------------------
# Figures saved to outputs/figures/.
#
# Every function builds a matplotlib Figure with the object-oriented API
# (no global pyplot state), saves it to disk and returns it. The notebook
# can then display the returned Figure, and the CLI script simply discards
# it. Inputs are plain arrays / dicts, so this module does not depend on
# the pipeline (dependencies point inward).
# ---------------------------------------------------------------------------
"""Plotting helpers for class balance, confusion matrices, ROC, loss curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from bcd.data.loader import CLASS_NAMES
from bcd.evaluation.metrics import METRIC_NAMES
from bcd.training.trainer import TrainingHistory

# One fixed colour per class / stage so the same thing always looks the same.
CLASS_COLOURS = {"benign": "#4C9BE8", "malignant": "#E8574C"}
STAGE_COLOURS = ("#7F7F7F", "#2CA02C")


def _save(fig: Figure, path: Path) -> Figure:
    """Tighten the layout, write the figure as PNG and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(counts: Mapping[str, Mapping[str, int]], path: Path) -> Figure:
    """Grouped bar chart of benign / malignant counts per data subset.

    Args:
        counts: {subset name: {"benign": n0, "malignant": n1}}, in display order.
        path: Where to save the PNG.

    Returns:
        The Figure.
    """
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    subsets = list(counts)
    x = np.arange(len(subsets))
    width = 0.38
    for offset, cls in zip((-width / 2, width / 2), CLASS_NAMES):
        values = [counts[s][cls] for s in subsets]
        bars = ax.bar(x + offset, values, width, label=cls, color=CLASS_COLOURS[cls])
        ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in subsets])
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution per subset")
    ax.legend()
    return _save(fig, path)


def plot_confusion_matrices(
    matrices: Mapping[tuple[str, str], np.ndarray],
    models: Sequence[str],
    stages: Sequence[str],
    path: Path,
    title: str = "Test-set confusion matrices",
) -> Figure:
    """Grid of confusion matrices: one row per model, one column per stage.

    Args:
        matrices: {(model, stage): 2x2 matrix [[TN, FP], [FN, TP]]}.
        models: Row order.
        stages: Column order.
        path: Where to save the PNG.
        title: Figure title.

    Returns:
        The Figure.
    """
    fig = Figure(figsize=(4.6 * len(stages), 4.0 * len(models)))
    # squeeze=False always returns a 2-D array of axes, even for one column.
    axes = fig.subplots(len(models), len(stages), squeeze=False)
    for i, model in enumerate(models):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            sns.heatmap(
                matrices[(model, stage)],
                annot=True,
                fmt="d",
                cbar=False,
                cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax,
                annot_kws={"size": 14},
            )
            ax.set_title(f"{model} - {stage}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    fig.suptitle(title)
    return _save(fig, path)


def plot_roc_curves(
    curves: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray, float]]],
    path: Path,
) -> Figure:
    """ROC curves on the test set: one panel per stage, one line per model.

    Args:
        curves: {stage: {model: (y_true, y_prob, auc)}}.
        path: Where to save the PNG.

    Returns:
        The Figure.
    """
    fig = Figure(figsize=(5.5 * len(curves), 5))
    axes = np.atleast_1d(fig.subplots(1, len(curves)))
    for ax, (stage, per_model) in zip(axes, curves.items()):
        # Different line styles, because near-perfect curves overlap almost
        # entirely and a solid line would hide the one drawn before it.
        for k, (model, (y_true, y_prob, auc)) in enumerate(per_model.items()):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, lw=2.5 - k, ls=("-", "--")[k % 2], label=f"{model} (AUC = {auc:.4f})")
        # The diagonal is a classifier that guesses at random (AUC = 0.5).
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random guess")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (recall)")
        ax.set_title(f"ROC curve - {stage}")
        ax.legend(loc="lower right")
    return _save(fig, path)


def plot_loss_curves(
    histories: Mapping[tuple[str, str], TrainingHistory],
    models: Sequence[str],
    stages: Sequence[str],
    path: Path,
) -> Figure:
    """Train vs validation loss per epoch: one row per model, one column per stage.

    Args:
        histories: {(model, stage): TrainingHistory}.
        models: Row order.
        stages: Column order.
        path: Where to save the PNG.

    Returns:
        The Figure.

    Notes:
        A dotted vertical line marks the best (restored) epoch. Reading the
        curves: both high and flat = underfitting; validation rising while
        training keeps falling (a widening gap) = overfitting; both falling
        and levelling off close together = good fit.
    """
    fig = Figure(figsize=(6 * len(stages), 4 * len(models)))
    axes = fig.subplots(len(models), len(stages), squeeze=False)
    for i, model in enumerate(models):
        for j, stage in enumerate(stages):
            ax = axes[i, j]
            h = histories[(model, stage)]
            epochs = np.arange(1, len(h.train_loss) + 1)
            ax.plot(epochs, h.train_loss, label="train loss")
            ax.plot(epochs, h.val_loss, label="validation loss")
            ax.axvline(h.best_epoch, color="k", ls=":", lw=1, label=f"best epoch ({h.best_epoch})")
            ax.set_title(f"{model} - {stage}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BCE loss")
            ax.legend()
    fig.suptitle("Learning curves (loss measured in eval mode after each epoch)")
    return _save(fig, path)


def plot_stage_comparison(comparison: pd.DataFrame, path: Path) -> Figure:
    """Grouped bars of test metrics, Stage 1 vs Stage 2, one panel per model.

    Args:
        comparison: DataFrame with columns model, metric, and one column per
            stage (in display order) holding the test-set value.
        path: Where to save the PNG.

    Returns:
        The Figure.
    """
    # Keep only the scalar scores (the table may also hold FN/FP counts).
    comparison = comparison[comparison["metric"].isin(METRIC_NAMES)]
    stage_cols = [c for c in comparison.columns if c not in ("model", "metric", "delta")]
    models = list(dict.fromkeys(comparison["model"]))
    fig = Figure(figsize=(6.5 * len(models), 4.8))
    axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
    x = np.arange(len(METRIC_NAMES))
    width = 0.8 / len(stage_cols)
    # Zoom the y-axis to just below the lowest score (rounded down to 0.1)
    # so small differences stay visible; exact values are printed on the bars.
    y_min = max(0.0, np.floor((comparison[stage_cols].to_numpy().min() - 0.05) * 10) / 10)
    for ax, model in zip(axes, models):
        sub = comparison[comparison["model"] == model].set_index("metric").loc[list(METRIC_NAMES)]
        for k, stage in enumerate(stage_cols):
            bars = ax.bar(
                x + (k - (len(stage_cols) - 1) / 2) * width,
                sub[stage].to_numpy(),
                width,
                label=stage,
                color=STAGE_COLOURS[k % len(STAGE_COLOURS)],
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1, rotation=90)
        ax.set_xticks(x, METRIC_NAMES)
        ax.set_title(f"{model}: test metrics")
        ax.set_ylim(y_min, 1.04)
    axes[0].set_ylabel("Score")
    # One shared legend below the panels, so it never covers a bar.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(stage_cols), frameon=False)
    fig.suptitle("Stage 1 (original data) vs Stage 2 (SMOTE)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
