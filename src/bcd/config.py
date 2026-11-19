# ---------------------------------------------------------------------------
# Configuration loading and validation.
#
# Reads config/config.yaml into frozen (immutable) dataclasses. Every other
# module receives these typed objects instead of raw dictionaries, so a typo
# in a key fails loudly here, at start-up, rather than silently deep inside
# training. Freezing the objects guarantees that no stage can modify a
# setting the other stage relies on.
# ---------------------------------------------------------------------------
"""Typed, validated access to config/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The project root is three levels above this file: src/bcd/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    """Split ratios. Both splits are stratified on the label."""

    test_size: float
    val_size: float


@dataclass(frozen=True)
class AugmentationConfig:
    """SMOTE settings for Stage 2."""

    method: str
    k_neighbors: int
    sampling_strategy: float


@dataclass(frozen=True)
class LogisticRegressionConfig:
    """Logistic regression settings (no hidden layer, no dropout)."""

    weight_decay: float


@dataclass(frozen=True)
class ANNConfig:
    """Multilayer Perceptron settings."""

    hidden_sizes: tuple[int, ...]
    dropout: float
    weight_decay: float


@dataclass(frozen=True)
class ModelsConfig:
    """Settings for both models."""

    logistic_regression: LogisticRegressionConfig
    ann: ANNConfig


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser, batching, early stopping and decision threshold."""

    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    threshold: float


@dataclass(frozen=True)
class OptionalConfig:
    """Settings for the optional sanity / stability checks."""

    cv_folds: int
    sklearn_lr_C: float
    sklearn_lr_max_iter: int


@dataclass(frozen=True)
class PathsConfig:
    """Output folders, resolved relative to the project root."""

    figures: Path
    results: Path


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    seed: int
    deterministic: bool
    data: DataConfig
    augmentation: AugmentationConfig
    models: ModelsConfig
    training: TrainingConfig
    optional: OptionalConfig
    paths: PathsConfig


def _require_fraction(name: str, value: float) -> None:
    """Raise if `value` is not strictly between 0 and 1."""
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {value}")


def _require_positive(name: str, value: float) -> None:
    """Raise if `value` is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate(cfg: Config) -> None:
    """Check value ranges so that a bad config fails before any training."""
    _require_fraction("data.test_size", cfg.data.test_size)
    _require_fraction("data.val_size", cfg.data.val_size)
    if cfg.augmentation.method != "smote":
        raise ValueError("augmentation.method must be 'smote'")
    _require_positive("augmentation.k_neighbors", cfg.augmentation.k_neighbors)
    if not 0.0 < cfg.augmentation.sampling_strategy <= 1.0:
        raise ValueError("augmentation.sampling_strategy must be in (0, 1]")
    if not 0.0 <= cfg.models.ann.dropout < 1.0:
        raise ValueError("models.ann.dropout must be in [0, 1)")
    if not cfg.models.ann.hidden_sizes or any(h <= 0 for h in cfg.models.ann.hidden_sizes):
        raise ValueError("models.ann.hidden_sizes must be a non-empty list of positive ints")
    if cfg.models.ann.weight_decay < 0 or cfg.models.logistic_regression.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    _require_positive("training.learning_rate", cfg.training.learning_rate)
    _require_positive("training.batch_size", cfg.training.batch_size)
    _require_positive("training.max_epochs", cfg.training.max_epochs)
    _require_positive("training.patience", cfg.training.patience)
    _require_fraction("training.threshold", cfg.training.threshold)
    if cfg.optional.cv_folds < 2:
        raise ValueError("optional.cv_folds must be >= 2")


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a Config from a parsed YAML dictionary.

    Args:
        raw: Dictionary with the same structure as config/config.yaml.

    Returns:
        A frozen, validated Config.

    Raises:
        KeyError / TypeError: if a key is missing or unexpected.
        ValueError: if a value is out of range.
    """
    models = raw["models"]
    cfg = Config(
        seed=int(raw["seed"]),
        deterministic=bool(raw["deterministic"]),
        data=DataConfig(**raw["data"]),
        augmentation=AugmentationConfig(**raw["augmentation"]),
        models=ModelsConfig(
            logistic_regression=LogisticRegressionConfig(**models["logistic_regression"]),
            ann=ANNConfig(
                hidden_sizes=tuple(models["ann"]["hidden_sizes"]),
                dropout=float(models["ann"]["dropout"]),
                weight_decay=float(models["ann"]["weight_decay"]),
            ),
        ),
        training=TrainingConfig(**raw["training"]),
        optional=OptionalConfig(**raw["optional"]),
        paths=PathsConfig(
            figures=PROJECT_ROOT / raw["paths"]["figures"],
            results=PROJECT_ROOT / raw["paths"]["results"],
        ),
    )
    _validate(cfg)
    return cfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load config/config.yaml (or another YAML file) into a Config.

    Args:
        path: Path to the YAML file. Defaults to the project's config.

    Returns:
        A frozen, validated Config.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)
