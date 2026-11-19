# ---------------------------------------------------------------------------
# Command-line entry point. Runs the whole project in one go:
#   prepare data -> Stage 1 + Stage 2 for both models -> tables and figures
#   -> (optionally) the sanity and stability checks.
# Usage (from the project root, with the venv active):
#   python scripts/run_experiments.py
#   python scripts/run_experiments.py --skip-optional
# ---------------------------------------------------------------------------
"""Run every experiment and write outputs/figures and outputs/results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# Non-interactive backend: the script only writes PNG files, no windows.
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from bcd.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from bcd.pipeline import experiment as exp  # noqa: E402
from bcd.pipeline import optional_checks  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to the YAML config (default: config/config.yaml)")
    parser.add_argument("--skip-optional", action="store_true",
                        help="Skip the optional scikit-learn and k-fold checks")
    return parser.parse_args()


def main() -> int:
    """Run the experiment, print a summary and save every output."""
    args = parse_args()
    cfg = load_config(args.config)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("Preparing data (split -> scale -> SMOTE on train only)...")
    data = exp.prepare_data(cfg)
    print(exp.class_distribution_table(data).round(2), "\n")

    print("Training 2 models x 2 stages...")
    results = exp.run_experiment(cfg, data)
    print(exp.training_summary_table(results).round(4), "\n")

    print("Train and test metrics (positive class = malignant):")
    cols = ["model", "stage", "split", "accuracy", "precision", "recall", "f1", "roc_auc", "fn", "fp"]
    print(exp.metrics_table(results)[cols].round(4).to_string(index=False), "\n")

    print("Stage 1 vs Stage 2 (test set):")
    print(exp.comparison_table(results).round(4).to_string(index=False), "\n")

    print("Test patients misclassified by at least one run (P(malignant) per run):")
    print(exp.misclassified_table(results, cfg.training.threshold).round(3).to_string(), "\n")

    paths = exp.save_tables(results, cfg.paths.results, cfg.training.threshold)
    exp.make_figures(results, cfg.paths.figures)
    print(f"Saved {len(paths)} result files to {cfg.paths.results}")
    print(f"Saved figures to {cfg.paths.figures}\n")

    if not args.skip_optional:
        print("[Optional] scikit-learn LogisticRegression sanity check (test set):")
        sanity = optional_checks.sklearn_sanity_check(results, cfg)
        print(sanity.round(4).to_string(index=False), "\n")
        sanity.to_csv(cfg.paths.results / "optional_sklearn_sanity_check.csv",
                      index=False, float_format="%.6f")

        print(f"[Optional] Stratified {cfg.optional.cv_folds}-fold CV on the training split:")
        cv = optional_checks.cross_validation_check(results, cfg)
        summary = optional_checks.summarise_cv(cv)
        print(summary.round(4).to_string(), "\n")
        cv.to_csv(cfg.paths.results / "optional_cv_folds.csv", index=False, float_format="%.6f")
        summary.to_csv(cfg.paths.results / "optional_cv_summary.csv", float_format="%.6f")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
