# ---------------------------------------------------------------------------
# bcd: breast cancer diagnosis.
#
# Package layout (dependencies point inward, towards data and models):
#   config.py      -> typed settings loaded from config/config.yaml
#   utils.py       -> seeding / determinism
#   data/          -> loading, splitting, scaling, SMOTE, PyTorch Dataset
#   models/        -> the two nn.Module classifiers
#   training/      -> the single shared training loop
#   evaluation/    -> metrics and figures
#   pipeline/      -> orchestrates Stage 1 and Stage 2 for both models
# ---------------------------------------------------------------------------
"""Breast cancer diagnosis with logistic regression and an ANN in PyTorch."""

__version__ = "1.0.0"
