# ---------------------------------------------------------------------------
# Model layer. Both classifiers are nn.Module subclasses with the SAME
# contract:
#   input : float tensor of shape (batch, n_features)
#   output: one raw logit per sample, shape (batch,)
# Because the contract is identical, the trainer and the pipeline never need
# to know which of the two models they are running.
# ---------------------------------------------------------------------------
"""The two classifiers: logistic regression and a small MLP."""

from bcd.models.ann import ANN
from bcd.models.logistic_regression import LogisticRegressionModel

__all__ = ["ANN", "LogisticRegressionModel"]
