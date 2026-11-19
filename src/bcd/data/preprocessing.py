# ---------------------------------------------------------------------------
# Step 3 of the pipeline: standardise the features.
#
# The scaler learns its mean and standard deviation from the TRAINING split
# only and then applies those same numbers to validation and test.
# ---------------------------------------------------------------------------
"""Feature standardisation without data leakage."""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler

from bcd.data.splitter import DataSplits


def scale_splits(splits: DataSplits) -> tuple[DataSplits, StandardScaler]:
    """Fit a StandardScaler on the training split and transform all three splits.

    Args:
        splits: Unscaled train / validation / test data.

    Returns:
        scaled: A new DataSplits with standardised features (labels unchanged).
        scaler: The fitted scaler (its mean_/scale_ come from training data only).

    Notes:
        NOTE: Why scale at all: the 30 features live on very different
        scales (e.g. "mean area" is in the hundreds, "mean smoothness" is
        around 0.1). Gradient descent on unscaled inputs converges slowly
        and lets large-valued features dominate the weights. Standardising
        to zero mean / unit variance puts every feature on the same footing.

        NOTE: Why fit on training data only: if the mean and standard
        deviation were computed on all 569 samples, statistics of the test
        patients would leak into training. The model would then be
        evaluated on data it had indirectly "seen", and the test score would
        be optimistic. In deployment, a new patient's measurements are
        scaled with statistics from past (training) data, which is exactly
        what we reproduce here.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)
    scaled = DataSplits(X_train, splits.y_train, X_val, splits.y_val, X_test, splits.y_test)
    return scaled, scaler
