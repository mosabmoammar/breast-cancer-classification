# ---------------------------------------------------------------------------
# Data layer: everything that happens to the data before a model sees it.
# Order of use in the pipeline:
#   loader -> splitter -> preprocessing (scaling) -> augmentation (Stage 2)
#   -> dataset (PyTorch Dataset / DataLoader)
# ---------------------------------------------------------------------------
"""Data loading, splitting, scaling, augmentation and PyTorch datasets."""
