# ---------------------------------------------------------------------------
# The ONE training loop used by both models and both stages.
#
# It only relies on the shared model contract (features in, one logit per
# sample out), so it trains logistic regression and the ANN identically:
# same loss, same optimiser, same batching, same early stopping.
# ---------------------------------------------------------------------------
"""Mini-batch training with backpropagation and early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """What happened during training (used for the learning curves).

    Attributes:
        train_loss: Loss on the training data after each epoch, measured in
            eval mode (dropout off), the same way as the validation loss.
        val_loss: Loss on the validation split after each epoch.
        best_epoch: 1-based epoch with the lowest validation loss; its
            weights are the ones the model ends up with.
        epochs_run: How many epochs were actually run before stopping.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0


def make_loss() -> nn.Module:
    """Return the loss function shared by both models.

    Notes:
        NOTE: BCEWithLogitsLoss = sigmoid + binary cross-entropy in one step.
        Binary cross-entropy, -[y log p + (1 - y) log(1 - p)] with
        p = sigmoid(logit), is exactly the logistic regression cost function.
        Computing sigmoid and log separately can overflow or produce log(0)
        for large |logit|; the fused version uses the log-sum-exp trick and
        stays numerically stable. That is why the models output raw logits.

        No pos_weight is used: in Stage 1 the classes are left as they are,
        and Stage 2 handles the imbalance with SMOTE. Mixing both would make
        it impossible to tell which one caused a change.
    """
    return nn.BCEWithLogitsLoss()


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    """Average loss of `model` over every sample in `loader`.

    Args:
        model: The model to evaluate.
        loader: DataLoader over the data to evaluate.
        loss_fn: The loss function (mean reduction).

    Returns:
        The mean per-sample loss.

    Notes:
        model.eval() switches dropout off so the measurement is deterministic,
        and @torch.no_grad() stops PyTorch from recording the computation
        graph: no gradients are needed for evaluation, which saves memory and time.
    """
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        # The loss is a batch mean; weight it by the batch size so the last
        # (smaller) batch does not count as much as a full one.
        total += loss_fn(model(xb), yb).item() * len(yb)
        n += len(yb)
    return total / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> TrainingHistory:
    """Train `model` with Adam and early stopping on validation loss.

    Args:
        model: Any classifier that maps (batch, n_features) to (batch,) logits.
        train_loader: Shuffled loader over the data the model is fitted on
            (the original training split in Stage 1, the SMOTE set in Stage 2).
        val_loader: Loader over the validation split (never resampled).
        learning_rate: Adam step size.
        weight_decay: L2 penalty strength (0 disables it).
        max_epochs: Upper bound on the number of passes over the training data.
        patience: Stop after this many epochs without a new best validation loss.

    Returns:
        The TrainingHistory. The model is modified in place and ends with
        the weights of its best validation epoch.

    Notes:
        NOTE: Adam adapts the step size of every weight from running
        averages of its gradients, so it converges reliably without a
        learning-rate schedule; lr = 0.001 is its standard default.
        weight_decay adds an L2 penalty that pulls weights towards zero,
        discouraging the large weights typical of an overfitted network.

        NOTE: Early stopping watches the VALIDATION loss, i.e. performance
        on patients the weights are not fitted on. Training loss keeps
        falling as long as the model can memorise more; validation loss
        starts rising once the model begins to fit noise. We keep the
        weights from the epoch with the lowest validation loss and stop
        after `patience` epochs without improvement. Patience prevents
        stopping on a small random bump in the validation curve.
    """
    loss_fn = make_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # A non-shuffled view of the training data, used only to MEASURE the
    # training loss at the end of each epoch (for the learning curve).
    train_eval_loader = DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False
    )

    history = TrainingHistory()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # train() mode: dropout is active (it only matters for the ANN).
        model.train()
        for xb, yb in train_loader:
            # 1. Forward pass: compute one logit per sample in the batch.
            logits = model(xb)
            # 2. Loss: how far the predictions are from the true labels.
            loss = loss_fn(logits, yb)
            # 3. Reset gradients: PyTorch ACCUMULATES gradients by default,
            #    so the previous batch's gradients must be cleared first.
            optimizer.zero_grad()
            # 4. Backward pass (backpropagation): compute d(loss)/d(weight)
            #    for every weight via the chain rule.
            loss.backward()
            # 5. Optimiser step: move every weight against its gradient.
            optimizer.step()

        # End of epoch: measure both losses in eval mode (no dropout, no grad)
        # so the two curves are directly comparable.
        train_loss = evaluate_loss(model, train_eval_loader, loss_fn)
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.epochs_run = epoch

        # Early stopping bookkeeping.
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Restore the weights of the best validation epoch, not the last epoch.
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return P(malignant) for every row of X.

    Args:
        model: A trained classifier returning logits.
        X: Scaled features, shape (n_samples, n_features).

    Returns:
        Probabilities in [0, 1], shape (n_samples,).

    Notes:
        This is the only place the sigmoid is applied explicitly: it maps
        the logit to a probability. eval() disables dropout.
    """
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()
