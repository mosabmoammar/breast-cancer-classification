# Design decisions

This document lists every important decision in the project, the reason
behind it, and the alternatives that were considered. All values live in
[`config/config.yaml`](../config/config.yaml); the code that implements each
decision is referenced next to it.

---

## 1. Problem framing: malignant is the positive class

**Decision.** Labels are remapped once, in
[`src/bcd/data/loader.py`](../src/bcd/data/loader.py): `1 = malignant`,
`0 = benign`. scikit-learn ships the opposite (`0 = malignant`).

**Why.**
- Precision, recall, F1 and the confusion-matrix terms (TP, FN, ...) are
  defined relative to a *positive* class, and scikit-learn's metric
  functions treat label `1` as positive by default.
- The class we need to detect is malignant. With `1 = malignant`, "recall"
  means "share of cancers caught" and a false negative means "a cancer we
  missed", which is the error that matters clinically.
- Remapping in exactly one place means no other module has to remember the
  encoding; the constants `MALIGNANT`, `BENIGN` and `CLASS_NAMES` are imported
  everywhere else. All metric calls also pass `pos_label=1` explicitly.

**Alternatives.** Keep scikit-learn's encoding and pass `pos_label=0`
everywhere. Rejected: every call site becomes a chance to forget it, and
the confusion matrix and ROC curve would still be oriented around benign.

## 2. Main metric: malignant recall

**Decision.** Recall of the malignant class is the main metric. F1 and
ROC-AUC are reported next to it, and precision shows the cost in false
alarms. Accuracy is reported but not used to rank models.

**Why.** A false negative (a missed cancer) delays treatment; a false
positive leads to an extra examination. The costs are very asymmetric, so the
metric must focus on false negatives. Accuracy hides this: with about 63%
benign samples, a model that always says "benign" already reaches about 63%
accuracy while catching no cancer at all. ROC-AUC is useful because it
measures how well the model *ranks* malignant above benign independently of
the 0.5 cut-off.

## 3. Splits: 20% test, then 20% of the rest for validation, stratified

**Decision.** [`splitter.py`](../src/bcd/data/splitter.py): stratified
`train_test_split` with `test_size=0.2, random_state=42`, then a second
stratified split of the remaining data with `test_size=0.2`. Result: 364
train / 91 validation / 114 test. The same split is used for both models and
both stages.

**Why.**
- **Test first, from the untouched data**: the test set must not influence
  anything (scaling, SMOTE, early stopping, any choice). It is used once.
- **A separate validation set** is needed for early stopping and learning
  curves. Using the test set for early stopping would be tuning on the test
  set and would make the reported test score optimistic.
- **Stratified**: with only 114 / 91 samples, a random split could give one
  subset noticeably fewer malignant cases by chance. Stratification keeps the
  ~37% malignant share in every subset.
- **80/20 then 80/20** keeps most of the small dataset (64%) for fitting
  while leaving enough malignant cases in validation (34) and test (42) for
  the metrics to be meaningful.
- **Same split everywhere** so that every comparison (LR vs ANN, Stage 1 vs
  Stage 2) is made on exactly the same patients.

**Alternatives.** Plain k-fold cross-validation instead of a fixed test set:
better use of data, but the brief asks for one fixed stratified train/test
split. A stratified 5-fold CV on the training split is reported as an
*optional* stability check (decision 13).

## 4. Scaling: StandardScaler fitted on the training split only

**Decision.** [`preprocessing.py`](../src/bcd/data/preprocessing.py):
`StandardScaler.fit` on the training split; `transform` on validation and test.

**Why scale.** The features differ by orders of magnitude ("mean area" is in
the hundreds, "mean smoothness" around 0.1). Gradient descent with one
learning rate for all weights converges badly when inputs have very
different scales, and SMOTE's Euclidean distances would be dominated by the
largest features.

**Why train only.** Computing mean and standard deviation on all data would
let test-set statistics leak into training (data leakage) and inflate the
test score. In real use, a new patient is scaled with statistics from past
data, which is what this reproduces. A test checks that the scaler's
statistics equal those of the training split.

**Alternatives.** Min-max scaling: sensitive to outliers, which this dataset
has (e.g. large "area" values). StandardScaler is the usual choice for
gradient-trained linear models and MLPs.

## 5. Both models in PyTorch with one shared training loop

**Decision.** Logistic regression is `nn.Linear(30, 1)`; the ANN is a small
MLP. Both are `nn.Module`s that return one logit per sample and are trained
by the same [`train_model`](../src/bcd/training/trainer.py) function, with the
same `Dataset`/`DataLoader`, loss, optimiser and early stopping.

**Why.** If logistic regression were trained with a different library and
optimiser, a difference in results could come from the training procedure
rather than from the model. With identical training, the only difference is
the model's capacity: linear decision boundary vs non-linear. The shared
input/output contract also keeps the pipeline model-agnostic.

**Alternative.** scikit-learn's `LogisticRegression`. It is trained as an
*optional sanity check* (decision 13) to confirm the PyTorch version behaves
like a standard implementation.

## 6. Loss: `BCEWithLogitsLoss`, no `pos_weight`

**Decision.** Both models are trained with `nn.BCEWithLogitsLoss`.

**Why.** Binary cross-entropy, `-[y log p + (1 - y) log(1 - p)]` with
`p = sigmoid(logit)`, is the logistic regression cost function (the negative
log-likelihood of a Bernoulli model). `BCEWithLogitsLoss` fuses the sigmoid
and the cross-entropy into a single, numerically stable computation
(log-sum-exp), avoiding `log(0)` or overflow for large logits. That is why the
models output raw logits and the sigmoid is applied only when producing
probabilities ([`predict_proba`](../src/bcd/training/trainer.py)).

**Why no `pos_weight`.** Re-weighting the loss is another way to handle class
imbalance. Using it in Stage 1 would mix two imbalance remedies, and the
Stage 1 vs Stage 2 comparison would no longer isolate the effect of SMOTE.

**Alternative.** `nn.Sigmoid` + `nn.BCELoss`: mathematically identical but
numerically less stable.

## 7. ANN architecture: 30 → 16 → 8 → 1, ReLU, Dropout(0.2)

**Decision.** [`ann.py`](../src/bcd/models/ann.py): two hidden layers of 16
and 8 units, ReLU after each, `Dropout(0.2)` after each, one output logit.

**Why.**
- **Small**: 641 trainable parameters for 364 training samples. A large
  network on 569 samples would memorise the training data.
- **Two hidden layers** let the model learn non-linear feature interactions
  that logistic regression cannot represent; the decreasing widths compress
  30 inputs towards one decision.
- **ReLU**: gradient 1 for positive inputs, so gradients do not shrink
  layer after layer during backpropagation (the sigmoid's derivative is at
  most 0.25, which causes vanishing gradients in stacked layers).
- **One output unit** with a sigmoid is sufficient for a binary problem.

**Alternatives.** One hidden layer (also reasonable; two layers was chosen
to show the depth/non-linearity contrast with logistic regression while
staying small), wider layers (more overfitting risk), sigmoid/tanh hidden
activations (vanishing gradients).

## 8. Optimiser: Adam, learning rate 0.001, batch size 32

**Decision.** `torch.optim.Adam(lr=0.001)`; mini-batches of 32 from a
`DataLoader`; only the training loader is shuffled.

**Why.** Adam adapts a per-weight step size from running averages of the
gradient and its square, so it converges reliably without a learning-rate
schedule; 0.001 is its standard default. Batches of 32 give ~12 updates per
epoch on 364 samples: frequent enough to train quickly, with some gradient
noise that helps generalisation. Shuffling the training set each epoch
changes batch composition so successive gradient estimates are less
correlated; validation/test order does not matter because they are only
evaluated. The shuffling generator is seeded for reproducibility.

**Alternatives.** Plain SGD (needs learning-rate tuning and usually
momentum), full-batch gradient descent (fewer updates per epoch).

## 9. Regularisation: small model, early stopping, dropout, L2

**Decision.**
- ANN: `Dropout(0.2)` and `weight_decay=1e-4` (L2) in Adam.
- Logistic regression: no dropout, no weight decay (`weight_decay=0`).
- Both: early stopping (decision 10).

**Why.** Overfitting is the main risk with 364 training samples. The
remedies are layered: a small network limits capacity; dropout randomly
removes 20% of hidden activations in each training step so the network cannot
rely on single neurons; L2 penalises large weights, which favours smoother
decision functions; early stopping halts training before the network starts
fitting noise. Dropout is "light" (0.2 rather than 0.5) because with only 16
and 8 units, heavy dropout would cause underfitting. Logistic regression is
kept as a plain baseline: with 31 parameters it has little capacity to
overfit, and early stopping already limits how large its weights can grow.

**Alternatives.** Batch normalisation (unnecessary for such a small network
on standardised inputs), stronger dropout/L2 (only justified if the learning
curves showed overfitting).

## 10. Early stopping on validation loss, patience 30, restore best weights

**Decision.** Up to 500 epochs; after each epoch the validation loss is
measured; training stops after 30 epochs without a new minimum, and the
weights of the best epoch are restored.

**Why.**
- **Validation loss**, not training loss: training loss keeps decreasing as
  the model memorises; validation loss on unseen patients rises once the
  model starts overfitting. Loss is used instead of accuracy because it is
  smooth and reacts to changes in confidence, while accuracy on 91 samples
  moves in steps of about 1.1%.
- **Patience 30**: the validation curve is noisy with 91 samples and
  mini-batch updates; a short patience could stop on a random bump.
- **Restore best**: the last epoch is by construction 30 epochs past the
  best one; keeping the best weights is what makes early stopping a
  regulariser.
- **500 epochs** is a ceiling, not a target; it is high enough that early
  stopping, not the ceiling, decides when to stop.

**Measurement detail.** At the end of each epoch both the training loss and
the validation loss are recomputed in `model.eval()` mode (dropout off).
Averaging the mini-batch losses during the epoch would measure the training
loss with dropout on and with weights that changed during the epoch, making
the two curves not directly comparable.

## 11. Decision threshold 0.5, no tuning

**Decision.** Predict malignant when `sigmoid(logit) >= 0.5`, for both
models and both stages.

**Why.** Lowering the threshold would raise recall on its own and would
hide the effect of the model or of SMOTE. With a fixed threshold, any change
in recall between Stage 1 and Stage 2 is caused by SMOTE alone. Tuning a
threshold would also require a separate set of data to tune it on (never the
test set). ROC-AUC is reported to show the threshold-independent quality.

## 12. Stage 2: SMOTE, k = 5, 1:1, on the scaled training split only

**Decision.** [`augmentation.py`](../src/bcd/data/augmentation.py):
`imblearn.over_sampling.SMOTE(k_neighbors=5, sampling_strategy=1.0,
random_state=42)` applied to the **scaled training split only**, after
scaling. Validation and test are never resampled. Everything else (split,
scaler, architectures, hyperparameters, seed) is identical to Stage 1.

**How SMOTE works.** For a malignant training sample, pick one of its 5
nearest malignant neighbours and create a synthetic sample at a random point
on the line segment between them. Repeat until there are as many malignant as
benign training samples.

**Why SMOTE.**
- **vs random oversampling** (duplicating malignant rows): duplicates add no
  new information and push the model, especially the ANN, to memorise those
  exact points (overfitting). SMOTE fills in the malignant region with new,
  plausible points.
- **vs class weights / `pos_weight`**: re-weighting changes the loss, not the
  data. It is a valid alternative, but the brief explicitly asks for a *data
  augmentation* technique, and SMOTE is the standard augmentation method for
  tabular data with real-valued features (all 30 features here are continuous,
  so interpolating between samples is meaningful).
- **vs undersampling the benign class**: would throw away real data from an
  already small dataset.

**Why after the split, on training only.** Synthetic samples are built from
real ones. SMOTE before splitting could create training points interpolated
from test patients (leakage). Validation and test must keep the real class
mix, otherwise early stopping and the reported metrics would describe an
artificial population. A test verifies validation and test are unchanged.

**Why after scaling.** SMOTE's nearest neighbours use Euclidean distance;
without scaling, the distance would be dominated by a few large-valued
features.

**Why k = 5 and 1:1.** k = 5 is the standard default: enough neighbours to
add variety, few enough to stay inside the local malignant region. 1:1
removes the imbalance entirely, which gives the clearest test of whether
balancing helps.

**Train metrics in Stage 2.** Train metrics are computed on the 364 real
training samples (not the synthetic ones), so the train/test gap measures
generalisation on real patients and is comparable to Stage 1. The training
loss curve in Stage 2 is measured on the SMOTE set, because that is the data
the model is fitted on.

## 13. Optional checks (reported separately)

1. **scikit-learn `LogisticRegression`** (L-BFGS, default `C = 1.0`) on the
   same scaled data, compared with the PyTorch logistic regression on the
   test set. Purpose: confirm the PyTorch model and training loop produce a
   standard logistic regression. It is not expected to be identical: it uses
   a different optimiser, runs to convergence without early stopping, and
   applies scikit-learn's default mild L2 penalty.
2. **Stratified 5-fold CV on the training split.** For each fold, the scaler,
   SMOTE and the model are refitted on the fold's training part; early
   stopping still uses the fixed validation split; the held-out fold is only
   scored. The test set is not touched. Purpose: show how much the metrics
   vary with a different subset of patients, since a single 114-sample test
   set gives a single, noisy estimate.

## 14. Reproducibility

- Seed 42 for `random`, numpy and torch, reset before **every** model is
  trained ([`utils.py`](../src/bcd/utils.py)), so no run depends on the runs
  before it. The splits, SMOTE and the DataLoader shuffling generator use
  the same seed.
- `torch.use_deterministic_algorithms(True, warn_only=True)`: deterministic
  kernels where they exist, a warning rather than a crash otherwise.
- `num_workers=0` in the DataLoaders (no worker processes).
- All hyperparameters in `config/config.yaml`, loaded into frozen
  dataclasses and validated at start-up ([`config.py`](../src/bcd/config.py)).
- Exact library versions are pinned in `requirements.txt`.
