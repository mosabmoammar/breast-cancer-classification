# Breast Cancer Diagnosis: Logistic Regression vs ANN, before and after SMOTE

Binary classification of breast tumours as **malignant** or **benign** from 30
cell-nucleus measurements (Breast Cancer Wisconsin (Diagnostic) dataset, 569
samples). Two models, a logistic regression baseline and a small Multilayer
Perceptron (ANN), are both implemented in PyTorch and trained by one shared
training loop:

- **Stage 1:** trained on the original (standardised) training data.
- **Stage 2:** trained on the training data rebalanced with **SMOTE**, with
  everything else identical.

Both stages are evaluated on the same held-out, stratified test set. Because a
missed cancer is far more costly than a false alarm, **malignant is the positive
class (label 1)** and **malignant recall** is the main metric. F1, ROC-AUC and
precision are reported next to it.

---

## Setup (Windows, PowerShell, Python 3.14)

From the project root:

```powershell
# 1. Confirm Python 3.14 is available
py -3.14 --version

# 2. Create and activate the virtual environment
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
# If activation is blocked:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install the dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Make the project package (src/bcd) importable
pip install -e . --no-deps

# 5. Register the venv as a Jupyter kernel for the notebook
python -m ipykernel install --user --name bcd-venv --display-name "Python 3.14 (bcd .venv)"
```

The extra index URL in step 3 is PyTorch's official wheel index. It provides the
**CPU-only** build of torch, which is much smaller than the default build and is
all this project needs.

## Dependencies

All versions are pinned in [`requirements.txt`](requirements.txt). All of them
provide Python 3.14 wheels for Windows.

| Library | Used for |
| --- | --- |
| numpy, pandas | arrays and result tables |
| scikit-learn | dataset, stratified split, `StandardScaler`, metrics |
| imbalanced-learn | SMOTE (Stage 2) |
| torch (CPU) | both models, `Dataset`/`DataLoader`, training loop |
| matplotlib, seaborn | figures |
| pyyaml | reading `config/config.yaml` |
| jupyter, ipykernel | the notebook |
| pytest | tests |

## How to run

Activate the venv first (`.venv\Scripts\Activate.ps1`).

**1. Full experiment from the command line.** This runs both stages for both
models, prints every metric, and writes all tables and figures to `outputs/`:

```powershell
python scripts/run_experiments.py                 # includes the optional checks
python scripts/run_experiments.py --skip-optional # required experiment only
```

**2. Notebook.** This runs the same experiment step by step, with explanations,
figures and the answers to the brief's questions:

```powershell
jupyter notebook notebooks/breast_cancer_classification.ipynb
```

Select the kernel **"Python 3.14 (bcd .venv)"**. To re-execute the notebook
non-interactively and save it with its outputs:

```powershell
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=bcd-venv notebooks/breast_cancer_classification.ipynb
```

**3. Tests:**

```powershell
python -m pytest -v
```

All settings (split ratios, seed, architecture, optimiser, early stopping,
SMOTE) live in [`config/config.yaml`](config/config.yaml). Each one is justified
in [`design/DECISIONS.md`](design/DECISIONS.md).

## Project structure

```text
breast-cancer-classification/
├── config/config.yaml            # every hyperparameter and setting
├── design/
│   └── DECISIONS.md              # every design decision and its justification
├── notebooks/breast_cancer_classification.ipynb
├── outputs/
│   ├── figures/                  # PNG figures
│   └── results/                  # CSV tables and classification reports
├── scripts/run_experiments.py    # CLI entry point
├── src/bcd/
│   ├── data/
│   │   ├── augmentation.py       # SMOTE on the scaled training split only
│   │   ├── dataset.py            # PyTorch Dataset + DataLoader
│   │   ├── loader.py             # load dataset, remap labels (1 = malignant)
│   │   ├── preprocessing.py      # StandardScaler fitted on train only
│   │   └── splitter.py           # stratified train / validation / test split
│   ├── evaluation/
│   │   ├── metrics.py            # metrics with malignant as positive class
│   │   └── plots.py              # all figures
│   ├── models/
│   │   ├── ann.py                  # 30 -> 16 -> 8 -> 1 MLP
│   │   └── logistic_regression.py  # nn.Linear(30, 1)
│   ├── pipeline/
│   │   ├── experiment.py         # Stage 1 + Stage 2 for both models, tables
│   │   └── optional_checks.py    # optional sklearn sanity check and k-fold CV
│   ├── training/trainer.py       # ONE shared training loop with early stopping
│   ├── config.py                 # loads and validates the config
│   └── utils.py                  # seeding / determinism
├── tests/                        # pytest suite
├── pyproject.toml                # makes src/bcd installable (pip install -e .)
├── README.md
└── requirements.txt              # pinned dependencies
```

## Results

All numbers below come from `python scripts/run_experiments.py` (full log in
[`outputs/run_log.txt`](outputs/run_log.txt)) and are identical in the executed
notebook. The run is deterministic: repeated runs give the same values.

**Data.** 569 samples (357 benign / 212 malignant). Stratified split:

| Subset | Benign | Malignant | Total |
| --- | --- | --- | --- |
| Train (Stage 1) | 228 | 136 | 364 |
| Train after SMOTE (Stage 2) | 228 | 228 | 456 |
| Validation | 57 | 34 | 91 |
| Test | 72 | 42 | 114 |

**Test-set metrics** (positive class = malignant, threshold 0.5):

| Model | Stage | Accuracy | Precision | Recall | F1 | ROC-AUC | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 1 (original) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9967 | 2 | 1 |
| ANN | 1 (original) | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9974 | 2 | 1 |
| Logistic Regression | 2 (SMOTE) | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9970 | 1 | 1 |
| ANN | 2 (SMOTE) | 0.9825 | 0.9762 | 0.9762 | 0.9762 | 0.9974 | 1 | 1 |

**Key findings.**
- **Accuracy hides the missed cancers.** In Stage 1 both models reach 97.4%
  accuracy, yet they miss 2 of 42 cancers (recall 0.952), against 1 false
  alarm out of 72 benign cases.
- **Logistic regression and the ANN make the same predictions on all 114 test
  patients in Stage 1.** The data is almost linearly separable (ROC-AUC ≈ 0.997
  for the linear model), so the ANN's extra capacity (641 vs 31 parameters)
  brings no measurable gain.
- **No overfitting in the final models.** The train/test gaps are small (about
  1 point in accuracy and at most 2.6 points in recall in Stage 1). The ANN's
  validation loss stops improving after epoch 48 while its training loss keeps
  falling; early stopping restores the epoch-48 weights.
- **SMOTE caught one more cancer at no extra false alarms.** Recall went from
  0.952 to 0.976 and F1 from 0.964 to 0.976, for both models. The recovered
  case was borderline (P(malignant) 0.45 → 0.57 / 0.71). ROC-AUC is unchanged,
  so SMOTE shifted the predicted probabilities towards malignant rather than
  improving class separation.
- **The SMOTE gain is small and not robust.** It amounts to one patient. The
  optional 5-fold CV on the training split shows no consistent recall
  improvement (LR 0.963 → 0.956, ANN 0.963 → 0.963, fold standard deviation
  0.03–0.05).

**Optional checks** (`outputs/results/optional_*.csv`):
- **scikit-learn `LogisticRegression`** agrees with the PyTorch logistic
  regression on 113 of 114 test patients (99.1%) in both stages. This confirms
  the PyTorch implementation behaves like a standard logistic regression.
- **Stratified 5-fold CV** gives mean recall 0.956–0.963 and ROC-AUC
  0.993–0.994 across models and stages. These values agree with the test-set
  results.

**Outputs.**
- `outputs/results/`:
  - `all_metrics.csv`: every model, stage, split and metric.
  - `stage_comparison.csv`: Stage 1 vs Stage 2.
  - `generalisation_gap.csv`, `training_summary.csv`, `class_distribution.csv`.
  - `misclassified_test_patients.csv`.
  - `classification_reports.txt`.
- `outputs/figures/`: class distribution, confusion matrices, ROC curves, loss
  curves (all runs and per stage), and the Stage 1 vs Stage 2 chart.

The full discussion of the brief's six questions is in section 7 of the
notebook.
