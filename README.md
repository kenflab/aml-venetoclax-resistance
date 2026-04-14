# AML Venetoclax Resistance

Code and resources to support the main bootstrap-based L1-logistic-regression analysis used in this study.

---

## Overview

This repository contains scripts and example data for the bootstrap-based feature selection analysis described in the paper.  
Its purpose is to improve transparency and reproducibility of the main computational results.

---

## Repository Contents

### `run_bootstrap_iteration.py`
Runs a single bootstrap iteration of **L1-regularized logistic regression**.

This step:
- resamples the input data with replacement,
- adds small Gaussian noise,
- fits an L1-penalized logistic regression model,
- returns the selected-feature mask (`|coef| > epsilon`), coefficients, and out-of-bag (OOB) sample indices.

### `fit_lasso_logistic_bootstrap.py`
Runs repeated bootstrap iterations in parallel.

This step:
- performs `n_bootstrap` iterations,
- collects coefficients across iterations,
- returns a coefficient matrix of shape `(n_bootstrap × n_features)` and the OOB index list.

### `summarize_bootstrap_coefficients.py`
Summarizes bootstrap results across iterations.

This step:
- computes feature-selection frequency,
- calculates coefficient means and standard deviations,
- identifies features passing a user-defined selection threshold,
- returns summary statistics for downstream interpretation.

---

## Analysis Outline

The core workflow is:

1. Prepare a processed feature matrix `X` and binary label vector `y`.
2. Run bootstrap-based L1-logistic regression across repeated resamples.
3. Summarize selection frequency and coefficient stability across bootstrap iterations.
4. Identify robustly selected features.

---

## Example Usage

```python
gene_names = X.columns
n_bootstrap = 10000
epsilon = 0.01
random_state = 2025
n_jobs = -1
threshold_ratio = 0.8

params = {
    "C": 10,
    "class_weight": "balanced",
    "max_iter": 10000,
}

coef_matrix, oob_indices = fit_lasso_logistic_bootstrap(
    X.values,
    y,
    gene_names=gene_names,
    n_bootstrap=n_bootstrap,
    epsilon=epsilon,
    n_jobs=n_jobs,
    random_state=random_state,
    params=params,
)

summary = summarize_bootstrap_coefficients(
    coef_matrix=coef_matrix,
    gene_names=gene_names,
    threshold_ratio=threshold_ratio,
)
```

#### Input Data Assumptions

- **X**: A numeric matrix where  
  - Rows represent **samples**  
  - Columns represent **features (genes)**  
  - Values are **Transcripts Per Million (TPM)**, then **log-transformed**, and **Z-score standardized** per gene across samples.

- **y**: A binary list or array of labels (`0` and `1`), corresponding to the sample classes.

---
#### Example Data
This repository includes synthetic example data for demonstration and code verification only.

- The example data were generated using NumPy random numbers with a fixed seed (42).
- No real patient or biological data are included in this repository.

- Files are located in the `data/` folder:
  - `data/X_sample.csv` — Processed feature matrix (rows = samples, columns = genes).  
    Values are TPM-normalized, then log-transformed, and **Z-score standardized per gene across samples**.
  - `data/y_sample.csv` — Labels (`sample_id`, `label`) with binary classes `0/1`.

**Shapes**
- `X_sample.csv`: 30 × 50
- `y_sample.csv`: 30 × 2 (`sample_id`, `label`)

---

#### Quick Load Example

```python
import pandas as pd

# Load X (processed) with sample IDs as index
X = pd.read_csv("data/X_sample.csv", index_col=0)

# Load y and align to X.index
y_df = pd.read_csv("data/y_sample.csv")  # columns: sample_id, label
y = y_df.set_index("sample_id").loc[X.index, "label"].values

print(X.shape, y.shape)
print(X.head(3))
print(y[:10])
```
print(y[:10])

---

#### Notes
This repository is intended to support the analysis presented in the manuscript.
It is not packaged as a general-purpose software tool, but rather as a focused, minimal resource for reproducing the main bootstrap-based feature selection workflow.

For the original study data, please refer to the manuscript and its data availability statement.
