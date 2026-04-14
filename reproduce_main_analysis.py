import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

sys.path.append(str(SCRIPTS_DIR))

from fit_lasso_logistic_bootstrap import fit_lasso_logistic_bootstrap
from summarize_bootstrap_coefficients import summarize_bootstrap_coefficients


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    x_path = DATA_DIR / "X_sample.csv"
    y_path = DATA_DIR / "y_sample.csv"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing input file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing input file: {y_path}")

    # Load processed feature matrix
    X = pd.read_csv(x_path, index_col=0)

    # Load labels and align to X index
    y_df = pd.read_csv(y_path)
    required_cols = {"sample_id", "label"}
    if not required_cols.issubset(y_df.columns):
        raise ValueError(
            f"{y_path.name} must contain columns: {sorted(required_cols)}"
        )

    y = y_df.set_index("sample_id").loc[X.index, "label"].values
    gene_names = X.columns.tolist()

    # Analysis parameters
    n_bootstrap = 1000
    epsilon = 0.01
    random_state = 2025
    n_jobs = -1
    threshold_ratio = 0.8

    params = {
        "C": 10,
        "class_weight": "balanced",
        "max_iter": 10000,
    }

    print("Running bootstrap-based L1-logistic-regression analysis...")
    print(f"Input X shape: {X.shape}")
    print(f"Input y length: {len(y)}")

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

    # Save outputs
    coef_df = pd.DataFrame(coef_matrix, columns=gene_names)
    coef_df.to_csv(RESULTS_DIR / "bootstrap_coefficients.csv", index=False)

    feature_summary_df = pd.DataFrame(
        {
            "gene": gene_names,
            "selection_count": [
                summary["feature_counts"][g] for g in gene_names
            ],
            "coef_mean": [
                summary["feature_coefs_mean"][g] for g in gene_names
            ],
            "coef_std": [
                summary["feature_coefs_std"][g] for g in gene_names
            ],
            "coef_sum_nonzero": [
                summary["coefficients_sum"][g] for g in gene_names
            ],
        }
    ).sort_values(["selection_count", "coef_mean"], ascending=[False, False])

    feature_summary_df.to_csv(
        RESULTS_DIR / "bootstrap_feature_summary.csv", index=False
    )

    with open(RESULTS_DIR / "selected_features.json", "w") as f:
        json.dump(summary["selected_features"], f, indent=2)

    with open(RESULTS_DIR / "analysis_metadata.json", "w") as f:
        json.dump(
            {
                "n_bootstrap": n_bootstrap,
                "epsilon": epsilon,
                "random_state": random_state,
                "n_jobs": n_jobs,
                "threshold_ratio": threshold_ratio,
                "params": params,
                "input_shape": list(X.shape),
                "n_selected_features": len(summary["selected_features"]),
            },
            f,
            indent=2,
        )

    print("Analysis completed.")
    print(f"Selected features: {len(summary['selected_features'])}")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
