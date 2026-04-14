def summarize_bootstrap_coefficients(coef_matrix, gene_names, threshold_ratio=0.9):

    n_bootstrap, n_features = coef_matrix.shape

    features_sum = np.count_nonzero(coef_matrix, axis=0)
    freq_threshold = int(threshold_ratio * n_bootstrap)

    selected_indices = np.where(features_sum >= freq_threshold)[0]
    selected_features = [gene_names[i] for i in selected_indices]

    feature_counts = {
        gene_names[i]: int(features_sum[i])
        for i in range(n_features)
    }

    coefficients_sum = {
        gene_names[i]: float(coef_matrix[:, i][coef_matrix[:, i] != 0.0].sum())
        for i in range(n_features)
    }

    per_fold_values = {
        gene_names[i]: list(coef_matrix[:, i][coef_matrix[:, i] != 0.0])
        for i in range(n_features) if np.any(coef_matrix[:, i] != 0.0)
    }

    mean_coefs = coef_matrix.mean(axis=0)
    std_coefs = coef_matrix.std(axis=0)

    feature_coefs_mean = {
        gene_names[i]: mean_coefs[i]
        for i in range(n_features)
    }

    feature_coefs_std = {
        gene_names[i]: std_coefs[i]
        for i in range(n_features)
    }

    summary = {
        "selected_features": selected_features,
        "feature_counts": feature_counts,
        "coefficients_sum": coefficients_sum,
        "per_fold_values": per_fold_values,
        "feature_coefs_mean": feature_coefs_mean,
        "feature_coefs_std": feature_coefs_std,
        "coef_matrix": coef_matrix
    }

    return summary
