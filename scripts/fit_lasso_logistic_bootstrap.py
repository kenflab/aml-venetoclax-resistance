def fit_lasso_logistic_bootstrap(
    X, 
    y, 
    gene_names,
    n_bootstrap=1000, 
    epsilon=0.01,
    n_jobs=-1,
    random_state=42,
    params=None,
    noise_std=0.001
    ):
    if params is None:
        params = {}

    logging.info(f"Bootstrap method start (Number of trials: {n_bootstrap})")
    n_samples, n_features = X.shape
    
    rng = np.random.RandomState(random_state)

    features_sum = np.zeros(n_features, dtype=int)
    coef_matrix = np.zeros((n_bootstrap, n_features))
    oob_indices_list = []

    seeds = [rng.randint(0, 10**9) for _ in range(n_bootstrap)]

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(run_bootstrap_iteration)(
            seed, X, y, epsilon, params, noise_std
        ) for seed in seeds
    )

    for i, (res_mask, coef, oob_idx) in enumerate(results):
        features_sum += res_mask.astype(int)
        coef_matrix[i] = coef            
        oob_indices_list.append(oob_idx) 

    return coef_matrix, oob_indices_list
    
