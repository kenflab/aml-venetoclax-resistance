from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import logging
logging.basicConfig(level=logging.INFO)

def run_bootstrap_iteration(seed, X, y, epsilon, params, noise_std):

    local_rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    
    bootstrap_idx = local_rng.choice(np.arange(n_samples), size=n_samples, replace=True)

    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[bootstrap_idx] = False
    oob_idx = np.where(oob_mask)[0]

    X_boot = X[bootstrap_idx]
    y_boot = y[bootstrap_idx]

    noise = local_rng.normal(loc=0.0, scale=noise_std, size=X_boot.shape)
    X_boot += noise

    
    # Lasso
    clf = LogisticRegression(
        penalty='l1',
        solver='saga',
        random_state=seed,
        **params
    )
    
    clf.fit(X_boot, y_boot)
    
    coef = clf.coef_[0]  
    selected_mask = np.abs(coef) > epsilon

    return selected_mask, coef, oob_idx
