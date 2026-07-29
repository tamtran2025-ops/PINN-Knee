"""
Early-Cycle Feature Extraction for Knee-Point Prediction.

Extracts ~20 features from the first N cycles of battery capacity data.
Inspired by Severson et al. (2019) Nature Energy.

Features categories:
1. Statistical features of capacity
2. Degradation rate features
3. Variance-based features (Severson's key insight)
4. Exponential fit parameters
5. Capacity fade shape features
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis, linregress
from sklearn.preprocessing import StandardScaler


def extract_early_features(capacity, cycles, n_early):
    """
    Extract features from first n_early cycles.

    Parameters
    ----------
    capacity : array - full capacity data
    cycles : array - full cycle indices
    n_early : int - number of early cycles to use

    Returns
    -------
    features : dict with ~20 features
    """
    # Clip to first n_early cycles
    n = min(n_early, len(capacity))
    cap = capacity[:n].copy()
    cyc = cycles[:n].astype(float)

    if n < 5:
        return None

    features = {}

    # =====================================================================
    #   1. Statistical features of capacity
    # =====================================================================
    features['cap_mean'] = np.mean(cap)
    features['cap_std'] = np.std(cap)
    features['cap_min'] = np.min(cap)
    features['cap_max'] = np.max(cap)
    features['cap_range'] = np.max(cap) - np.min(cap)

    # =====================================================================
    #   2. Degradation rate features
    # =====================================================================
    # Linear fit: Q(n) = a*n + b
    slope, intercept, r_value, _, _ = linregress(cyc, cap)
    features['linear_slope'] = slope               # Degradation rate
    features['linear_intercept'] = intercept        # Initial capacity (fit)
    features['linear_r2'] = r_value ** 2            # How linear is the decay?

    # Quadratic fit: Q(n) = a*n^2 + b*n + c
    if n >= 5:
        coeffs = np.polyfit(cyc, cap, 2)
        features['quad_a'] = coeffs[0]              # Curvature
        features['quad_b'] = coeffs[1]              # Linear term
    else:
        features['quad_a'] = 0.0
        features['quad_b'] = slope

    # =====================================================================
    #   3. Variance-based features (Severson's key insight)
    # =====================================================================
    # Cycle-to-cycle differences
    dQ = np.diff(cap)

    features['dQ_mean'] = np.mean(dQ)
    features['dQ_std'] = np.std(dQ)
    features['dQ_min'] = np.min(dQ)

    # Log variance of dQ (Severson's most predictive feature)
    dQ_var = np.var(dQ)
    features['log_var_dQ'] = np.log10(dQ_var + 1e-15)

    # Skewness and kurtosis
    if n >= 10:
        features['dQ_skewness'] = skew(dQ)
        features['dQ_kurtosis'] = kurtosis(dQ)
    else:
        features['dQ_skewness'] = 0.0
        features['dQ_kurtosis'] = 0.0

    # =====================================================================
    #   4. Capacity fade shape features
    # =====================================================================
    # Total capacity drop
    features['total_drop'] = cap[0] - cap[-1]

    # First half vs second half drop ratio
    mid = n // 2
    if mid > 1 and (n - mid) > 1:
        drop_first = cap[0] - cap[mid]
        drop_second = cap[mid] - cap[-1]
        features['drop_ratio'] = drop_second / (drop_first + 1e-10)
    else:
        features['drop_ratio'] = 1.0

    # Capacity at specific fractions of n_early
    features['cap_start'] = cap[0]
    features['cap_end'] = cap[-1]
    features['cap_mid'] = cap[mid] if mid < n else cap[-1]

    # =====================================================================
    #   5. Exponential fit features
    # =====================================================================
    try:
        def exp_model(x, a, b, c):
            return a * np.exp(-b * x) + c

        # Normalize cycles for fitting
        cyc_norm = (cyc - cyc[0]) / (cyc[-1] - cyc[0] + 1e-10)

        popt, _ = curve_fit(exp_model, cyc_norm, cap,
                            p0=[cap[0] * 0.1, 1.0, cap[-1]],
                            maxfev=2000, bounds=([0, 0, 0], [5, 50, 5]))
        features['exp_a'] = popt[0]     # Amplitude
        features['exp_b'] = popt[1]     # Decay rate
        features['exp_c'] = popt[2]     # Offset
    except Exception:
        features['exp_a'] = 0.0
        features['exp_b'] = 0.0
        features['exp_c'] = np.mean(cap)

    return features


def build_feature_matrix(cells, n_early):
    """
    Build feature matrix X and target vector y from list of cells.

    Parameters
    ----------
    cells : list of cell dicts (must have 'knee_cycle' key)
    n_early : int - number of early cycles

    Returns
    -------
    X : ndarray (n_cells, n_features)
    y : ndarray (n_cells,) - knee cycle targets
    feature_names : list of str
    valid_indices : list of int - indices of valid cells
    """
    all_features = []
    all_knees = []
    valid_indices = []
    feature_names = None

    for i, cell in enumerate(cells):
        # Skip cells with too few cycles
        if len(cell['capacity']) < n_early:
            continue

        # Skip cells without knee
        if cell.get('knee_cycle') is None:
            continue

        # Skip cells where knee occurs before n_early
        if cell['knee_cycle'] <= n_early:
            continue

        feats = extract_early_features(cell['capacity'], cell['cycles'], n_early)
        if feats is None:
            continue

        if feature_names is None:
            feature_names = sorted(feats.keys())

        feat_vec = [feats[name] for name in feature_names]
        all_features.append(feat_vec)
        all_knees.append(cell['knee_cycle'])
        valid_indices.append(i)

    if len(all_features) == 0:
        return np.array([]), np.array([]), [], []

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_knees, dtype=np.float32)

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names, valid_indices


def normalize_features(X_train, X_test=None, X_cal=None):
    """
    Normalize features using StandardScaler.

    Returns
    -------
    X_train_norm, X_test_norm, X_cal_norm, scaler
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    X_test_norm = scaler.transform(X_test) if X_test is not None else None
    X_cal_norm = scaler.transform(X_cal) if X_cal is not None else None

    return X_train_norm, X_test_norm, X_cal_norm, scaler


if __name__ == '__main__':
    """Test feature extraction on synthetic data."""
    print("Testing feature extraction...")

    np.random.seed(42)
    cycles = np.arange(1, 501)
    capacity = 1.1 - 0.0002 * cycles + np.random.normal(0, 0.002, 500)

    for n_early in [20, 50, 100]:
        feats = extract_early_features(capacity, cycles, n_early)
        print(f"\n  n_early={n_early}: {len(feats)} features")
        for k, v in sorted(feats.items()):
            print(f"    {k:25s}: {v:.6f}")

    print("\n  Done!")
