"""
Knee-Point Detection Algorithms for Battery Degradation Curves.

Implements three methods:
1. Bacon-Watts changepoint model
2. Maximum curvature method
3. Second derivative inflection method
4. Ensemble (median of all three)

Reference: Fermn-Cueto et al. (2020), Diao et al. (2019)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import UnivariateSpline

from config import KNEE_SMOOTH_WINDOW, KNEE_MIN_CYCLE, KNEE_MAX_FRACTION, KNEE_ACCELERATION_RATIO


def _smooth_capacity(capacity, window=KNEE_SMOOTH_WINDOW):
    """Smooth capacity data using Savitzky-Golay filter."""
    if len(capacity) < window:
        window = max(5, len(capacity) // 2 * 2 - 1)  # Ensure odd
    if window % 2 == 0:
        window += 1
    if len(capacity) <= window:
        return capacity.copy()
    return savgol_filter(capacity, window, polyorder=3)


# ==========================================================================
#   Method 1: Bacon-Watts Changepoint Model
# ==========================================================================

def detect_knee_bacon_watts(cycles, capacity, smooth_window=KNEE_SMOOTH_WINDOW):
    """
    Bacon-Watts changepoint detection.

    Fits: Q(n) = alpha0 + alpha1*(n - n_knee) + alpha2*(n - n_knee)*tanh((n - n_knee)/gamma)

    Parameters
    ----------
    cycles : array - cycle indices
    capacity : array - discharge capacity

    Returns
    -------
    knee_cycle : int or None
    confidence_score : float (R^2 of fit)
    params : dict with fitted parameters
    """
    cap_smooth = _smooth_capacity(capacity, smooth_window)
    n = cycles.astype(float)

    # Normalize for numerical stability
    n_min, n_max = n.min(), n.max()
    n_norm = (n - n_min) / (n_max - n_min + 1e-10)
    q_min, q_max = cap_smooth.min(), cap_smooth.max()
    q_norm = (cap_smooth - q_min) / (q_max - q_min + 1e-10)

    def bacon_watts_model(params, x):
        alpha0, alpha1, alpha2, n_knee, gamma = params
        gamma = max(gamma, 0.01)
        z = (x - n_knee) / gamma
        z = np.clip(z, -10, 10)
        return alpha0 + alpha1 * (x - n_knee) + alpha2 * (x - n_knee) * np.tanh(z)

    def objective(params):
        pred = bacon_watts_model(params, n_norm)
        return np.sum((q_norm - pred) ** 2)

    # Bounds for parameters
    bounds = [
        (0.3, 1.0),      # alpha0: capacity at knee
        (-2.0, 0.0),     # alpha1: average slope (negative)
        (-2.0, 0.0),     # alpha2: slope change (negative)
        (0.1, 0.9),      # n_knee: normalized knee location
        (0.01, 0.3),     # gamma: transition sharpness
    ]

    try:
        result = differential_evolution(objective, bounds, seed=42,
                                         maxiter=500, tol=1e-8, polish=True)
        params_opt = result.x
        n_knee_norm = params_opt[3]
        knee_cycle = int(n_min + n_knee_norm * (n_max - n_min))

        # Confidence: R^2
        pred = bacon_watts_model(params_opt, n_norm)
        ss_res = np.sum((q_norm - pred) ** 2)
        ss_tot = np.sum((q_norm - np.mean(q_norm)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return knee_cycle, r2, {
            'alpha0': params_opt[0], 'alpha1': params_opt[1],
            'alpha2': params_opt[2], 'n_knee': params_opt[3],
            'gamma': params_opt[4]
        }
    except Exception:
        return None, 0.0, {}


# ==========================================================================
#   Method 2: Maximum Curvature
# ==========================================================================

def detect_knee_curvature(cycles, capacity, smooth_window=KNEE_SMOOTH_WINDOW):
    """
    Detect knee-point as the cycle with maximum curvature.

    Curvature: kappa = |f''| / (1 + f'^2)^(3/2)

    Returns
    -------
    knee_cycle : int or None
    max_curvature : float
    """
    cap_smooth = _smooth_capacity(capacity, smooth_window)

    if len(cap_smooth) < 10:
        return None, 0.0

    # Compute derivatives via finite differences
    dn = np.diff(cycles.astype(float))
    dn[dn == 0] = 1.0

    dq = np.diff(cap_smooth) / dn
    d2q = np.diff(dq) / dn[:-1]

    # Curvature at interior points
    dq_mid = dq[:-1]
    curvature = np.abs(d2q) / (1 + dq_mid ** 2) ** 1.5

    # Only consider cycles after KNEE_MIN_CYCLE
    valid_start = max(0, KNEE_MIN_CYCLE - int(cycles[0]))
    valid_end = len(curvature)

    if valid_start >= valid_end:
        return None, 0.0

    # Find maximum curvature in valid range
    curvature_valid = curvature[valid_start:valid_end]
    max_idx = np.argmax(curvature_valid) + valid_start
    knee_cycle = int(cycles[max_idx + 1])  # +1 because of double diff

    return knee_cycle, float(curvature[max_idx])


# ==========================================================================
#   Method 3: Second Derivative (Acceleration)
# ==========================================================================

def detect_knee_second_derivative(cycles, capacity, smooth_window=KNEE_SMOOTH_WINDOW):
    """
    Detect knee-point as the cycle with maximum negative second derivative
    (maximum acceleration of capacity fade).

    Returns
    -------
    knee_cycle : int or None
    max_d2q : float (most negative value)
    """
    cap_smooth = _smooth_capacity(capacity, smooth_window)

    if len(cap_smooth) < 10:
        return None, 0.0

    # Compute second derivative
    dn = np.diff(cycles.astype(float))
    dn[dn == 0] = 1.0

    dq = np.diff(cap_smooth) / dn
    d2q = np.diff(dq) / dn[:-1]

    # Only consider after KNEE_MIN_CYCLE
    valid_start = max(0, KNEE_MIN_CYCLE - int(cycles[0]))
    valid_end = len(d2q)

    if valid_start >= valid_end:
        return None, 0.0

    d2q_valid = d2q[valid_start:valid_end]

    # Most negative d2q = maximum acceleration of fade
    min_idx = np.argmin(d2q_valid) + valid_start
    knee_cycle = int(cycles[min_idx + 1])

    return knee_cycle, float(d2q[min_idx])


# ==========================================================================
#   Ensemble Method (Median of All Three)
# ==========================================================================

def detect_knee_ensemble(cycles, capacity, smooth_window=KNEE_SMOOTH_WINDOW):
    """
    Ensemble knee detection: run all three methods, return median.

    Returns
    -------
    knee_cycle : int or None
    per_method : dict with each method's result
    agreement_score : float (0-1, based on std of estimates)
    """
    results = {}

    # Method 1: Bacon-Watts
    bw_cycle, bw_conf, bw_params = detect_knee_bacon_watts(cycles, capacity, smooth_window)
    if bw_cycle is not None:
        results['bacon_watts'] = bw_cycle

    # Method 2: Curvature
    curv_cycle, curv_val = detect_knee_curvature(cycles, capacity, smooth_window)
    if curv_cycle is not None:
        results['curvature'] = curv_cycle

    # Method 3: Second derivative
    d2_cycle, d2_val = detect_knee_second_derivative(cycles, capacity, smooth_window)
    if d2_cycle is not None:
        results['second_derivative'] = d2_cycle

    if len(results) == 0:
        return None, results, 0.0

    # Median of detected knees
    knee_values = list(results.values())
    knee_cycle = int(np.median(knee_values))

    # Agreement score: 1 - normalized std
    if len(knee_values) >= 2:
        std = np.std(knee_values)
        max_range = max(cycles) - min(cycles)
        agreement = max(0, 1.0 - std / (max_range + 1e-10))
    else:
        agreement = 0.5

    return knee_cycle, results, agreement


# ==========================================================================
#   Validation
# ==========================================================================

def validate_knee_point(cycles, capacity, knee_cycle,
                         min_acceleration_ratio=KNEE_ACCELERATION_RATIO):
    """
    Validate that detected knee is physically plausible.

    Checks:
    1. Knee cycle is within valid range
    2. Degradation rate after knee > ratio * rate before knee
    3. Enough data exists on both sides of knee

    Returns
    -------
    is_valid : bool
    diagnostics : dict
    """
    cap_smooth = _smooth_capacity(capacity)
    total_life = len(cycles)

    diagnostics = {
        'knee_cycle': knee_cycle,
        'total_cycles': total_life,
        'knee_fraction': knee_cycle / (max(cycles) + 1e-10),
    }

    # Check 1: Valid range
    if knee_cycle < KNEE_MIN_CYCLE:
        diagnostics['reason'] = 'too_early'
        return False, diagnostics

    if knee_cycle > max(cycles) * KNEE_MAX_FRACTION:
        diagnostics['reason'] = 'too_late'
        return False, diagnostics

    # Check 2: Enough data on both sides (at least 10 cycles before, 5 after)
    knee_idx = np.searchsorted(cycles, knee_cycle)
    if knee_idx < 10 or (total_life - knee_idx) < 5:
        diagnostics['reason'] = 'insufficient_data'
        return False, diagnostics

    # Check 3: Acceleration ratio
    # Rate before knee (last 50% of pre-knee data)
    pre_start = max(0, knee_idx // 2)
    pre_rate = abs(cap_smooth[knee_idx] - cap_smooth[pre_start]) / (knee_idx - pre_start + 1e-10)

    # Rate after knee (first 50% of post-knee data)
    post_end = min(total_life - 1, knee_idx + (total_life - knee_idx) // 2)
    post_rate = abs(cap_smooth[post_end] - cap_smooth[knee_idx]) / (post_end - knee_idx + 1e-10)

    ratio = post_rate / (pre_rate + 1e-10)
    diagnostics['pre_knee_rate'] = pre_rate
    diagnostics['post_knee_rate'] = post_rate
    diagnostics['acceleration_ratio'] = ratio

    if ratio < min_acceleration_ratio:
        diagnostics['reason'] = 'insufficient_acceleration'
        return False, diagnostics

    diagnostics['reason'] = 'valid'
    return True, diagnostics


# ==========================================================================
#   Master Dispatcher
# ==========================================================================

def detect_knee_point(cycles, capacity, method='ensemble'):
    """
    Detect knee-point using specified method.

    Parameters
    ----------
    cycles : array
    capacity : array
    method : str, one of 'bacon_watts', 'curvature', 'second_derivative', 'ensemble'

    Returns
    -------
    knee_cycle : int or None
    """
    if method == 'bacon_watts':
        knee, _, _ = detect_knee_bacon_watts(cycles, capacity)
    elif method == 'curvature':
        knee, _ = detect_knee_curvature(cycles, capacity)
    elif method == 'second_derivative':
        knee, _ = detect_knee_second_derivative(cycles, capacity)
    elif method == 'ensemble':
        knee, _, _ = detect_knee_ensemble(cycles, capacity)
    else:
        raise ValueError(f"Unknown method: {method}")

    return knee


if __name__ == '__main__':
    """Test knee detection on a synthetic curve with known knee."""
    print("Testing knee detection on synthetic data...")

    # Create synthetic battery curve with knee at cycle 500
    cycles = np.arange(1, 1001)
    np.random.seed(42)

    # Pre-knee: slow linear decay
    cap_pre = 1.1 - 0.0001 * cycles
    # Post-knee: accelerated decay
    knee_true = 500
    cap_post = np.where(cycles > knee_true,
                         cap_pre - 0.001 * (cycles - knee_true) ** 1.2,
                         0)
    capacity = np.where(cycles <= knee_true, cap_pre, cap_pre + cap_post)
    capacity += np.random.normal(0, 0.002, len(capacity))

    print(f"\n  True knee: cycle {knee_true}")

    # Test all methods
    for method in ['bacon_watts', 'curvature', 'second_derivative', 'ensemble']:
        knee = detect_knee_point(cycles, capacity, method=method)
        error = abs(knee - knee_true) if knee else 'N/A'
        print(f"  {method:25s}: knee={knee}, error={error} cycles")

    # Validate
    knee = detect_knee_point(cycles, capacity, method='ensemble')
    is_valid, diag = validate_knee_point(cycles, capacity, knee)
    print(f"\n  Validation: {'PASS' if is_valid else 'FAIL'}")
    print(f"  Acceleration ratio: {diag.get('acceleration_ratio', 'N/A'):.2f}")
    print("\n  Done!")
