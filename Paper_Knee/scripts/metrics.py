"""
Knee-point specific evaluation metrics for battery degradation prediction.

Paper: Physics-Constrained Deep Learning with Conformal Prediction
       for Early Knee-Point Prediction in Lithium-Ion Batteries.

Metrics cover:
  1. Point prediction accuracy (MAE, RMSE, MAPE, etc.)
  2. Uncertainty quantification quality (PICP, MPIW, CWC, interval score)
  3. Comprehensive evaluation via `evaluate_knee_predictions()`
"""

import numpy as np
from typing import Dict, Optional, Union

ArrayLike = Union[np.ndarray, list]


# =============================================================================
#   1. Point Prediction Metrics
# =============================================================================

def knee_mae(pred: ArrayLike, true: ArrayLike) -> float:
    """Mean Absolute Error in cycles.

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).

    Returns
    -------
    float
        MAE in cycles. Returns 0.0 for empty inputs.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - true)))


def knee_rmse(pred: ArrayLike, true: ArrayLike) -> float:
    """Root Mean Squared Error in cycles.

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).

    Returns
    -------
    float
        RMSE in cycles. Returns 0.0 for empty inputs.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def knee_mape(pred: ArrayLike, true: ArrayLike) -> float:
    """Mean Absolute Percentage Error (%).

    Entries where true == 0 are excluded from computation to avoid
    division by zero.

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).

    Returns
    -------
    float
        MAPE as a percentage (e.g. 5.2 means 5.2%). Returns 0.0 when
        no valid entries exist.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    mask = true != 0.0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100.0)


def knee_median_ae(pred: ArrayLike, true: ArrayLike) -> float:
    """Median Absolute Error in cycles (robust to outliers).

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).

    Returns
    -------
    float
        Median absolute error in cycles. Returns 0.0 for empty inputs.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    return float(np.median(np.abs(pred - true)))


def knee_within_tolerance(pred: ArrayLike, true: ArrayLike,
                          tol: float = 50.0) -> float:
    """Fraction of predictions within +/- tol cycles of truth.

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).
    tol : float, default=50
        Tolerance in cycles.

    Returns
    -------
    float
        Fraction in [0, 1]. Returns 0.0 for empty inputs.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - true) <= tol))


def knee_bias(pred: ArrayLike, true: ArrayLike) -> float:
    """Mean signed error (pred - true).

    Positive bias means the model predicts later knee-points on average;
    negative bias means earlier predictions.

    Parameters
    ----------
    pred : array-like
        Predicted knee-point cycles, shape (n,).
    true : array-like
        Ground-truth knee-point cycles, shape (n,).

    Returns
    -------
    float
        Mean signed error in cycles. Returns 0.0 for empty inputs.
    """
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if pred.size == 0 or true.size == 0:
        return 0.0
    return float(np.mean(pred - true))


# =============================================================================
#   2. Uncertainty Quantification Metrics
# =============================================================================

def knee_picp(true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of true knee-point values that fall within [lower, upper].

    Parameters
    ----------
    true : array-like
        Ground-truth knee-point cycles, shape (n,).
    lower : array-like
        Lower bound of prediction interval, shape (n,).
    upper : array-like
        Upper bound of prediction interval, shape (n,).

    Returns
    -------
    float
        Coverage probability in [0, 1]. Returns 0.0 for empty inputs.
    """
    true = np.asarray(true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if true.size == 0:
        return 0.0
    covered = (true >= lower) & (true <= upper)
    return float(np.mean(covered))


def knee_mpiw(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean Prediction Interval Width in cycles.

    Parameters
    ----------
    lower : array-like
        Lower bound of prediction interval, shape (n,).
    upper : array-like
        Upper bound of prediction interval, shape (n,).

    Returns
    -------
    float
        Mean interval width in cycles. Returns 0.0 for empty inputs.
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.size == 0:
        return 0.0
    return float(np.mean(upper - lower))


def knee_nmpiw(lower: ArrayLike, upper: ArrayLike,
               y_range: float) -> float:
    """Normalized Mean Prediction Interval Width.

    MPIW divided by the range of observed knee-point values, giving
    a dimensionless measure of interval sharpness.

    Parameters
    ----------
    lower : array-like
        Lower bound of prediction interval, shape (n,).
    upper : array-like
        Upper bound of prediction interval, shape (n,).
    y_range : float
        Range of ground-truth knee-point cycles (max - min). Must be > 0.

    Returns
    -------
    float
        Normalized MPIW. Returns 0.0 if y_range <= 0 or inputs are empty.
    """
    if y_range <= 0:
        return 0.0
    mpiw = knee_mpiw(lower, upper)
    return mpiw / y_range


def knee_cwc(true: ArrayLike, lower: ArrayLike, upper: ArrayLike,
             alpha: float = 0.10, eta: float = 50.0) -> float:
    """Coverage Width-based Criterion.

    CWC = MPIW * (1 + gamma * exp(-eta * (PICP - (1-alpha))))

    where gamma = 1 if PICP < (1-alpha), else 0. This penalizes
    intervals that fail to meet the target coverage.

    Parameters
    ----------
    true : array-like
        Ground-truth knee-point cycles, shape (n,).
    lower : array-like
        Lower bound of prediction interval, shape (n,).
    upper : array-like
        Upper bound of prediction interval, shape (n,).
    alpha : float, default=0.10
        Significance level (1-alpha = target coverage).
    eta : float, default=50.0
        Penalty factor for under-coverage.

    Returns
    -------
    float
        CWC value (lower is better when coverage is met). Returns 0.0
        for empty inputs.
    """
    true = np.asarray(true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if true.size == 0:
        return 0.0

    picp = knee_picp(true, lower, upper)
    mpiw = knee_mpiw(lower, upper)
    target = 1.0 - alpha

    if picp < target:
        penalty = 1.0 + np.exp(-eta * (picp - target))
    else:
        penalty = 1.0

    return float(mpiw * penalty)


def knee_interval_score(true: ArrayLike, lower: ArrayLike,
                        upper: ArrayLike, alpha: float = 0.10) -> float:
    """Winkler interval score (average).

    For each sample i:
      S_i = (upper_i - lower_i)
            + (2/alpha) * (lower_i - true_i) * 1[true_i < lower_i]
            + (2/alpha) * (true_i - upper_i) * 1[true_i > upper_i]

    A proper scoring rule that rewards narrow intervals while penalizing
    missed coverage.

    Parameters
    ----------
    true : array-like
        Ground-truth knee-point cycles, shape (n,).
    lower : array-like
        Lower bound of prediction interval, shape (n,).
    upper : array-like
        Upper bound of prediction interval, shape (n,).
    alpha : float, default=0.10
        Significance level.

    Returns
    -------
    float
        Mean interval score (lower is better). Returns 0.0 for empty inputs.
    """
    true = np.asarray(true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if true.size == 0:
        return 0.0

    width = upper - lower
    below = np.maximum(lower - true, 0.0)
    above = np.maximum(true - upper, 0.0)
    score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return float(np.mean(score))


# =============================================================================
#   3. Comprehensive Evaluator
# =============================================================================

def evaluate_knee_predictions(
    true_knees: ArrayLike,
    pred_knees: ArrayLike,
    lower: Optional[ArrayLike] = None,
    upper: Optional[ArrayLike] = None,
    pred_std: Optional[ArrayLike] = None,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Evaluate knee-point predictions with all available metrics.

    Computes point-prediction metrics unconditionally. If prediction
    intervals (lower, upper) are provided, UQ metrics are also included.

    Parameters
    ----------
    true_knees : array-like
        Ground-truth knee-point cycles, shape (n,).
    pred_knees : array-like
        Predicted knee-point cycles, shape (n,).
    lower : array-like or None
        Lower bound of prediction interval, shape (n,).
    upper : array-like or None
        Upper bound of prediction interval, shape (n,).
    pred_std : array-like or None
        Predicted standard deviation per sample, shape (n,). Used to
        compute mean predictive uncertainty.
    alpha : float, default=0.10
        Significance level for UQ metrics.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    true_knees = np.asarray(true_knees, dtype=np.float64)
    pred_knees = np.asarray(pred_knees, dtype=np.float64)

    # GUARD(H7): a model that returns (N,1) silently broadcasts against a
    # (N,) target, turning every metric into an all-pairs average instead of
    # an element-wise one (MAE inflated ~1.3-1.4x). Flatten trailing singleton
    # axes, then refuse anything still mismatched rather than fail silently.
    true_knees = np.squeeze(true_knees) if true_knees.ndim > 1 else true_knees
    pred_knees = np.squeeze(pred_knees) if pred_knees.ndim > 1 else pred_knees
    true_knees = np.atleast_1d(true_knees)
    pred_knees = np.atleast_1d(pred_knees)
    if true_knees.shape != pred_knees.shape:
        raise ValueError(
            f"shape mismatch: true {true_knees.shape} vs pred {pred_knees.shape}. "
            "Broadcasting here would average metrics over all pairs; "
            "ravel the model output before evaluating."
        )

    results: Dict[str, float] = {
        'MAE': knee_mae(pred_knees, true_knees),
        'RMSE': knee_rmse(pred_knees, true_knees),
        'MAPE': knee_mape(pred_knees, true_knees),
        'MedianAE': knee_median_ae(pred_knees, true_knees),
        'Within_50': knee_within_tolerance(pred_knees, true_knees, tol=50),
        'Within_100': knee_within_tolerance(pred_knees, true_knees, tol=100),
        'Bias': knee_bias(pred_knees, true_knees),
    }

    if pred_std is not None:
        pred_std = np.asarray(pred_std, dtype=np.float64)
        results['Mean_Std'] = float(np.mean(pred_std))

    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        y_range = float(np.ptp(true_knees)) if true_knees.size > 1 else 1.0

        results['PICP'] = knee_picp(true_knees, lower, upper)
        results['MPIW'] = knee_mpiw(lower, upper)
        results['NMPIW'] = knee_nmpiw(lower, upper, y_range)
        results['CWC'] = knee_cwc(true_knees, lower, upper, alpha=alpha)
        results['Interval_Score'] = knee_interval_score(
            true_knees, lower, upper, alpha=alpha,
        )

    return results
