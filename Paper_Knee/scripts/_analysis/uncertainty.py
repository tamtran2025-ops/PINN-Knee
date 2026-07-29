"""
Uncertainty quantification methods for scalar knee-cycle prediction.

Paper: Physics-Constrained Deep Learning with Conformal Prediction
       for Early Knee-Point Prediction in Lithium-Ion Batteries.

Methods:
  1. Split Conformal Prediction (distribution-free coverage guarantee)
  2. MC Dropout inference
  3. Deep Ensemble inference
  4. Normal-approximation prediction intervals
"""

import math
import numpy as np
import torch
from typing import Tuple

from config import MC_DROPOUT_SAMPLES, CONFORMAL_ALPHA, DEVICE


# =============================================================================
#   1. Split Conformal Prediction
# =============================================================================

class KneeConformalPredictor:
    """Distribution-free prediction intervals via split conformal inference.

    Uses absolute residuals |true - pred| as nonconformity scores on a
    held-out calibration set, then applies the finite-sample quantile
    correction of Vovk et al. (2005) to guarantee marginal coverage
    at level (1 - alpha).

    Parameters
    ----------
    alpha : float, default=0.10
        Significance level. Prediction intervals target (1 - alpha)
        coverage, e.g. 0.10 gives 90 % intervals.

    Attributes
    ----------
    scores_ : np.ndarray or None
        Nonconformity scores from calibration, shape (n_cal,).
    q_hat_ : float or None
        Calibrated quantile threshold.
    calibrated_ : bool
        Whether ``calibrate()`` has been called.
    """

    def __init__(self, alpha: float = CONFORMAL_ALPHA):
        self.alpha = alpha
        self.scores_: np.ndarray | None = None
        self.q_hat_: float | None = None
        self.calibrated_: bool = False

    def calibrate(self, pred_cal: np.ndarray, true_cal: np.ndarray) -> None:
        """Compute nonconformity scores and the conformal quantile.

        Parameters
        ----------
        pred_cal : np.ndarray
            Model predictions on the calibration set, shape (n_cal,).
        true_cal : np.ndarray
            Ground-truth knee-point cycles for calibration, shape (n_cal,).

        Raises
        ------
        ValueError
            If the calibration set is empty.
        """
        pred_cal = np.asarray(pred_cal, dtype=np.float64).ravel()
        true_cal = np.asarray(true_cal, dtype=np.float64).ravel()

        if pred_cal.size == 0:
            raise ValueError("Calibration set must be non-empty.")

        self.scores_ = np.abs(true_cal - pred_cal)

        n = len(self.scores_)
        quantile_level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)

        self.q_hat_ = float(np.quantile(self.scores_, quantile_level))
        self.calibrated_ = True

    def predict(
        self, pred_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct prediction intervals around test predictions.

        Parameters
        ----------
        pred_test : np.ndarray
            Point predictions on the test set, shape (n_test,).

        Returns
        -------
        pred : np.ndarray
            Pass-through of ``pred_test``, shape (n_test,).
        lower : np.ndarray
            Lower bounds of prediction intervals, shape (n_test,).
        upper : np.ndarray
            Upper bounds of prediction intervals, shape (n_test,).

        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called.
        """
        if not self.calibrated_:
            raise RuntimeError(
                "Must call calibrate() before predict()."
            )

        pred_test = np.asarray(pred_test, dtype=np.float64).ravel()
        lower = pred_test - self.q_hat_
        upper = pred_test + self.q_hat_

        return pred_test, lower, upper


# =============================================================================
#   2. MC Dropout Inference
# =============================================================================

@torch.no_grad()
def mc_dropout_knee(
    model: torch.nn.Module,
    features_tensor: torch.Tensor,
    n_samples: int = MC_DROPOUT_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo Dropout forward passes for knee-point prediction.

    Enables dropout at inference time to produce a distribution of
    predictions per cell. The model must contain ``nn.Dropout`` layers.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model with dropout layers.
    features_tensor : torch.Tensor
        Input features, shape (n_cells, n_features) or compatible.
    n_samples : int, default=MC_DROPOUT_SAMPLES
        Number of stochastic forward passes.

    Returns
    -------
    knee_mean : np.ndarray
        Mean predicted knee-point per cell, shape (n_cells,).
    knee_std : np.ndarray
        Standard deviation per cell, shape (n_cells,).
    all_preds : np.ndarray
        Raw predictions, shape (n_samples, n_cells).
    """
    features_tensor = features_tensor.to(DEVICE)
    model.to(DEVICE)

    # Enable MC Dropout mode if available (PINN_Knee hybrid)
    if hasattr(model, 'enable_mc_dropout'):
        model.enable_mc_dropout()
    model.train()  # keep dropout active

    preds = []
    for _ in range(n_samples):
        out = model(features_tensor)  # (n_cells, 1) or (n_cells,)
        out = out.detach().cpu().numpy().ravel()
        preds.append(out)

    all_preds = np.stack(preds, axis=0)  # (n_samples, n_cells)
    knee_mean = all_preds.mean(axis=0)
    knee_std = all_preds.std(axis=0)

    # Disable MC Dropout mode
    if hasattr(model, 'disable_mc_dropout'):
        model.disable_mc_dropout()
    model.eval()  # restore eval mode
    return knee_mean, knee_std, all_preds


# =============================================================================
#   3. Deep Ensemble Inference
# =============================================================================

@torch.no_grad()
def ensemble_knee_predict(
    models: list,
    features_tensor: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average knee-point predictions from an ensemble of models.

    Each model produces an independent point prediction. The ensemble
    mean and standard deviation capture epistemic uncertainty.

    Parameters
    ----------
    models : list of torch.nn.Module
        Ensemble members (each independently trained).
    features_tensor : torch.Tensor
        Input features, shape (n_cells, n_features) or compatible.

    Returns
    -------
    knee_mean : np.ndarray
        Ensemble mean prediction per cell, shape (n_cells,).
    knee_std : np.ndarray
        Ensemble standard deviation per cell, shape (n_cells,).
    all_preds : np.ndarray
        Individual member predictions, shape (n_members, n_cells).
    """
    features_tensor = features_tensor.to(DEVICE)

    preds = []
    for m in models:
        m.to(DEVICE)
        m.eval()
        out = m(features_tensor).detach().cpu().numpy().ravel()
        preds.append(out)

    all_preds = np.stack(preds, axis=0)  # (n_members, n_cells)
    knee_mean = all_preds.mean(axis=0)
    knee_std = all_preds.std(axis=0)

    return knee_mean, knee_std, all_preds


# =============================================================================
#   4. Normal-Approximation Prediction Intervals
# =============================================================================

def mc_dropout_prediction_interval(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction interval from MC Dropout using a normal approximation.

    Computes mean +/- z_{1-alpha/2} * std.

    Parameters
    ----------
    mean : np.ndarray
        MC Dropout mean predictions, shape (n,).
    std : np.ndarray
        MC Dropout standard deviations, shape (n,).
    alpha : float, default=CONFORMAL_ALPHA
        Significance level (e.g. 0.10 for 90 % interval).

    Returns
    -------
    lower : np.ndarray
        Lower bounds, shape (n,).
    upper : np.ndarray
        Upper bounds, shape (n,).
    """
    from scipy.stats import norm

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    z = norm.ppf(1.0 - alpha / 2.0)

    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


def ensemble_prediction_interval(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction interval from ensemble using a normal approximation.

    Computes mean +/- z_{1-alpha/2} * std.

    Parameters
    ----------
    mean : np.ndarray
        Ensemble mean predictions, shape (n,).
    std : np.ndarray
        Ensemble standard deviations, shape (n,).
    alpha : float, default=CONFORMAL_ALPHA
        Significance level (e.g. 0.10 for 90 % interval).

    Returns
    -------
    lower : np.ndarray
        Lower bounds, shape (n,).
    upper : np.ndarray
        Upper bounds, shape (n,).
    """
    from scipy.stats import norm

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    z = norm.ppf(1.0 - alpha / 2.0)

    lower = mean - z * std
    upper = mean + z * std
    return lower, upper
