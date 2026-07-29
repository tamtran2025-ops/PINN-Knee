"""
Main experiment runner for:

  Physics-Constrained Deep Learning with Conformal Prediction
  for Early Knee-Point Prediction in Lithium-Ion Batteries.

Target: Applied Energy / Journal of Power Sources (Q1)

Experiments
-----------
1. Main knee prediction accuracy (all models, 3 n_early, 5 seeds)
2. UQ quality (PINN_Knee + Ensemble_NN, conformal + MC Dropout)
3. Ablation study (PINN_Knee physics components)
4. Early prediction (how few cycles are needed?)
5. Cross-dataset transfer (Severson -> NASA/CALCE)
"""

import os
import sys
import copy
import time
import traceback
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
#   Local imports
# ---------------------------------------------------------------------------
from config import (
    DEVICE,
    RESULTS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    RESULTS_CSV,
    UQ_CSV,
    ABLATION_CSV,
    EARLY_CSV,
    TRANSFER_CSV,
    MODEL_NAMES,
    N_SEEDS,
    N_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_CYCLE_COUNTS,
    EARLY_CYCLE_EXTENDED,
    MAX_CYCLE_LIFE,
    N_COLLOCATION,
    HIDDEN_SIZE,
    MC_DROPOUT_SAMPLES,
    CONFORMAL_ALPHA,
    PHYSICS_LAMBDA,
    TRAIN_FRACTION,
    CALIBRATION_FRACTION,
    print_config,
)
from data_loader import load_all_cells_with_knees, get_train_cal_test_split
from features import build_feature_matrix, normalize_features
from models import (
    create_model,
    is_nn_model,
    is_sequence_model,
    is_classical_model,
    is_pinn_model,
    count_parameters,
)
from metrics import (
    evaluate_knee_predictions,
    knee_picp,
    knee_mpiw,
    knee_interval_score,
)
from uncertainty import (
    KneeConformalPredictor,
    mc_dropout_knee,
    mc_dropout_prediction_interval,
    ensemble_knee_predict,
    ensemble_prediction_interval,
)
from train import train_pinn_knee


# ==========================================================================
#   Constants
# ==========================================================================
_CSV_LOCK_SUFFIX = '.lock'
_PATIENCE = 150                  # Early-stopping patience (epochs)
_MIN_DELTA = 0.01                # Minimum improvement for early stopping
_LOG_INTERVAL = 200              # Print loss every N epochs


# ==========================================================================
#   Utility: CSV I/O with checkpoint/resume
# ==========================================================================

def _csv_header(csv_path):
    """Return the header row of a CSV, or empty list if file missing."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def load_completed_runs(csv_path, key_columns):
    """Load a set of completed run keys from *csv_path*.

    Parameters
    ----------
    csv_path : str
        Path to the results CSV file.
    key_columns : list[str]
        Column names whose values, concatenated, form the run key.

    Returns
    -------
    completed : set[str]
        Set of pipe-separated key strings already in the CSV.
    """
    completed = set()
    if not os.path.isfile(csv_path):
        return completed
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = '|'.join(str(row.get(c, '')) for c in key_columns)
            completed.add(key)
    return completed


def append_result_to_csv(csv_path, row_dict):
    """Append a single result row to *csv_path*, creating the file if needed.

    Thread-safe via a simple lock file (sufficient for sequential runs).

    Parameters
    ----------
    csv_path : str
    row_dict : dict
        Column -> value mapping. The first call determines column order.
    """
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(row_dict.keys())

    # If the file exists, read existing header to maintain column order
    if file_exists:
        existing_header = _csv_header(csv_path)
        if existing_header:
            # Merge: existing columns first, then any new ones
            new_cols = [c for c in fieldnames if c not in existing_header]
            fieldnames = existing_header + new_cols

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ==========================================================================
#   Utility: Timing helpers
# ==========================================================================

def _fmt_eta(elapsed_s, done, total):
    """Return a human-readable ETA string."""
    if done == 0:
        return '???'
    rate = elapsed_s / done
    remaining = rate * (total - done)
    return str(timedelta(seconds=int(remaining)))


def _fmt_duration(seconds):
    """Format seconds into mm:ss or hh:mm:ss."""
    return str(timedelta(seconds=int(seconds)))


# ==========================================================================
#   Core: train a single model on given data
# ==========================================================================

def _prepare_tensors(X, y, device=DEVICE):
    """Convert numpy arrays to float32 tensors on *device*."""
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    return X_t, y_t


def train_nn_model(model, X_train, y_train, X_val=None, y_val=None,
                   n_epochs=N_EPOCHS, lr=LEARNING_RATE, wd=WEIGHT_DECAY,
                   patience=_PATIENCE, verbose=False):
    """Train a PyTorch knee-prediction model.

    Parameters
    ----------
    model : nn.Module
    X_train, y_train : ndarray
    X_val, y_val : ndarray or None  (for early stopping)
    n_epochs : int
    lr : float
    wd : float
    patience : int  (0 = no early stopping)
    verbose : bool

    Returns
    -------
    model : nn.Module  (best weights restored)
    history : dict  {'train_loss': [...], 'val_loss': [...]}
    """
    X_t, y_t = _prepare_tensors(X_train, y_train)
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_v, y_v = _prepare_tensors(X_val, y_val)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=50, min_lr=1e-6)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': []}

    # Only use physics loss for models explicitly designated as PINN
    use_pinn = is_pinn_model(type(model).__name__)
    n_col = torch.linspace(0, 1, N_COLLOCATION, device=DEVICE)

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        if use_pinn and hasattr(model, 'compute_loss'):
            loss = model.compute_loss(X_t, y_t, knee_max=MAX_CYCLE_LIFE,
                                      n_collocation=n_col)
        else:
            pred = model(X_t)
            # FIX(H1): model outputs (N,1) but y_t is (N,). Without ravel,
            # torch broadcasts to (N,N) and the loss collapses predictions
            # toward the training mean. Affected all 9 NN baselines.
            loss = nn.functional.mse_loss(pred.reshape(-1), y_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss = loss.item()
        history['train_loss'].append(train_loss)

        # Validation
        val_loss = train_loss
        if has_val:
            model.eval()
            with torch.no_grad():
                v_pred = model(X_v)
                # FIX(H1): same broadcast bug on the validation path, which
                # also corrupted early stopping / model selection.
                val_loss = nn.functional.mse_loss(v_pred.reshape(-1), y_v).item()
            model.train()
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping
        if patience > 0:
            if val_loss < best_val_loss - _MIN_DELTA:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"    Early stop at epoch {epoch}")
                    break

        if verbose and epoch % _LOG_INTERVAL == 0:
            print(f"    Epoch {epoch:>5d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def train_classical_model(model, X_train, y_train):
    """Train an XGBoost or RandomForest model.

    Returns
    -------
    model : fitted model
    """
    model.fit(X_train, y_train)
    return model


# ==========================================================================
#   Core: run a single knee experiment
# ==========================================================================

def run_single_knee_experiment(
    model_name, n_early, seed,
    train_cells, cal_cells, test_cells,
    verbose=False,
):
    """Train one model and evaluate on the test set.

    Parameters
    ----------
    model_name : str
    n_early : int
    seed : int
    train_cells, cal_cells, test_cells : list[dict]
    verbose : bool

    Returns
    -------
    result : dict   row to append to CSV
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build feature matrices
    X_train, y_train, feat_names, _ = build_feature_matrix(train_cells, n_early)
    X_cal,   y_cal,   _,          _ = build_feature_matrix(cal_cells,   n_early)
    X_test,  y_test,  _,          _ = build_feature_matrix(test_cells,  n_early)

    if X_train.size == 0 or X_test.size == 0:
        return {
            'model': model_name, 'n_early': n_early, 'seed': seed,
            'status': 'skipped_no_data',
            'n_train': 0, 'n_test': 0,
        }

    # Normalize
    X_train_n, X_test_n, X_cal_n, scaler = normalize_features(
        X_train, X_test, X_cal if X_cal.size > 0 else None,
    )

    n_features = X_train_n.shape[1]
    t0 = time.time()

    # --- Log-transform targets for NN models (improves learning on skewed data) ---
    use_log_target = is_nn_model(model_name) and not is_sequence_model(model_name)
    if use_log_target:
        y_train_t = np.log1p(y_train)  # log(1 + y) for numerical stability
        y_cal_t = np.log1p(y_cal) if y_cal.size > 0 else y_cal
    else:
        y_train_t = y_train
        y_cal_t = y_cal

    # --- Train ---
    if is_classical_model(model_name):
        model = create_model(model_name)
        model = train_classical_model(model, X_train_n, y_train)
        y_pred = model.predict(X_test_n)
    else:
        model = create_model(model_name, n_features=n_features, device=DEVICE)
        # For sequence models, reshape input
        if is_sequence_model(model_name):
            # Use raw capacity as sequence input (not hand-crafted features)
            # Build sequences from cells directly
            X_train_seq = _build_sequences(train_cells, n_early)
            X_test_seq = _build_sequences(test_cells, n_early)
            X_val_seq = _build_sequences(cal_cells, n_early) if len(cal_cells) > 0 else None

            if X_train_seq is None or X_test_seq is None:
                return {
                    'model': model_name, 'n_early': n_early, 'seed': seed,
                    'status': 'skipped_seq_fail',
                    'n_train': len(train_cells), 'n_test': len(test_cells),
                }

            model, _ = train_nn_model(
                model, X_train_seq, y_train,
                X_val=X_val_seq, y_val=y_cal if y_cal.size > 0 else None,
                verbose=verbose,
            )
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(DEVICE)
                y_pred = model(X_test_t).cpu().numpy()
        elif is_pinn_model(model_name):
            # train_pinn_knee handles the log-transform internally;
            # pass the RAW y values so we don't double-log.
            model, _ = train_pinn_knee(
                model, X_train_n, y_train, train_cells, n_early,
                X_val=X_cal_n, y_val=y_cal if y_cal.size > 0 else None,
                use_log_target=use_log_target,
            )
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
                y_pred_raw = model(X_test_t).cpu().numpy().squeeze()
                y_pred = np.expm1(y_pred_raw) if use_log_target else y_pred_raw
        elif model_name == 'Ensemble_NN':
            # Proper ensemble: train N members with different seeds,
            # then average predictions.
            from config import ENSEMBLE_SIZE
            from models import Ensemble_NN_Member
            members = []
            X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
            preds = []
            for m_idx in range(ENSEMBLE_SIZE):
                torch.manual_seed(seed * 1000 + m_idx * 17 + 3)
                np.random.seed(seed * 1000 + m_idx * 17 + 3)
                member = Ensemble_NN_Member(n_features=n_features).to(DEVICE)
                member, _ = train_nn_model(
                    member, X_train_n, y_train_t,
                    X_val=X_cal_n, y_val=y_cal_t if y_cal.size > 0 else None,
                    verbose=False,
                )
                member.eval()
                with torch.no_grad():
                    p_log = member(X_test_t).cpu().numpy().squeeze()
                preds.append(np.expm1(p_log) if use_log_target else p_log)
                members.append(member)
            preds = np.array(preds)  # (n_members, n_test)
            y_pred = preds.mean(axis=0)
            model = members[0]  # For parameter-counting
        else:
            model, _ = train_nn_model(
                model, X_train_n, y_train_t,
                X_val=X_cal_n, y_val=y_cal_t if y_cal.size > 0 else None,
                verbose=verbose,
            )
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
                y_pred_raw = model(X_test_t).cpu().numpy().squeeze()
                y_pred = np.expm1(y_pred_raw) if use_log_target else y_pred_raw

    elapsed = time.time() - t0

    # --- Evaluate ---
    metrics = evaluate_knee_predictions(y_test, y_pred)

    result = {
        'model': model_name,
        'n_early': n_early,
        'seed': seed,
        'status': 'ok',
        'n_train': len(y_train),
        'n_cal': len(y_cal) if y_cal.size > 0 else 0,
        'n_test': len(y_test),
        'n_features': n_features,
        'train_time_s': round(elapsed, 2),
    }
    if is_nn_model(model_name):
        result['n_params'] = count_parameters(model)

    for k, v in metrics.items():
        result[k] = round(v, 4)

    return result


def _build_sequences(cells, n_early):
    """Build (n_cells, n_early, 1) capacity sequence arrays.

    Returns None if no valid cells are available.
    """
    seqs = []
    for cell in cells:
        cap = cell['capacity']
        if len(cap) < n_early:
            continue
        if cell.get('knee_cycle') is None or cell['knee_cycle'] <= n_early:
            continue
        seqs.append(cap[:n_early])

    if len(seqs) == 0:
        return None

    # Pad / truncate to exactly n_early
    arr = np.array(seqs, dtype=np.float32)  # (n_cells, n_early)
    # Normalize per cell
    mins = arr.min(axis=1, keepdims=True)
    maxs = arr.max(axis=1, keepdims=True)
    arr = (arr - mins) / (maxs - mins + 1e-8)
    return arr.reshape(arr.shape[0], arr.shape[1], 1)


# ==========================================================================
#   Experiment 1: Main Knee Prediction Accuracy
# ==========================================================================

def _kfold_split(cells, n_folds=5, seed=42):
    """Split cells into k folds for cross-validation.

    Returns list of (train_cells, cal_cells, test_cells) tuples.
    Each fold uses ~60% train, ~20% cal, ~20% test.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(cells))
    fold_size = len(cells) // n_folds

    splits = []
    for fold in range(n_folds):
        test_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        remaining = np.concatenate([indices[:fold * fold_size],
                                     indices[(fold + 1) * fold_size:]])
        # Use ~25% of remaining for calibration
        n_cal = max(1, len(remaining) // 4)
        cal_idx = remaining[:n_cal]
        train_idx = remaining[n_cal:]

        splits.append((
            [cells[i] for i in train_idx],
            [cells[i] for i in cal_idx],
            [cells[i] for i in test_idx],
        ))
    return splits


def run_experiment_1_main(cells):
    """Experiment 1: benchmark all models with 5-fold cross-validation.

    - 5-fold CV (each cell appears in test set exactly once)
    - Models: all MODEL_NAMES
    - n_early: EARLY_CYCLE_COUNTS
    - Seeds: range(N_SEEDS) (model initialization seeds)
    - Checkpoint/resume via RESULTS_CSV
    """
    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 1: Main Knee Prediction (5-Fold CV)")
    print("=" * 70)

    N_FOLDS = 5
    folds = _kfold_split(cells, n_folds=N_FOLDS, seed=42)

    for fi, (tr, ca, te) in enumerate(folds):
        print(f"  Fold {fi}: train={len(tr)}, cal={len(ca)}, test={len(te)}")

    key_cols = ['model', 'n_early', 'seed', 'fold']
    completed = load_completed_runs(RESULTS_CSV, key_cols)

    # Build task list
    tasks = []
    for n_early in EARLY_CYCLE_COUNTS:
        for model_name in MODEL_NAMES:
            for seed in range(N_SEEDS):
                for fold in range(N_FOLDS):
                    key = f"{model_name}|{n_early}|{seed}|{fold}"
                    if key not in completed:
                        tasks.append((model_name, n_early, seed, fold))

    total = len(tasks)
    full_total = len(EARLY_CYCLE_COUNTS) * len(MODEL_NAMES) * N_SEEDS * N_FOLDS
    skipped = full_total - total

    if skipped > 0:
        print(f"  Resuming: {skipped} runs already completed, {total} remaining")
    if total == 0:
        print("  All runs complete. Skipping Experiment 1.")
        _print_exp1_summary()
        return

    print(f"  Total runs: {total} ({len(MODEL_NAMES)} models x "
          f"{len(EARLY_CYCLE_COUNTS)} n_early x {N_SEEDS} seeds x {N_FOLDS} folds)")
    print("-" * 70)

    t_start = time.time()
    errors = []

    for i, (model_name, n_early, seed, fold) in enumerate(tasks, 1):
        elapsed = time.time() - t_start
        eta = _fmt_eta(elapsed, i - 1, total) if i > 1 else '???'
        print(f"  [{i:>4d}/{total}]  {model_name:<15s}  "
              f"ne={n_early:<4d} s={seed} f={fold}  ETA={eta}", end='')

        train_cells, cal_cells, test_cells = folds[fold]

        try:
            result = run_single_knee_experiment(
                model_name, n_early, seed,
                train_cells, cal_cells, test_cells,
            )
            result['fold'] = fold
            append_result_to_csv(RESULTS_CSV, result)

            status = result.get('status', 'ok')
            mae = result.get('MAE', -1)
            print(f"  -> MAE={mae:.1f}  [{status}]")

        except Exception as e:
            err_msg = f"{model_name}/n{n_early}/s{seed}/f{fold}: {e}"
            errors.append(err_msg)
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    # Summary
    total_time = time.time() - t_start
    print("\n" + "-" * 70)
    print(f"  Experiment 1 complete in {_fmt_duration(total_time)}")
    print(f"  Successful: {total - len(errors)} / {total}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
    print("-" * 70)

    _print_exp1_summary()


def _print_exp1_summary():
    """Print a concise summary table from RESULTS_CSV."""
    if not os.path.isfile(RESULTS_CSV):
        return
    print("\n  --- Experiment 1 Summary (mean +/- std across seeds) ---")
    rows = _read_csv_dicts(RESULTS_CSV)
    if not rows:
        return

    # Group by (model, n_early)
    groups = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        key = (r['model'], r['n_early'])
        groups.setdefault(key, []).append(r)

    print(f"  {'Model':<15s} {'n_early':>7s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE':>10s} {'W50':>7s} {'runs':>5s}")
    print("  " + "-" * 65)

    for (model, n_early) in sorted(groups.keys(), key=lambda x: (x[1], x[0])):
        rs = groups[(model, n_early)]
        maes  = [float(r.get('MAE', 0))  for r in rs]
        rmses = [float(r.get('RMSE', 0)) for r in rs]
        mapes = [float(r.get('MAPE', 0)) for r in rs]
        w50s  = [float(r.get('Within_50', 0)) for r in rs]
        print(f"  {model:<15s} {n_early:>7s} "
              f"{np.mean(maes):>6.1f}+/-{np.std(maes):>4.1f} "
              f"{np.mean(rmses):>6.1f}+/-{np.std(rmses):>4.1f} "
              f"{np.mean(mapes):>6.1f}+/-{np.std(mapes):>4.1f} "
              f"{np.mean(w50s):>5.2f} "
              f"{len(rs):>5d}")
    print()


# ==========================================================================
#   Experiment 2: UQ Quality
# ==========================================================================

def run_experiment_2_uq(cells):
    """Experiment 2: Uncertainty Quantification quality.

    - Models with UQ: PINN_Knee (MC Dropout), Ensemble_NN (member disagreement)
    - Also: Conformal Prediction on calibration set
    - n_early: [50, 100]
    - Metrics: PICP, MPIW, Interval Score
    """
    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 2: Uncertainty Quantification Quality")
    print("=" * 70)

    train_cells, cal_cells, test_cells = get_train_cal_test_split(cells, seed=42)

    uq_models = ['PINN_Knee', 'Ensemble_NN']
    uq_n_early = [50, 100]

    key_cols = ['model', 'n_early', 'seed', 'uq_method']
    completed = load_completed_runs(UQ_CSV, key_cols)

    tasks = []
    for n_early in uq_n_early:
        for model_name in uq_models:
            for seed in range(N_SEEDS):
                for uq_method in _get_uq_methods(model_name):
                    key = f"{model_name}|{n_early}|{seed}|{uq_method}"
                    if key not in completed:
                        tasks.append((model_name, n_early, seed, uq_method))

    total = len(tasks)
    skipped_total = (len(uq_n_early) * len(uq_models) * N_SEEDS *
                     3) - total  # approximate
    if total == 0:
        print("  All UQ runs complete. Skipping.")
        _print_exp2_summary()
        return

    print(f"  UQ models:   {uq_models}")
    print(f"  n_early:     {uq_n_early}")
    print(f"  Total runs:  {total}")
    print("-" * 70)

    t_start = time.time()
    errors = []

    for i, (model_name, n_early, seed, uq_method) in enumerate(tasks, 1):
        elapsed = time.time() - t_start
        eta = _fmt_eta(elapsed, i - 1, total) if i > 1 else '???'
        print(f"  [{i:>4d}/{total}]  {model_name:<15s}  "
              f"n_early={n_early}  seed={seed}  method={uq_method}  ETA={eta}",
              end='')

        try:
            result = _run_uq_experiment(
                model_name, n_early, seed, uq_method,
                train_cells, cal_cells, test_cells,
            )
            append_result_to_csv(UQ_CSV, result)
            picp = result.get('PICP', -1)
            mpiw = result.get('MPIW', -1)
            print(f"  -> PICP={picp:.3f}  MPIW={mpiw:.1f}")

        except Exception as e:
            errors.append(f"{model_name}/{uq_method}/n{n_early}/s{seed}: {e}")
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\n  Experiment 2 complete in {_fmt_duration(total_time)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    _print_exp2_summary()


def _get_uq_methods(model_name):
    """Return applicable UQ methods for a given model."""
    methods = ['conformal']
    if model_name == 'PINN_Knee':
        methods.append('mc_dropout')
    if model_name == 'Ensemble_NN':
        methods.append('ensemble_disagreement')
    return methods


def _run_uq_experiment(model_name, n_early, seed, uq_method,
                       train_cells, cal_cells, test_cells):
    """Run a single UQ evaluation."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train, y_train, _, _ = build_feature_matrix(train_cells, n_early)
    X_cal,   y_cal,   _, _ = build_feature_matrix(cal_cells,   n_early)
    X_test,  y_test,  _, _ = build_feature_matrix(test_cells,  n_early)

    if X_train.size == 0 or X_test.size == 0:
        return {
            'model': model_name, 'n_early': n_early, 'seed': seed,
            'uq_method': uq_method, 'status': 'skipped_no_data',
        }

    X_train_n, X_test_n, X_cal_n, scaler = normalize_features(
        X_train, X_test, X_cal if X_cal.size > 0 else None,
    )
    n_features = X_train_n.shape[1]

    # Log-transform targets for NN models
    use_log = is_nn_model(model_name) and not is_sequence_model(model_name)
    y_train_t = np.log1p(y_train) if use_log else y_train
    y_cal_t = np.log1p(y_cal) if (use_log and y_cal.size > 0) else y_cal

    # Train the model
    model = create_model(model_name, n_features=n_features, device=DEVICE)
    if is_pinn_model(model_name):
        model, _ = train_pinn_knee(model, X_train_n, y_train, train_cells, n_early,
                                   use_log_target=use_log)
    else:
        model, _ = train_nn_model(model, X_train_n, y_train_t,
                                  X_val=X_cal_n, y_val=y_cal_t if y_cal.size > 0 else None)

    # Get point predictions on test and calibration sets
    model.eval()
    X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
    X_cal_t = torch.tensor(X_cal_n, dtype=torch.float32).to(DEVICE) \
        if X_cal_n is not None else None

    with torch.no_grad():
        # FIX(H7): ravel to (N,). Without it y_pred_test is (N,1) and
        # evaluate_knee_predictions broadcasts against y_test (N,) -> (N,N),
        # so MAE is averaged over ALL pairs instead of the diagonal.
        y_pred_raw = model(X_test_t).cpu().numpy().ravel()
        y_pred_test = np.expm1(y_pred_raw) if use_log else y_pred_raw

    # UQ method dispatch
    lower, upper, pred_std = None, None, None

    if uq_method == 'conformal':
        # Conformal prediction using calibration set
        if X_cal_t is None or y_cal.size == 0:
            return {
                'model': model_name, 'n_early': n_early, 'seed': seed,
                'uq_method': uq_method, 'status': 'skipped_no_cal',
            }
        with torch.no_grad():
            y_pred_cal_raw = model(X_cal_t).cpu().numpy()
            y_pred_cal = np.expm1(y_pred_cal_raw) if use_log else y_pred_cal_raw

        cp = KneeConformalPredictor(alpha=CONFORMAL_ALPHA)
        cp.calibrate(y_pred_cal, y_cal)  # Use original scale
        _, lower, upper = cp.predict(y_pred_test)

    elif uq_method == 'mc_dropout':
        # MC Dropout: run in log-space, then inverse-transform
        y_mc_mean_raw, y_mc_std_raw, all_preds_raw = mc_dropout_knee(
            model, X_test_t, n_samples=MC_DROPOUT_SAMPLES,
        )
        if use_log:
            # Transform all MC samples back to original scale
            all_preds = np.expm1(all_preds_raw) if all_preds_raw is not None \
                else None
            y_mc_mean = np.expm1(y_mc_mean_raw)
            # Recompute std in original scale from transformed samples
            if all_preds is not None:
                y_mc_std = np.std(all_preds, axis=0)
            else:
                # Approximate: delta method
                y_mc_std = y_mc_std_raw * np.exp(y_mc_mean_raw)
        else:
            y_mc_mean = y_mc_mean_raw
            y_mc_std = y_mc_std_raw

        y_pred_test = y_mc_mean
        pred_std = y_mc_std
        lower, upper = mc_dropout_prediction_interval(
            y_mc_mean, y_mc_std, alpha=CONFORMAL_ALPHA,
        )

    elif uq_method == 'ensemble_disagreement':
        if hasattr(model, 'predict_all'):
            all_preds = model.predict_all(X_test_t).detach().cpu().numpy()
            y_pred_test = all_preds.mean(axis=0)
            pred_std = all_preds.std(axis=0)
            lower, upper = ensemble_prediction_interval(
                y_pred_test, pred_std, alpha=CONFORMAL_ALPHA,
            )
        else:
            return {
                'model': model_name, 'n_early': n_early, 'seed': seed,
                'uq_method': uq_method, 'status': 'skipped_no_ensemble',
            }

    # Compute metrics
    result = {
        'model': model_name,
        'n_early': n_early,
        'seed': seed,
        'uq_method': uq_method,
        'status': 'ok',
        'n_test': len(y_test),
        'n_cal': len(y_cal) if y_cal.size > 0 else 0,
    }

    # Point prediction metrics
    point_metrics = evaluate_knee_predictions(
        y_test, y_pred_test, lower=lower, upper=upper, pred_std=pred_std,
    )
    for k, v in point_metrics.items():
        result[k] = round(v, 4)

    return result


def _print_exp2_summary():
    """Print UQ summary from UQ_CSV."""
    if not os.path.isfile(UQ_CSV):
        return
    rows = _read_csv_dicts(UQ_CSV)
    if not rows:
        return
    print("\n  --- Experiment 2 Summary (UQ Quality) ---")
    print(f"  {'Model':<15s} {'UQ Method':<22s} {'n_early':>7s} "
          f"{'PICP':>8s} {'MPIW':>8s} {'IntScore':>10s}")
    print("  " + "-" * 75)

    groups = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        key = (r['model'], r.get('uq_method', ''), r['n_early'])
        groups.setdefault(key, []).append(r)

    for key in sorted(groups.keys()):
        model, method, n_early = key
        rs = groups[key]
        picps = [float(r.get('PICP', 0)) for r in rs]
        mpiws = [float(r.get('MPIW', 0)) for r in rs]
        iscs  = [float(r.get('Interval_Score', 0)) for r in rs]
        print(f"  {model:<15s} {method:<22s} {n_early:>7s} "
              f"{np.mean(picps):>8.3f} {np.mean(mpiws):>8.1f} "
              f"{np.mean(iscs):>10.1f}")
    print()


# ==========================================================================
#   Experiment 3: Ablation Study
# ==========================================================================

def run_experiment_3_ablation(cells):
    """Experiment 3: Ablation of PINN_Knee physics components.

    Only PINN_Knee, n_early=100, 5 seeds.
    Configs: full, no_physics, no_monotonic, no_sei, no_knee_transition,
             no_ode, no_ic
    """
    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 3: Ablation Study (PINN Physics Components)")
    print("=" * 70)

    train_cells, cal_cells, test_cells = get_train_cal_test_split(cells, seed=42)
    n_early = 100

    # Define ablation configurations
    ablation_configs = _get_ablation_configs()

    key_cols = ['config', 'seed']
    completed = load_completed_runs(ABLATION_CSV, key_cols)

    tasks = []
    for config_name in ablation_configs:
        for seed in range(N_SEEDS):
            key = f"{config_name}|{seed}"
            if key not in completed:
                tasks.append((config_name, seed))

    total = len(tasks)
    if total == 0:
        print("  All ablation runs complete. Skipping.")
        _print_exp3_summary()
        return

    print(f"  Model: PINN_Knee only")
    print(f"  n_early: {n_early}")
    print(f"  Configs: {list(ablation_configs.keys())}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Total runs: {total}")
    print("-" * 70)

    t_start = time.time()
    errors = []

    for i, (config_name, seed) in enumerate(tasks, 1):
        elapsed = time.time() - t_start
        eta = _fmt_eta(elapsed, i - 1, total) if i > 1 else '???'
        print(f"  [{i:>4d}/{total}]  config={config_name:<20s}  "
              f"seed={seed}  ETA={eta}", end='')

        try:
            result = _run_ablation_experiment(
                config_name, ablation_configs[config_name],
                n_early, seed,
                train_cells, cal_cells, test_cells,
            )
            append_result_to_csv(ABLATION_CSV, result)
            mae = result.get('MAE', -1)
            print(f"  -> MAE={mae:.1f}")

        except Exception as e:
            errors.append(f"{config_name}/s{seed}: {e}")
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\n  Experiment 3 complete in {_fmt_duration(total_time)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    _print_exp3_summary()


def _get_ablation_configs():
    """Return ablation config dict: name -> modified PHYSICS_LAMBDA."""
    base = dict(PHYSICS_LAMBDA)  # copy of original
    configs = {}

    # Full model (all physics)
    configs['full'] = dict(base)

    # No physics at all
    configs['no_physics'] = {k: 0.0 for k in base}

    # Ablate each component individually
    configs['no_monotonic'] = {**base, 'monotonic_decay': 0.0}
    configs['no_sei'] = {**base, 'sei_sqrt_t': 0.0}
    configs['no_knee_transition'] = {**base, 'knee_transition': 0.0}
    configs['no_ode'] = {**base, 'degradation_ode': 0.0}
    configs['no_ic'] = {**base, 'initial_condition': 0.0}

    return configs


def _run_ablation_experiment(config_name, physics_lambda, n_early, seed,
                             train_cells, cal_cells, test_cells):
    """Run a single ablation configuration."""
    import config as cfg

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train, y_train, _, _ = build_feature_matrix(train_cells, n_early)
    X_cal,   y_cal,   _, _ = build_feature_matrix(cal_cells,   n_early)
    X_test,  y_test,  _, _ = build_feature_matrix(test_cells,  n_early)

    if X_train.size == 0 or X_test.size == 0:
        return {
            'config': config_name, 'seed': seed,
            'status': 'skipped_no_data',
        }

    X_train_n, X_test_n, X_cal_n, _ = normalize_features(
        X_train, X_test, X_cal if X_cal.size > 0 else None,
    )
    n_features = X_train_n.shape[1]

    # Pass physics_lambda directly to train_pinn_knee (with log-transform)
    model = create_model('PINN_Knee', n_features=n_features, device=DEVICE)
    model, _ = train_pinn_knee(
        model, X_train_n, y_train, train_cells, n_early,
        physics_lambda=physics_lambda,
        X_val=X_cal_n, y_val=y_cal if y_cal.size > 0 else None,
        use_log_target=True,
    )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
        y_pred_log = model(X_test_t).cpu().numpy().ravel()   # FIX(H7)
        y_pred = np.expm1(y_pred_log)  # Inverse log-transform

    metrics = evaluate_knee_predictions(y_test, y_pred)

    result = {
        'config': config_name,
        'seed': seed,
        'model': 'PINN_Knee',
        'n_early': n_early,
        'status': 'ok',
        'n_train': len(y_train),
        'n_test': len(y_test),
    }
    # Record which lambdas were used
    for k, v in physics_lambda.items():
        result[f'lambda_{k}'] = v

    for k, v in metrics.items():
        result[k] = round(v, 4)

    return result


def _print_exp3_summary():
    """Print ablation summary from ABLATION_CSV."""
    if not os.path.isfile(ABLATION_CSV):
        return
    rows = _read_csv_dicts(ABLATION_CSV)
    if not rows:
        return
    print("\n  --- Experiment 3 Summary (Ablation Study) ---")
    print(f"  {'Config':<22s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE':>10s} {'W50':>7s}")
    print("  " + "-" * 62)

    groups = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        groups.setdefault(r['config'], []).append(r)

    # Print 'full' first, then rest sorted
    order = ['full'] + [c for c in sorted(groups.keys()) if c != 'full']
    for config in order:
        if config not in groups:
            continue
        rs = groups[config]
        maes  = [float(r.get('MAE', 0))  for r in rs]
        rmses = [float(r.get('RMSE', 0)) for r in rs]
        mapes = [float(r.get('MAPE', 0)) for r in rs]
        w50s  = [float(r.get('Within_50', 0)) for r in rs]
        marker = " <-- full" if config == 'full' else ""
        print(f"  {config:<22s} "
              f"{np.mean(maes):>6.1f}+/-{np.std(maes):>4.1f} "
              f"{np.mean(rmses):>6.1f}+/-{np.std(rmses):>4.1f} "
              f"{np.mean(mapes):>6.1f}+/-{np.std(mapes):>4.1f} "
              f"{np.mean(w50s):>5.2f}{marker}")
    print()


# ==========================================================================
#   Experiment 4: Early Prediction (how few cycles?)
# ==========================================================================

def run_experiment_4_early_prediction(cells):
    """Experiment 4: How few early cycles are needed?

    - Models: PINN_Knee, XGBoost, Pure_NN
    - n_early: EARLY_CYCLE_EXTENDED = [10, 15, 20, 30, 50, 75, 100, 150]
    - Seeds: N_SEEDS
    """
    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 4: Early Prediction (Minimum Cycles Needed)")
    print("=" * 70)

    train_cells, cal_cells, test_cells = get_train_cal_test_split(cells, seed=42)

    early_models = ['PINN_Knee', 'XGBoost', 'Pure_NN']

    key_cols = ['model', 'n_early', 'seed']
    completed = load_completed_runs(EARLY_CSV, key_cols)

    tasks = []
    for n_early in EARLY_CYCLE_EXTENDED:
        for model_name in early_models:
            for seed in range(N_SEEDS):
                key = f"{model_name}|{n_early}|{seed}"
                if key not in completed:
                    tasks.append((model_name, n_early, seed))

    total = len(tasks)
    if total == 0:
        print("  All early prediction runs complete. Skipping.")
        _print_exp4_summary()
        return

    print(f"  Models:     {early_models}")
    print(f"  n_early:    {EARLY_CYCLE_EXTENDED}")
    print(f"  Seeds:      {N_SEEDS}")
    print(f"  Total runs: {total}")
    print("-" * 70)

    t_start = time.time()
    errors = []

    for i, (model_name, n_early, seed) in enumerate(tasks, 1):
        elapsed = time.time() - t_start
        eta = _fmt_eta(elapsed, i - 1, total) if i > 1 else '???'
        print(f"  [{i:>4d}/{total}]  model={model_name:<12s}  "
              f"n_early={n_early:<4d}  seed={seed}  ETA={eta}", end='')

        try:
            result = run_single_knee_experiment(
                model_name, n_early, seed,
                train_cells, cal_cells, test_cells,
            )
            # Add experiment tag
            result['experiment'] = 'early_prediction'
            append_result_to_csv(EARLY_CSV, result)

            mae = result.get('MAE', -1)
            print(f"  -> MAE={mae:.1f}  [{result.get('status', '?')}]")

        except Exception as e:
            errors.append(f"{model_name}/n{n_early}/s{seed}: {e}")
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\n  Experiment 4 complete in {_fmt_duration(total_time)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    _print_exp4_summary()


def _print_exp4_summary():
    """Print early prediction summary from EARLY_CSV."""
    if not os.path.isfile(EARLY_CSV):
        return
    rows = _read_csv_dicts(EARLY_CSV)
    if not rows:
        return
    print("\n  --- Experiment 4 Summary (Early Prediction) ---")
    print(f"  {'Model':<15s} {'n_early':>7s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE':>10s}")
    print("  " + "-" * 55)

    groups = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        key = (r['model'], int(r['n_early']))
        groups.setdefault(key, []).append(r)

    for key in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        model, n_early = key
        rs = groups[key]
        maes  = [float(r.get('MAE', 0))  for r in rs]
        rmses = [float(r.get('RMSE', 0)) for r in rs]
        mapes = [float(r.get('MAPE', 0)) for r in rs]
        print(f"  {model:<15s} {n_early:>7d} "
              f"{np.mean(maes):>6.1f}+/-{np.std(maes):>4.1f} "
              f"{np.mean(rmses):>6.1f}+/-{np.std(rmses):>4.1f} "
              f"{np.mean(mapes):>6.1f}+/-{np.std(mapes):>4.1f}")
    print()


# ==========================================================================
#   Experiment 5: Cross-Dataset Transfer
# ==========================================================================

def run_experiment_5_transfer(cells):
    """Experiment 5: Cross-dataset transfer.

    - Train on Severson only
    - Test on NASA + CALCE cells (if they have knees)
    - Compare: train from scratch on NASA/CALCE vs transfer from Severson
    """
    print("\n")
    print("=" * 70)
    print("  EXPERIMENT 5: Cross-Dataset Transfer Learning")
    print("=" * 70)

    # Split cells by dataset
    severson_cells = [c for c in cells if c.get('dataset') == 'severson']
    nasa_cells     = [c for c in cells if c.get('dataset') == 'nasa']
    calce_cells    = [c for c in cells if c.get('dataset') == 'calce']

    if len(severson_cells) == 0:
        print("  No Severson cells available. Skipping transfer experiment.")
        return

    target_cells = nasa_cells + calce_cells
    if len(target_cells) == 0:
        print("  No NASA/CALCE cells available. Skipping transfer experiment.")
        return

    print(f"  Source (Severson): {len(severson_cells)} cells")
    print(f"  Target (NASA+CALCE): {len(target_cells)} cells "
          f"(NASA={len(nasa_cells)}, CALCE={len(calce_cells)})")

    transfer_models = ['PINN_Knee', 'Pure_NN', 'XGBoost']
    n_early = 100

    key_cols = ['model', 'scenario', 'seed']
    completed = load_completed_runs(TRANSFER_CSV, key_cols)

    tasks = []
    for model_name in transfer_models:
        for seed in range(N_SEEDS):
            for scenario in ['source_only', 'target_only', 'transfer']:
                key = f"{model_name}|{scenario}|{seed}"
                if key not in completed:
                    tasks.append((model_name, scenario, seed))

    total = len(tasks)
    if total == 0:
        print("  All transfer runs complete. Skipping.")
        _print_exp5_summary()
        return

    print(f"  Models:     {transfer_models}")
    print(f"  n_early:    {n_early}")
    print(f"  Scenarios:  source_only, target_only, transfer")
    print(f"  Seeds:      {N_SEEDS}")
    print(f"  Total runs: {total}")
    print("-" * 70)

    t_start = time.time()
    errors = []

    for i, (model_name, scenario, seed) in enumerate(tasks, 1):
        elapsed = time.time() - t_start
        eta = _fmt_eta(elapsed, i - 1, total) if i > 1 else '???'
        print(f"  [{i:>4d}/{total}]  model={model_name:<12s}  "
              f"scenario={scenario:<15s}  seed={seed}  ETA={eta}", end='')

        try:
            result = _run_transfer_experiment(
                model_name, scenario, n_early, seed,
                severson_cells, target_cells,
            )
            append_result_to_csv(TRANSFER_CSV, result)
            mae = result.get('MAE', -1)
            print(f"  -> MAE={mae:.1f}  [{result.get('status', '?')}]")

        except Exception as e:
            errors.append(f"{model_name}/{scenario}/s{seed}: {e}")
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    total_time = time.time() - t_start
    print(f"\n  Experiment 5 complete in {_fmt_duration(total_time)}")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    _print_exp5_summary()


def _run_transfer_experiment(model_name, scenario, n_early, seed,
                             source_cells, target_cells):
    """Run a single transfer learning experiment.

    Scenarios:
      - source_only:  Train on Severson, test on NASA/CALCE (zero-shot)
      - target_only:  Train on NASA/CALCE split, test on held-out NASA/CALCE
      - transfer:     Pre-train on Severson, fine-tune on small NASA/CALCE
                      training split, test on held-out NASA/CALCE
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Split target into train/test (70/30 for the small target set)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(target_cells))
    n_target_train = max(1, int(0.7 * len(target_cells)))
    target_train = [target_cells[j] for j in indices[:n_target_train]]
    target_test  = [target_cells[j] for j in indices[n_target_train:]]

    if len(target_test) == 0:
        # If too few target cells, use leave-one-out
        target_test = [target_cells[indices[-1]]]
        target_train = [target_cells[j] for j in indices[:-1]]

    # Build feature matrices
    X_src, y_src, _, _ = build_feature_matrix(source_cells, n_early)
    X_tgt_train, y_tgt_train, _, _ = build_feature_matrix(target_train, n_early)
    X_tgt_test,  y_tgt_test,  _, _ = build_feature_matrix(target_test, n_early)

    if X_tgt_test.size == 0:
        return {
            'model': model_name, 'scenario': scenario, 'seed': seed,
            'n_early': n_early, 'status': 'skipped_no_target_test',
        }

    # Determine training data and normalization
    if scenario == 'source_only':
        if X_src.size == 0:
            return {
                'model': model_name, 'scenario': scenario, 'seed': seed,
                'n_early': n_early, 'status': 'skipped_no_source',
            }
        X_train_all = X_src
        y_train_all = y_src
    elif scenario == 'target_only':
        if X_tgt_train.size == 0:
            return {
                'model': model_name, 'scenario': scenario, 'seed': seed,
                'n_early': n_early, 'status': 'skipped_no_target_train',
            }
        X_train_all = X_tgt_train
        y_train_all = y_tgt_train
    else:  # transfer
        if X_src.size == 0:
            return {
                'model': model_name, 'scenario': scenario, 'seed': seed,
                'n_early': n_early, 'status': 'skipped_no_source',
            }
        X_train_all = X_src
        y_train_all = y_src

    X_train_n, X_test_n, _, scaler = normalize_features(X_train_all, X_tgt_test)
    n_features = X_train_n.shape[1]

    t0 = time.time()

    if is_classical_model(model_name):
        model = create_model(model_name)
        model = train_classical_model(model, X_train_n, y_train_all)

        if scenario == 'transfer' and X_tgt_train.size > 0:
            # Fine-tune: re-train on combined source + target
            X_combined = np.vstack([X_src, X_tgt_train])
            y_combined = np.concatenate([y_src, y_tgt_train])
            X_combined_n = scaler.transform(X_combined)
            model = create_model(model_name)
            model = train_classical_model(model, X_combined_n, y_combined)

        y_pred = model.predict(X_test_n)

    else:
        model = create_model(model_name, n_features=n_features, device=DEVICE)
        use_log = is_nn_model(model_name) and not is_sequence_model(model_name)
        y_train_log = np.log1p(y_train_all) if use_log else y_train_all

        # Pre-train on source
        if is_pinn_model(model_name):
            model, _ = train_pinn_knee(model, X_train_n, y_train_all,
                                        source_cells, n_early, use_log_target=use_log)
        else:
            model, _ = train_nn_model(model, X_train_n, y_train_log)

        if scenario == 'transfer' and X_tgt_train.size > 0:
            # Fine-tune on target with lower LR
            X_tgt_train_n = scaler.transform(X_tgt_train)
            y_tgt_log = np.log1p(y_tgt_train) if use_log else y_tgt_train
            if is_pinn_model(model_name):
                model, _ = train_pinn_knee(model, X_tgt_train_n, y_tgt_train,
                                            target_cells, n_early, n_epochs=500,
                                            lr=LEARNING_RATE * 0.1, use_log_target=use_log)
            else:
                model, _ = train_nn_model(
                    model, X_tgt_train_n, y_tgt_log,
                    n_epochs=500, lr=LEARNING_RATE * 0.1, patience=50,
                )

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
            y_pred_raw = model(X_test_t).cpu().numpy().ravel()   # FIX(H7)
            y_pred = np.expm1(y_pred_raw) if use_log else y_pred_raw

    elapsed = time.time() - t0

    metrics = evaluate_knee_predictions(y_tgt_test, y_pred)

    result = {
        'model': model_name,
        'scenario': scenario,
        'seed': seed,
        'n_early': n_early,
        'status': 'ok',
        'n_source': len(y_src) if X_src.size > 0 else 0,
        'n_target_train': len(y_tgt_train) if X_tgt_train.size > 0 else 0,
        'n_target_test': len(y_tgt_test),
        'train_time_s': round(elapsed, 2),
    }
    for k, v in metrics.items():
        result[k] = round(v, 4)

    return result


def _print_exp5_summary():
    """Print transfer learning summary from TRANSFER_CSV."""
    if not os.path.isfile(TRANSFER_CSV):
        return
    rows = _read_csv_dicts(TRANSFER_CSV)
    if not rows:
        return
    print("\n  --- Experiment 5 Summary (Cross-Dataset Transfer) ---")
    print(f"  {'Model':<15s} {'Scenario':<18s} {'MAE':>10s} {'RMSE':>10s} "
          f"{'MAPE':>10s}")
    print("  " + "-" * 62)

    groups = {}
    for r in rows:
        if r.get('status') != 'ok':
            continue
        key = (r['model'], r['scenario'])
        groups.setdefault(key, []).append(r)

    for key in sorted(groups.keys()):
        model, scenario = key
        rs = groups[key]
        maes  = [float(r.get('MAE', 0))  for r in rs]
        rmses = [float(r.get('RMSE', 0)) for r in rs]
        mapes = [float(r.get('MAPE', 0)) for r in rs]
        print(f"  {model:<15s} {scenario:<18s} "
              f"{np.mean(maes):>6.1f}+/-{np.std(maes):>4.1f} "
              f"{np.mean(rmses):>6.1f}+/-{np.std(rmses):>4.1f} "
              f"{np.mean(mapes):>6.1f}+/-{np.std(mapes):>4.1f}")
    print()


# ==========================================================================
#   Utility: read CSV as list of dicts
# ==========================================================================

def _read_csv_dicts(csv_path):
    """Read a CSV file into a list of dicts."""
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, 'r', newline='') as f:
        return list(csv.DictReader(f))


# ==========================================================================
#   Main entry point
# ==========================================================================

if __name__ == '__main__':
    t_global = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n")
    print("#" * 70)
    print("#")
    print("#   PINN-Knee: Early Knee-Point Prediction Experiments")
    print("#")
    print(f"#   Started: {timestamp}")
    print(f"#   Device:  {DEVICE}")
    print("#")
    print("#" * 70)

    print_config()

    # ------------------------------------------------------------------
    #   Load all cells with knees
    # ------------------------------------------------------------------
    cells = load_all_cells_with_knees(include_severson=True)

    if len(cells) == 0:
        print("\nERROR: No cells with valid knee-points found.")
        print("Check data paths and knee-detection settings in config.py")
        sys.exit(1)

    print(f"\nLoaded {len(cells)} cells with valid knee-points.")

    # ------------------------------------------------------------------
    #   Run all 5 experiments sequentially
    # ------------------------------------------------------------------
    run_experiment_1_main(cells)
    run_experiment_2_uq(cells)
    run_experiment_3_ablation(cells)
    run_experiment_4_early_prediction(cells)
    run_experiment_5_transfer(cells)

    # ------------------------------------------------------------------
    #   Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_global
    print("\n")
    print("#" * 70)
    print("#")
    print("#   All experiments complete!")
    print("#")
    print(f"#   Total time:  {_fmt_duration(total_time)}")
    print(f"#   Results in:  {RESULTS_DIR}")
    print("#")
    print("#   Output files:")
    for csv_path in [RESULTS_CSV, UQ_CSV, ABLATION_CSV, EARLY_CSV, TRANSFER_CSV]:
        exists = os.path.isfile(csv_path)
        basename = os.path.basename(csv_path)
        status = "OK" if exists else "MISSING"
        if exists:
            n_rows = sum(1 for _ in open(csv_path)) - 1
            print(f"#     {basename:<30s}  [{status}]  {n_rows} rows")
        else:
            print(f"#     {basename:<30s}  [{status}]")
    print("#")
    print("#" * 70)
