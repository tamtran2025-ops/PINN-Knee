"""
Training Loop with Checkpoint & Resume
=========================================
- Saves result to CSV after EVERY completed run
- On resume, skips already-completed runs
- Maximum data loss if crash: 1 run (~3-5 min)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os

from config import (N_EPOCHS, LEARNING_RATE, SCHEDULER_T_MAX, DEVICE,
                    N_COLLOCATION, WINDOW_SIZE, ENSEMBLE_SIZE,
                    MC_DROPOUT_SAMPLES, CONFORMAL_CAL_FRACTION,
                    CONFORMAL_ALPHA, CHECKPOINTS_DIR, EOL_THRESHOLD)
from models import (create_model, is_sequence_model, is_pinn_model,
                     is_uq_model, count_parameters, Ensemble_NN_Member, PINN_UQ)
# Knee-project imports (these functions have different names from Paper 7)
try:
    from data_loader import load_all_cells_with_knees, get_train_cal_test_split
except ImportError:
    pass

try:
    from uncertainty import (KneeConformalPredictor, mc_dropout_knee,
                             mc_dropout_prediction_interval,
                             ensemble_knee_predict, ensemble_prediction_interval)
except ImportError:
    pass

try:
    from metrics import evaluate_knee_predictions
except ImportError:
    pass


# Helper: create sliding-window sequences for sequence models
def create_sequences(data, window_size):
    """Create sliding window sequences from 1D array."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])


# ================================================================
# PINN TRAINING
# ================================================================
def train_pinn(model, train_data, n_epochs=N_EPOCHS, lr=LEARNING_RATE, verbose=False):
    """Train PINN model with physics-informed loss.

    Args:
        model: PINN_UQ model
        train_data: dict with cycles, capacity, Q0, n_max
        n_epochs: number of epochs
        lr: learning rate
        verbose: print progress

    Returns:
        model: trained model
        train_time: training time in seconds
        loss_history: list of loss values
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    cycles = train_data['cycles']
    capacity = train_data['capacity']
    Q0 = train_data['Q0']
    n_max = train_data['n_max']

    # Normalize inputs
    n_t = torch.FloatTensor(cycles / n_max).unsqueeze(1).to(DEVICE)
    Q_t = torch.FloatTensor(capacity).unsqueeze(1).to(DEVICE)

    # Collocation points for physics loss
    n_col = torch.linspace(0.005, 1.0, N_COLLOCATION).unsqueeze(1).to(DEVICE)

    loss_history = []
    t_start = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss, loss_dict = model.compute_loss(n_t, Q_t, n_col)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss_dict['total'])

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={loss_dict['total']:.6f} "
                  f"(data={loss_dict['data']:.6f} phys={loss_dict['phys']:.6f})")

    train_time = time.time() - t_start
    return model, train_time, loss_history


# ================================================================
# STANDARD NN TRAINING (Pure NN, Neural ODE)
# ================================================================
def train_direct_model(model, train_data, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train a direct (non-sequence) model with MSE loss.

    Input: normalized cycle number  output: capacity
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    cycles = train_data['cycles']
    capacity = train_data['capacity']
    n_max = train_data['n_max']

    n_t = torch.FloatTensor(cycles / n_max).unsqueeze(1).to(DEVICE)
    Q_t = torch.FloatTensor(capacity).unsqueeze(1).to(DEVICE)

    t_start = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(n_t)
        loss = criterion(pred, Q_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_time = time.time() - t_start
    return model, train_time


# ================================================================
# SEQUENCE MODEL TRAINING (LSTM, GRU, Transformer, Informer, Bayesian)
# ================================================================
def train_sequence_model(model, train_data, n_epochs=N_EPOCHS, lr=LEARNING_RATE,
                         window_size=WINDOW_SIZE):
    """Train a sequence model with sliding window.

    Input: window of past capacities  output: next capacity
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    capacity = train_data['capacity']

    # Create sequences
    X, y = create_sequences(capacity, window_size)
    if len(X) == 0:
        return model, 0.0

    # Normalize
    Q_min, Q_max = capacity.min(), capacity.max()
    Q_range = Q_max - Q_min if Q_max > Q_min else 1.0
    X_norm = (X - Q_min) / Q_range
    y_norm = (y - Q_min) / Q_range

    X_t = torch.FloatTensor(X_norm).to(DEVICE)
    y_t = torch.FloatTensor(y_norm).unsqueeze(1).to(DEVICE)

    t_start = time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_time = time.time() - t_start

    # Store normalization params for prediction
    model._Q_min = Q_min
    model._Q_max = Q_max
    model._Q_range = Q_range

    return model, train_time


# ================================================================
# ENSEMBLE TRAINING
# ================================================================
def train_ensemble(train_data, n_members=ENSEMBLE_SIZE, n_epochs=N_EPOCHS,
                   lr=LEARNING_RATE):
    """Train ensemble of Pure NNs with different seeds.

    Returns:
        models: list of trained models
        train_time: total training time
    """
    models = []
    t_start = time.time()

    for i in range(n_members):
        torch.manual_seed(42 + i * 1000)
        model = Ensemble_NN_Member().to(DEVICE)
        model, _ = train_direct_model(model, train_data, n_epochs, lr)
        models.append(model)

    train_time = time.time() - t_start
    return models, train_time


# ================================================================
# PREDICTION
# ================================================================
def predict_direct(model, test_data, train_data):
    """Predict capacity for direct models (PINN, Pure NN, Neural ODE)."""
    n_max = train_data['n_max']
    cycles = test_data['cycles']
    n_t = torch.FloatTensor(cycles / n_max).unsqueeze(1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(n_t).squeeze().cpu().numpy()

    return pred


def predict_sequence(model, train_data, test_data, window_size=WINDOW_SIZE):
    """Autoregressive prediction for sequence models."""
    capacity = train_data['capacity']
    Q_min = getattr(model, '_Q_min', capacity.min())
    Q_range = getattr(model, '_Q_range', capacity.max() - capacity.min())
    if Q_range == 0:
        Q_range = 1.0

    # Start with last window from training
    window = list((capacity[-window_size:] - Q_min) / Q_range)
    n_predict = len(test_data['cycles'])
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(n_predict):
            x = torch.FloatTensor([window[-window_size:]]).to(DEVICE)
            pred_norm = model(x).squeeze().item()
            pred_norm = max(0, min(1, pred_norm))  # Clip
            predictions.append(pred_norm * Q_range + Q_min)
            window.append(pred_norm)

    return np.array(predictions)


# ================================================================
# FULL SINGLE RUN: TRAIN + PREDICT + EVALUATE
# ================================================================
def run_single_experiment(cell, fraction, model_name, seed):
    """Run one complete experiment: train, predict, evaluate.

    Args:
        cell: cell dict from data_loader
        fraction: training fraction (e.g., 0.25)
        model_name: model name string
        seed: random seed

    Returns:
        result: dict with all metrics and metadata
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare data
    train_data, test_data = prepare_train_test(cell, fraction)
    Q0 = cell['Q0']

    if len(test_data['capacity']) < 5:
        return None  # Not enough test data

    # --- TRAIN ---
    if model_name == 'Ensemble_NN':
        models, train_time = train_ensemble(train_data)
        n_params = count_parameters(models[0]) * len(models)
    else:
        model = create_model(model_name, Q0=Q0)
        n_params = count_parameters(model)

        if is_pinn_model(model_name):
            model, train_time, _ = train_pinn(model, train_data)
        elif is_sequence_model(model_name):
            model, train_time = train_sequence_model(model, train_data)
        else:
            model, train_time = train_direct_model(model, train_data)

    # --- PREDICT ---
    if model_name == 'Ensemble_NN':
        # Ensemble prediction
        n_max = train_data['n_max']
        cycles_test = test_data['cycles']
        n_t = torch.FloatTensor(cycles_test / n_max).unsqueeze(1).to(DEVICE)
        mean, std, _ = ensemble_predict(models, n_t)
        pred = mean
        lower, upper = ensemble_prediction_interval(mean, std)
    elif is_sequence_model(model_name):
        pred = predict_sequence(model, train_data, test_data)
        mean, std, lower, upper = None, None, None, None

        # MC Dropout UQ for Bayesian_LSTM
        if model_name == 'Bayesian_LSTM':
            # Autoregressive MC Dropout (simplified)
            capacity = train_data['capacity']
            Q_min = getattr(model, '_Q_min', capacity.min())
            Q_range = getattr(model, '_Q_range', capacity.max() - capacity.min())
            if Q_range == 0:
                Q_range = 1.0

            mc_preds = []
            for _ in range(MC_DROPOUT_SAMPLES):
                model.train()  # Dropout ON
                window = list((capacity[-WINDOW_SIZE:] - Q_min) / Q_range)
                preds_i = []
                with torch.no_grad():
                    for j in range(len(test_data['cycles'])):
                        x = torch.FloatTensor([window[-WINDOW_SIZE:]]).to(DEVICE)
                        p = model(x).squeeze().item()
                        p = max(0, min(1, p))
                        preds_i.append(p * Q_range + Q_min)
                        window.append(p)
                mc_preds.append(preds_i)

            mc_preds = np.array(mc_preds)
            mean = np.mean(mc_preds, axis=0)
            std = np.std(mc_preds, axis=0)
            pred = mean
            lower, upper = mc_dropout_prediction_interval(mean, std)
            model.eval()
    else:
        pred = predict_direct(model, test_data, train_data)
        mean, std, lower, upper = None, None, None, None

        # MC Dropout UQ for PINN_UQ
        if model_name == 'PINN_UQ':
            n_max = train_data['n_max']
            cycles_test = test_data['cycles']
            n_t = torch.FloatTensor(cycles_test / n_max).unsqueeze(1).to(DEVICE)
            mean, std, _ = mc_dropout_predict(model, n_t)
            pred = mean
            lower, upper = mc_dropout_prediction_interval(mean, std)

    # --- CONFORMAL PREDICTION (for PINN_UQ) ---
    cp_lower, cp_upper = None, None
    if model_name == 'PINN_UQ':
        # Split training data for calibration
        n_cal = max(5, int(len(train_data['capacity']) * CONFORMAL_CAL_FRACTION))
        cal_cycles = train_data['cycles'][-n_cal:]
        cal_capacity = train_data['capacity'][-n_cal:]

        n_max = train_data['n_max']
        cal_n = torch.FloatTensor(cal_cycles / n_max).unsqueeze(1).to(DEVICE)

        cp = ConformalPredictor(alpha=CONFORMAL_ALPHA)
        cp.calibrate(model, cal_n, cal_capacity)

        test_n = torch.FloatTensor(test_data['cycles'] / n_max).unsqueeze(1).to(DEVICE)
        _, cp_lower, cp_upper = cp.predict(model, test_n)

    # --- EVALUATE ---
    true_cap = test_data['capacity']

    # Compute RUL
    true_rul = test_data['rul']
    pred_rul = None
    if pred is not None:
        # Estimate RUL from predicted capacity curve
        eol_cap = Q0 * EOL_THRESHOLD
        full_pred = np.concatenate([train_data['capacity'], pred])
        pred_rul_full, pred_eol = compute_rul(full_pred, Q0, EOL_THRESHOLD)
        pred_rul = pred_rul_full[len(train_data['capacity']):]

        # Ensure same length
        min_len = min(len(pred_rul), len(true_rul))
        pred_rul = pred_rul[:min_len]
        true_rul_eval = true_rul[:min_len]
    else:
        true_rul_eval = true_rul

    # Point metrics
    min_len = min(len(pred), len(true_cap))
    eval_results = evaluate_predictions(
        y_true=true_cap[:min_len],
        y_pred=pred[:min_len],
        y_lower=lower[:min_len] if lower is not None else None,
        y_upper=upper[:min_len] if upper is not None else None,
        y_mean=mean[:min_len] if mean is not None else None,
        y_std=std[:min_len] if std is not None else None,
        true_rul=true_rul_eval if pred_rul is not None else None,
        pred_rul=pred_rul[:min_len] if pred_rul is not None else None,
    )

    # Conformal prediction metrics (separate)
    if cp_lower is not None:
        from metrics import picp, mpiw
        eval_results['CP_PICP'] = picp(true_cap[:min_len], cp_lower[:min_len], cp_upper[:min_len])
        eval_results['CP_MPIW'] = mpiw(cp_lower[:min_len], cp_upper[:min_len])

    # Physics params (PINN only)
    physics_params = None
    if model_name == 'PINN_UQ' and hasattr(model, 'get_physics_params'):
        physics_params = model.get_physics_params()

    # --- BUILD RESULT ---
    result = {
        'cell': cell['name'],
        'dataset': cell['dataset'],
        'fraction': fraction,
        'model': model_name,
        'seed': seed,
        'train_cycles': len(train_data['cycles']),
        'test_cycles': len(test_data['cycles']),
        'train_time_s': train_time,
        'n_params': n_params,
        **eval_results,
    }

    if physics_params:
        for k, v in physics_params.items():
            result[f'phys_{k}'] = v

    return result


# ================================================================
# BATCH RUNNER WITH CHECKPOINT
# ================================================================
def load_completed_runs(csv_path):
    """Load already-completed runs from CSV to skip on resume."""
    completed = set()
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['cell'], row['fraction'], row['model'], row['seed'])
                completed.add(key)
    return completed


def append_result_to_csv(result, csv_path):
    """Append a single result to CSV (create file with header if needed)."""
    import csv
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def run_all_experiments(cells, fractions, model_names, seeds, csv_path,
                        verbose=True):
    """Run all experiments with checkpoint/resume.

    Saves result after EVERY run. Skips already-completed runs.
    Maximum data loss on crash: 1 run.

    Args:
        cells: list of cell dicts
        fractions: list of training fractions
        model_names: list of model names
        seeds: list of seeds
        csv_path: path to results CSV
        verbose: print progress
    """
    # Count total runs
    total_runs = len(cells) * len(fractions) * len(model_names) * len(seeds)

    # Load completed runs
    completed = load_completed_runs(csv_path)
    n_completed = len(completed)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT RUNNER")
        print(f"{'=' * 70}")
        print(f"  Total planned:   {total_runs}")
        print(f"  Already done:    {n_completed}")
        print(f"  Remaining:       {total_runs - n_completed}")
        print(f"  Results CSV:     {csv_path}")
        print(f"  Device:          {DEVICE}")
        print(f"{'=' * 70}\n")

    run_count = n_completed
    t_global = time.time()

    for cell in cells:
        for frac in fractions:
            for model_name in model_names:
                for seed in seeds:
                    # Check if already done
                    key = (cell['name'], str(frac), model_name, str(seed))
                    if key in completed:
                        continue

                    run_count += 1

                    if verbose:
                        elapsed = time.time() - t_global
                        remaining_runs = total_runs - run_count
                        if run_count > n_completed + 1:
                            avg_time = elapsed / (run_count - n_completed - 1)
                            eta = avg_time * remaining_runs
                            eta_str = f"ETA: {eta/3600:.1f}h"
                        else:
                            eta_str = "ETA: calculating..."

                        print(f"  [{run_count}/{total_runs}] "
                              f"{cell['name']} f={frac} {model_name} s={seed} "
                              f"... ", end='', flush=True)

                    try:
                        result = run_single_experiment(cell, frac, model_name, seed)

                        if result is not None:
                            # SAVE IMMEDIATELY after each run
                            append_result_to_csv(result, csv_path)

                            if verbose:
                                rmse_val = result.get('RMSE', np.nan)
                                t = result.get('train_time_s', 0)
                                print(f"RMSE={rmse_val:.4f} [{t:.0f}s] {eta_str}")
                        else:
                            if verbose:
                                print(f"SKIPPED (insufficient data)")

                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {str(e)[:80]}")
                        # Log error but continue
                        error_result = {
                            'cell': cell['name'],
                            'dataset': cell['dataset'],
                            'fraction': frac,
                            'model': model_name,
                            'seed': seed,
                            'error': str(e)[:200],
                        }
                        # Don't save errors to main CSV, just log
                        import traceback
                        traceback.print_exc()
                        continue

    # Final summary
    elapsed_total = time.time() - t_global
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ALL EXPERIMENTS COMPLETE")
        print(f"  Total runs: {run_count}")
        print(f"  Total time: {elapsed_total/3600:.1f} hours")
        print(f"  Results: {csv_path}")
        print(f"{'=' * 70}")


# ================================================================
# PINN Knee Training (for knee-point prediction project)
# ================================================================
def train_pinn_knee(model, X_train, y_train, cells=None, n_early=None,
                    n_epochs=N_EPOCHS, lr=LEARNING_RATE, physics_lambda=None,
                    X_val=None, y_val=None, use_log_target=True, verbose=False):
    """Train a PINN_Knee model for knee-point prediction.

    Args:
        model: PINN_Knee model (nn.Module with compute_loss)
        X_train: (N, n_features) numpy array of input features
        y_train: (N,) numpy array of target knee cycles (RAW, not log-transformed)
        cells: list of cell dicts (for physics constraints context)
        n_early: number of early cycles used
        n_epochs: training epochs
        lr: learning rate
        physics_lambda: dict of physics loss weights
        X_val, y_val: optional validation data for early stopping (RAW)
        use_log_target: if True, apply log1p transform to targets
        verbose: print progress

    Returns:
        model: trained model
        history: dict with train_loss, val_loss lists
    """
    from config import MAX_CYCLE_LIFE, PHYSICS_LAMBDA as DEFAULT_PHYSICS_LAMBDA

    if physics_lambda is None:
        physics_lambda = DEFAULT_PHYSICS_LAMBDA

    # Log-transform targets (same as run_single_knee_experiment)
    if use_log_target:
        y_train_t = np.log1p(y_train)
        y_val_t = np.log1p(y_val) if y_val is not None else None
    else:
        y_train_t = y_train
        y_val_t = y_val

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train_t, dtype=torch.float32).to(DEVICE)

    has_val = X_val is not None and y_val_t is not None
    if has_val:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_v = torch.tensor(y_val_t, dtype=torch.float32).to(DEVICE)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # ReduceLROnPlateau  -  better adaptation than CosineAnnealing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=80, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience = 200
    wait = 0
    min_delta = 1e-4

    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        model_class = type(model).__name__
        if model_class == 'PINN_Knee' and hasattr(model, 'compute_loss'):
            # Residual Physics v2: pass MAX_CYCLE_LIFE (raw), the model
            # handles log-space internally for the data loss.
            loss = model.compute_loss(X_t, y_t, knee_max=MAX_CYCLE_LIFE,
                                      physics_lambda=physics_lambda)
        else:
            pred = model(X_t)
            loss = nn.functional.mse_loss(pred.squeeze(), y_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        history['train_loss'].append(train_loss)

        # Validation loss (MSE in log-space, consistent with data loss)
        if has_val:
            model.eval()
            with torch.no_grad():
                v_pred = model(X_v).squeeze()
                # Model returns log-space output via its forward()
                val_loss = nn.functional.mse_loss(v_pred, y_v).item()
            model.train()
        else:
            val_loss = train_loss
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Early stopping on validation loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}, val_loss={val_loss:.4f}")
                break

        if verbose and epoch % 200 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:>5d}/{n_epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
