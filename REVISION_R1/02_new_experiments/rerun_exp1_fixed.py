"""Experiment 1 rerun after the bug fixes, used for the Major Revision.

Changes against the original:
  [H1] mse_loss was called with a prediction of shape (N,1) and a target of shape (N,),
       which PyTorch broadcasts silently to (N,N). Fixed by squeezing before the loss,
       in both training and validation.
  [H2] PINN-UQ now runs through the same corrected path, so the 445-cycle outlier is gone.
  [H5] Seeds 0, 1 and 2 on a single pool per budget.
  [H6] Test-cell identities are recorded per run so the MAE can later be recomputed on
       the intersection of 105 cells common to all three budgets, which removes the
       confound between the budget and the difficulty of its evaluation set.

Results are appended to rerun_exp1_fixed.csv after every run, so the job resumes after
a crash. The pool reproduces the submitted 117/113/108 cells.
"""
import os, sys, csv, copy, time, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import (DEVICE, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
                    MAX_CYCLE_LIFE, EARLY_CYCLE_COUNTS, RESULTS_DIR)
from features import build_feature_matrix, normalize_features
from models import (create_model, is_sequence_model, is_pinn_model,
                    is_classical_model, is_nn_model, count_parameters,
                    Ensemble_NN_Member)
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from run_experiments import _build_sequences, _kfold_split

OUT = os.path.join(HERE, 'rerun_exp1_fixed.csv')
PATIENCE, MIN_DELTA = 150, 0.01
SEEDS = [0, 1, 2]
MODELS = ['PINN_Knee', 'PINN_UQ', 'XGBoost', 'RandomForest', 'GaussianProcess',
          'Pure_NN', 'Ensemble_NN', 'Neural_ODE', 'LSTM', 'GRU',
          'Transformer', 'Informer', 'PatchTST', 'Bayesian_LSTM']


def train_nn_model_FIXED(model, X_tr, y_tr, X_val, y_val):
    """Corrected version of run_experiments.train_nn_model.

    The only difference from the original is .squeeze() before mse_loss,
    in both the training and the validation branch.
    """
    X_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    has_val = X_val is not None and y_val is not None and len(y_val) > 0
    if has_val:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    opt = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50, min_lr=1e-6)
    best, best_state, wait = float('inf'), None, 0

    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        opt.zero_grad()
        pred = model(X_t)
        loss = nn.functional.mse_loss(pred.squeeze(-1), y_t)      # <-- H1 fix
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        vl = loss.item()
        if has_val:
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(model(X_v).squeeze(-1), y_v).item()  # <-- H1 fix
            model.train()
        sch.step(vl)

        if vl < best - MIN_DELTA:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, epoch


def run_one(model_name, n_early, seed, tr, cal, te):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_tr, y_tr, _, _ = build_feature_matrix(tr, n_early)
    X_c,  y_c,  _, _ = build_feature_matrix(cal, n_early)
    X_te, y_te, _, idx_te = build_feature_matrix(te, n_early)
    if X_tr.size == 0 or X_te.size == 0:
        return None

    X_tr_n, X_te_n, X_c_n, _ = normalize_features(X_tr, X_te,
                                                  X_c if X_c.size else None)
    nf = X_tr_n.shape[1]
    use_log = is_nn_model(model_name) and not is_sequence_model(model_name)
    y_tr_t = np.log1p(y_tr) if use_log else y_tr
    y_c_t = np.log1p(y_c) if (use_log and y_c.size) else y_c

    t0 = time.time()
    if is_classical_model(model_name):
        m = create_model(model_name)
        m.fit(X_tr_n, y_tr)
        y_pred = m.predict(X_te_n)
        npar = 0
    else:
        m = create_model(model_name, n_features=nf, device=DEVICE)
        if is_sequence_model(model_name):
            S_tr = _build_sequences(tr, n_early)
            S_te = _build_sequences(te, n_early)
            S_c = _build_sequences(cal, n_early) if cal else None
            if S_tr is None or S_te is None:
                return None
            m, _ = train_nn_model_FIXED(m, S_tr, y_tr, S_c, y_c if y_c.size else None)
            with torch.no_grad():
                y_pred = m(torch.tensor(S_te, dtype=torch.float32)
                           .to(DEVICE)).cpu().numpy().squeeze()
        elif is_pinn_model(model_name):
            m, _ = train_pinn_knee(m, X_tr_n, y_tr, tr, n_early,
                                   X_val=X_c_n, y_val=y_c if y_c.size else None,
                                   use_log_target=use_log)
            with torch.no_grad():
                r = m(torch.tensor(X_te_n, dtype=torch.float32)
                      .to(DEVICE)).cpu().numpy().squeeze()
            y_pred = np.expm1(r) if use_log else r
        elif model_name == 'Ensemble_NN':
            from config import ENSEMBLE_SIZE
            X_te_t = torch.tensor(X_te_n, dtype=torch.float32).to(DEVICE)
            ps = []
            for k in range(ENSEMBLE_SIZE):
                torch.manual_seed(seed * 1000 + k * 17 + 3)
                np.random.seed(seed * 1000 + k * 17 + 3)
                mem = Ensemble_NN_Member(n_features=nf).to(DEVICE)
                mem, _ = train_nn_model_FIXED(mem, X_tr_n, y_tr_t, X_c_n,
                                              y_c_t if y_c.size else None)
                with torch.no_grad():
                    pl = mem(X_te_t).cpu().numpy().squeeze()
                ps.append(np.expm1(pl) if use_log else pl)
                m = mem
            y_pred = np.array(ps).mean(axis=0)
        else:
            m, _ = train_nn_model_FIXED(m, X_tr_n, y_tr_t, X_c_n,
                                        y_c_t if y_c.size else None)
            with torch.no_grad():
                r = m(torch.tensor(X_te_n, dtype=torch.float32)
                      .to(DEVICE)).cpu().numpy().squeeze()
            y_pred = np.expm1(r) if use_log else r
        npar = count_parameters(m)

    el = time.time() - t0
    met = evaluate_knee_predictions(y_te, y_pred)
    row = {'model': model_name, 'n_early': n_early, 'seed': seed,
           'n_train': len(y_tr), 'n_cal': len(y_c), 'n_test': len(y_te),
           'n_params': npar, 'train_time_s': round(el, 2),
           # H6: record cell identities and predictions so the intersection set can be recomputed
           'test_cells': '|'.join(te[i]['name'] for i in idx_te),
           'y_true': '|'.join(f'{v:.0f}' for v in y_te),
           'y_pred': '|'.join(f'{v:.2f}' for v in np.atleast_1d(y_pred))}
    row.update({k: round(v, 4) for k, v in met.items()})
    return row


def load_paper_pool():
    """Reproduce exactly the 117-cell pool used in the submission.

    load_all_cells_with_knees() additionally filters on has_knee_point -> chi con 113,
    and yields 113/110/105 instead of 117/113/108, a consequence of the loader
    the original "Paper 7" source is missing, so the code falls back to the cache branch.

    Verified: filtering only on knee_cycle is not None gives 117 cells, and
    build_feature_matrix returns exactly 117/113/108, as implied by the column
    n_train+n_cal+n_test of all_experiments.csv.
    """
    import pickle
    with open(os.path.join(RESULTS_DIR, '_severson_cache.pkl'), 'rb') as f:
        raw = pickle.load(f)
    return [c for c in raw if c.get('knee_cycle') is not None]


def main():
    cells = load_paper_pool()
    print(f"Pool: {len(cells)} cells (reproducing the submitted pool)\nDevice: {DEVICE}\n")

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, encoding='utf-8')):
            done.add((r['model'], r['n_early'], r['seed'], r.get('fold', '')))
        print(f"Resume: already have {len(done)} run\n")

    splits = _kfold_split(cells, n_folds=5, seed=42)
    total = len(MODELS) * len(EARLY_CYCLE_COUNTS) * len(SEEDS) * 5
    cnt, t0 = 0, time.time()

    for ne in EARLY_CYCLE_COUNTS:
        for mn in MODELS:
            for fold, (tr, cal, te) in enumerate(splits):
                for seed in SEEDS:
                    cnt += 1
                    key = (mn, str(ne), str(seed), str(fold))
                    if key in done:
                        continue
                    try:
                        row = run_one(mn, ne, seed, tr, cal, te)
                    except Exception as e:
                        print(f"  [{cnt}/{total}] {mn} ne={ne} f={fold} s={seed} ERROR: {str(e)[:70]}")
                        continue
                    if row is None:
                        continue
                    row['fold'] = fold
                    new = not os.path.exists(OUT)
                    with open(OUT, 'a', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if new:
                            w.writeheader()
                        w.writerow(row)
                    el = time.time() - t0
                    eta = el / max(cnt, 1) * (total - cnt) / 60
                    print(f"  [{cnt}/{total}] {mn:<16s} ne={ne:>3d} f={fold} s={seed}  "
                          f"MAE={row['MAE']:7.1f}  [{row['train_time_s']:5.1f}s]  ETA {eta:5.1f}p")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == '__main__':
    main()
