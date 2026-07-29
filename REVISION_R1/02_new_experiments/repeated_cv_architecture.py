"""Repeated cross-validation for the architecture comparison.

Fixes H8: np.array_split covers every cell, so the remainder is no longer dropped.
The full model and the physics-only variant share a single training run and differ
only at inference. The fold partition is verified before the runs start.
"""
import os, sys, csv, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, N_EPOCHS, LEARNING_RATE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import create_model, count_parameters
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from ablation_architecture import MatchedMLP, train_plain

OUT_CSV = os.path.join(HERE, 'repeated_cv_architecture.csv')
N_EARLY = 100
SPLIT_SEEDS = [42, 43, 44, 45, 46]      # 5 repetitions
MODEL_SEEDS = [0, 1]                    # 2 initialisation seeds per fold
N_FOLDS = 5
ZERO_LAMBDA = {k: 0.0 for k in PHYSICS_LAMBDA}


def kfold_split_all(cells, n_folds=5, seed=42):
    """Fixes H8: np.array_split covers every cell, so the remainder is not dropped."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(cells))
    folds = np.array_split(idx, n_folds)
    splits = []
    for f in range(n_folds):
        test_idx = folds[f]
        remaining = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        n_cal = max(1, len(remaining) // 4)
        splits.append(([cells[i] for i in remaining[n_cal:]],
                       [cells[i] for i in remaining[:n_cal]],
                       [cells[i] for i in test_idx]))
    return splits


def one_cell(rep_seed, fold, mseed, tr, cal, te):
    """Return {variant: MAE} for one (repetition, fold, seed)."""
    np.random.seed(mseed); torch.manual_seed(mseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mseed)

    Xtr, ytr, _, _ = build_feature_matrix(tr, N_EARLY)
    Xc, yc, _, _ = build_feature_matrix(cal, N_EARLY)
    Xte, yte, _, _ = build_feature_matrix(te, N_EARLY)
    if Xtr.size == 0 or Xte.size == 0:
        return None
    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
    nf = Xtr_n.shape[1]
    Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
    out = {'n_train': len(ytr), 'n_test': len(yte), 'n_cal': len(yc)}

    # --- full and physics_only share one training run and differ only at inference ---
    m = create_model('PINN_Knee', n_features=nf, device=DEVICE)
    m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, N_EARLY,
                           physics_lambda=dict(PHYSICS_LAMBDA),
                           X_val=Xc_n, y_val=yc if yc.size else None,
                           use_log_target=True)
    m.eval()
    with torch.no_grad():
        out['full'] = evaluate_knee_predictions(
            yte, m.predict_raw(Xte_t).cpu().numpy().ravel())['MAE']
        saved = m._delta_scale
        m._delta_scale = 0.0
        out['physics_only'] = evaluate_knee_predictions(
            yte, m.predict_raw(Xte_t).cpu().numpy().ravel())['MAE']
        m._delta_scale = saved

    # --- arch_only: bo 5 physics loss ---
    np.random.seed(mseed); torch.manual_seed(mseed)
    m2 = create_model('PINN_Knee', n_features=nf, device=DEVICE)
    m2, _ = train_pinn_knee(m2, Xtr_n, ytr, tr, N_EARLY,
                            physics_lambda=dict(ZERO_LAMBDA),
                            X_val=Xc_n, y_val=yc if yc.size else None,
                            use_log_target=True)
    m2.eval()
    with torch.no_grad():
        out['arch_only'] = evaluate_knee_predictions(
            yte, m2.predict_raw(Xte_t).cpu().numpy().ravel())['MAE']

    # --- mlp_matched ---
    np.random.seed(mseed); torch.manual_seed(mseed)
    m3 = MatchedMLP(nf).to(DEVICE)
    m3 = train_plain(m3, Xtr_n, np.log1p(ytr), Xc_n,
                     np.log1p(yc) if yc.size else None)
    with torch.no_grad():
        out['mlp_matched'] = evaluate_knee_predictions(
            yte, np.expm1(m3(Xte_t).cpu().numpy().ravel()))['MAE']

    # --- constant_median ---
    out['constant_median'] = evaluate_knee_predictions(
        yte, np.full(len(yte), float(np.median(ytr))))['MAE']

    out['n_params_pinn'] = count_parameters(m)
    out['n_params_mlp'] = count_parameters(m3)
    return out


def main():
    cells = load_paper_pool()
    print(f"Pool: {len(cells)} cells   n_early={N_EARLY}   device={DEVICE}")

    # verify the fold partition before running
    sp = kfold_split_all(cells, N_FOLDS, 42)
    seen = [c['name'] for _, _, te in sp for c in te]
    assert len(seen) == len(cells) == len(set(seen)), "fold partition is WRONG"
    print(f"Split check: {len(seen)} cells, each tested exactly once  OK")
    print(f"Test size per fold: {[len(te) for _, _, te in sp]}\n")

    done = set()
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV, encoding='utf-8')):
            done.add((r['split_seed'], r['fold'], r['model_seed']))
        print(f"Resume: already have {len(done)} points\n")

    total = len(SPLIT_SEEDS) * N_FOLDS * len(MODEL_SEEDS)
    i, t0 = 0, time.time()
    for rs in SPLIT_SEEDS:
        splits = kfold_split_all(cells, N_FOLDS, rs)
        for fold, (tr, cal, te) in enumerate(splits):
            for ms in MODEL_SEEDS:
                i += 1
                if (str(rs), str(fold), str(ms)) in done:
                    continue
                try:
                    r = one_cell(rs, fold, ms, tr, cal, te)
                except Exception as e:
                    print(f"  [{i}/{total}] rs={rs} f={fold} ms={ms} ERROR: {str(e)[:60]}")
                    continue
                if r is None:
                    continue
                r.update({'split_seed': rs, 'fold': fold, 'model_seed': ms})
                new = not os.path.exists(OUT_CSV)
                with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(r.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(r)
                eta = (time.time() - t0) / i * (total - i) / 60
                print(f"  [{i:>3d}/{total}] rs={rs} f={fold} ms={ms}  "
                      f"arch={r['arch_only']:6.1f}  mlp={r['mlp_matched']:6.1f}  "
                      f"full={r['full']:6.1f}   ETA {eta:5.1f}p")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")
    print("Run repeated_cv_analysis.py for the statistical results.")


if __name__ == '__main__':
    main()
