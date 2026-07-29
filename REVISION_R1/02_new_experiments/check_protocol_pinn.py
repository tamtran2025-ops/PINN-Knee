"""Decisive check: give PINN-Knee the same amount of data as the classical baselines.

An earlier version of this check (check_protocol_fairness.py) gave the classical
models the extra calibration split but left PINN-Knee as in Table 1, which is a biased
comparison. This script runs PINN-Knee under both allocations, one model per run and
no ensembling:

  A. train on the training split, early stopping on calibration  (as in Table 1)
  B. train on train + calibration, no early stopping             (same data as classical)

Branch A must reproduce the Table 1 value. If it does not, the harness is wrong and
the results must not be used.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments')):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter('ignore')

from config import DEVICE, PHYSICS_LAMBDA                        # noqa: E402
from features import build_feature_matrix, normalize_features    # noqa: E402
from models import create_model                                  # noqa: E402
from train import train_pinn_knee                                # noqa: E402
from metrics import evaluate_knee_predictions                    # noqa: E402
from rerun_exp1_fixed import load_paper_pool                     # noqa: E402
from run_experiments import _kfold_split                         # noqa: E402

SEEDS = [0, 1, 2]
TABLE1 = {50: 159.2, 100: 139.6, 150: 117.4}


def run(use_cal_for_training):
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    out = {}
    for ne in (50, 100, 150):
        rec = []
        for tr, cal, te in splits:
            fit_cells = (tr + cal) if use_cal_for_training else tr
            for seed in SEEDS:
                np.random.seed(seed)
                torch.manual_seed(seed)
                Xtr, ytr, _, _ = build_feature_matrix(fit_cells, ne)
                Xc, yc, _, _ = build_feature_matrix(cal, ne)
                Xte, yte, _, _ = build_feature_matrix(te, ne)
                if Xtr.size == 0 or Xte.size == 0:
                    continue
                Xtr_n, Xte_n, Xc_n, _ = normalize_features(
                    Xtr, Xte, Xc if Xc.size else None)
                m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
                if use_cal_for_training:
                    # the calibration split is now inside the training set, so it is no longer a validation set
                    m, _ = train_pinn_knee(m, Xtr_n, ytr, fit_cells, ne,
                                           physics_lambda=dict(PHYSICS_LAMBDA),
                                           X_val=None, y_val=None, use_log_target=True)
                else:
                    m, _ = train_pinn_knee(m, Xtr_n, ytr, fit_cells, ne,
                                           physics_lambda=dict(PHYSICS_LAMBDA),
                                           X_val=Xc_n, y_val=yc if yc.size else None,
                                           use_log_target=True)
                m.eval()
                with torch.no_grad():
                    Xe = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
                    pred = np.atleast_1d(m.predict_raw(Xe).cpu().numpy().ravel())
                met = evaluate_knee_predictions(yte, pred)
                rec.append((met['MAE'], met['MedianAE'],
                            met.get('Within_100', np.nan)))
        out[ne] = np.array(rec, float).mean(axis=0)
    return out


def main():
    print("A. PINN trained on `tr`, early stopping on `cal` (as in Table 1) ...")
    a = run(False)
    print("B. PINN trained on `tr+cal`, no early stopping ...")
    b = run(True)

    print("\nControl: branch A against Table 1")
    ok = True
    for ne in (50, 100, 150):
        got = a[ne][0]
        good = abs(got - TABLE1[ne]) <= 2.0
        ok &= good
        print(f"   n_early={ne:>3}: A={got:>6.1f}   Table 1={TABLE1[ne]:>6.1f}   "
              f"{'MATCH' if good else '>>> MISMATCH <<<'}")
    if not ok:
        print("\nWARNING: branch A does not reproduce Table 1. Read the results below with caution.")

    # classical baselines on train + calibration, taken from the previous run
    CLASSICAL_TRCAL = {50: ('GaussianProcess', 148.08, 78.1, 0.58),
                       100: ('RandomForest', 134.35, 83.3, 0.55),
                       150: ('GaussianProcess', 114.18, 60.9, 0.65)}

    print("\n" + "=" * 86)
    print("EQUAL DATA ALLOCATION (tr+cal): PINN against the best classical baseline")
    print("=" * 86)
    print(f"{'n_early':>8}{'phuong phap':>34}{'MAE':>9}{'MedAE':>9}{'W100':>8}")
    print("-" * 86)
    for ne in (50, 100, 150):
        nm, cm, cmed, cw = CLASSICAL_TRCAL[ne]
        print(f"{ne:>8}{'PINN trained on tr (Table 1)':>34}"
              f"{a[ne][0]:>9.2f}{a[ne][1]:>9.1f}{100*a[ne][2]:>7.0f}%")
        print(f"{ne:>8}{'PINN trained on tr+cal':>34}"
              f"{b[ne][0]:>9.2f}{b[ne][1]:>9.1f}{100*b[ne][2]:>7.0f}%")
        print(f"{ne:>8}{f'{nm} on tr+cal':>34}{cm:>9.2f}{cmed:>9.1f}{100*cw:>7.0f}%")
        win = 'PINN' if b[ne][0] < cm else 'classical'
        print(f"{'':>8}   -> wins on MAE: {win}   (gap {abs(b[ne][0]-cm):.2f} cycles)")
        print()

    rows = []
    for ne in (50, 100, 150):
        rows.append(dict(n_early=ne, pinn_tr=a[ne][0], pinn_trcal=b[ne][0],
                         pinn_tr_medae=a[ne][1], pinn_trcal_medae=b[ne][1],
                         classical_trcal=CLASSICAL_TRCAL[ne][1]))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'check_protocol_pinn.csv'), index=False)
    print("Wrote: check_protocol_pinn.csv")


if __name__ == '__main__':
    main()
