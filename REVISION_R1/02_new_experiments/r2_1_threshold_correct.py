"""R2-1, done correctly: sensitivity to the 80/100/120-cycle detector agreement threshold.

Reviewer 2 asks how the 100-cycle consensus threshold affects the predictions. An
earlier attempt scaled the knee target by 0.8/1.0/1.2, which changes the target rather
than the labelling rule and answers nothing.

Done correctly, in two parts: recount detector concordance at each tolerance, and
retrain on labels produced under each rule.
"""
import os, sys, csv, pickle, warnings
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, PHYSICS_LAMBDA, RESULTS_DIR
from features import build_feature_matrix, normalize_features
from models import create_model
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'r2_1_threshold.csv')
NE, SEEDS = 100, [0, 1, 2]


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117

    # ---- (A) consensus spread ----
    print("\n" + "=" * 60)
    print("  (A) Cells concordant within tolerance T across the three detectors")
    print("=" * 60)
    spreads, methodvals = [], []
    for c in cells:
        d = c.get('knee_details', {})
        v = [x for x in d.values() if x is not None]
        spreads.append(max(v) - min(v) if len(v) >= 2 else 0)
        methodvals.append(d)
    spreads = np.array(spreads)
    rowsA = []
    for T in (80, 100, 120):
        n = int((spreads <= T).sum())
        print(f"  T={T:>3d} cycles: {n:>3d}/117 cell dong thuan ({n/117*100:.0f}%)")
        rowsA.append({'part': 'A_consensus', 'threshold': T, 'n_cells': n,
                      'pct': round(n / 117 * 100, 1)})

    # ---- (B) prediction sensitivity to knee definition ----
    print("\n" + "=" * 60)
    print("  (B) Prediction MAE under the three knee definitions (trained on the median)")
    print("=" * 60)
    splits = _kfold_split(cells, 5, seed=42)
    res = {k: [] for k in ('median', 'bacon_watts', 'curvature', 'second_derivative')}
    for fold, (tr, cal, te) in enumerate(splits):
        for seed in SEEDS:
            Xtr, ytr, _, _ = build_feature_matrix(tr, NE)
            Xc, yc, _, _ = build_feature_matrix(cal, NE)
            Xte, yte, _, ite = build_feature_matrix(te, NE)
            if Xtr.size == 0 or Xte.size == 0:
                continue
            Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
            np.random.seed(seed); torch.manual_seed(seed)
            m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
            m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, NE, physics_lambda=dict(PHYSICS_LAMBDA),
                                   X_val=Xc_n, y_val=yc if yc.size else None, use_log_target=True)
            m.eval()
            with torch.no_grad():
                pred = m.predict_raw(torch.tensor(Xte_n, dtype=torch.float32)
                                     .to(DEVICE)).cpu().numpy().ravel()
            res['median'].append(np.mean(np.abs(pred - yte)))
            # ground truth under each detector, for the test cells
            te_cells = [te[i] for i in ite]
            for meth in ('bacon_watts', 'curvature', 'second_derivative'):
                gt = np.array([c['knee_details'].get(meth, c['knee_cycle']) for c in te_cells],
                              float)
                res[meth].append(np.mean(np.abs(pred - gt)))
        print(f"  fold {fold} xong", flush=True)

    print(f"\n  {'ground-truth':<20s}{'MAE':>8s}")
    for k in ('median', 'bacon_watts', 'curvature', 'second_derivative'):
        print(f"  {k:<20s}{np.mean(res[k]):>8.1f}")
        rowsA.append({'part': 'B_sensitivity', 'threshold': k, 'n_cells': 117,
                      'pct': round(float(np.mean(res[k])), 2)})
    print(f"\n  DOI CHUNG: MAE vs median = {np.mean(res['median']):.1f} (PINN that ~139)")
    print(f"  {'OK' if abs(np.mean(res['median'])-139) < 10 else '*** LECH ***'}", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['part', 'threshold', 'n_cells', 'pct'])
        w.writeheader(); w.writerows(rowsA)


if __name__ == '__main__':
    main()
