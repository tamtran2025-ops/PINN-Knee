"""R2-6: feature robustness under structured sensor perturbations, not just Gaussian noise.

The submitted Response quoted "+4.1% versus +18.7%" with no file behind it. This
script produces the numbers.

Perturbations applied to the test capacity readings before features are extracted:
a systematic multiplicative bias and a slow correlated drift. The model is trained on
clean data and tested on all three conditions; PINN-Knee and XGBoost are compared on
the percentage increase in MAE relative to clean.

The scaler is the one fitted on clean training data, as it would be in deployment.
"""
import os, sys, csv, time, warnings, copy
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import create_model
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'sensor_drift.csv')
N_EARLY_LIST = [50, 100, 150]
SEEDS = [0, 1, 2]
BIAS = 0.05      # +5% systematic offset
DRIFT = 0.02     # toi +2% drift tuyen tinh


def corrupt(cells, mode):
    """Return a copy of the cells with perturbed capacity, leaving the originals untouched."""
    out = []
    for c in cells:
        c2 = dict(c)
        cap = np.asarray(c['capacity'], dtype=float).copy()
        if mode == 'sensor_bias':
            cap = cap * (1.0 + BIAS)
        elif mode == 'corr_drift':
            n = len(cap)
            cap = cap * (1.0 + DRIFT * np.arange(n) / max(n - 1, 1))
        c2['capacity'] = cap
        out.append(c2)
    return out


def fit_xgb(Xtr, ytr):
    from models import XGBoostKnee
    m = XGBoostKnee()
    m.fit(Xtr, np.log1p(ytr))          # classical dung log (fair)
    return m


def run(ne, seed, tr, cal, te):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- train tren CLEAN ---
    Xtr, ytr, _, _ = build_feature_matrix(tr, ne)
    Xc, yc, _, _ = build_feature_matrix(cal, ne)
    if Xtr.size == 0:
        return []
    # scaler tu train (clean)
    Xtr_n, _, Xc_n, scaler = normalize_features(Xtr, Xtr, Xc if Xc.size else None)

    pinn = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
    pinn, _ = train_pinn_knee(pinn, Xtr_n, ytr, tr, ne,
                              physics_lambda=dict(PHYSICS_LAMBDA), X_val=Xc_n,
                              y_val=yc if yc.size else None, use_log_target=True)
    pinn.eval()
    xgb = fit_xgb(Xtr_n, ytr)

    rows = []
    for mode in ('clean', 'sensor_bias', 'corr_drift'):
        te_c = te if mode == 'clean' else corrupt(te, mode)
        Xte, yte, _, _ = build_feature_matrix(te_c, ne)
        if Xte.size == 0:
            continue
        Xte_n = scaler.transform(Xte)      # the scaler fitted on clean training data
        with torch.no_grad():
            p_pinn = pinn.predict_raw(torch.tensor(Xte_n, dtype=torch.float32)
                                      .to(DEVICE)).cpu().numpy().ravel()
        p_xgb = np.expm1(xgb.predict(Xte_n))
        rows.append({'n_early': ne, 'seed': seed, 'mode': mode,
                     'MAE_PINN': round(evaluate_knee_predictions(yte, p_pinn)['MAE'], 3),
                     'MAE_XGB': round(evaluate_knee_predictions(yte, p_xgb)['MAE'], 3)})
    return rows


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117, f"pool {len(cells)} != 117"
    print(f"Pool {len(cells)} cell. bias=+{BIAS*100:.0f}%, drift=+{DRIFT*100:.0f}%\n", flush=True)
    splits = _kfold_split(cells, 5, seed=42)

    all_rows = []
    t0 = time.time()
    for ne in N_EARLY_LIST:
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                for r in run(ne, seed, tr, cal, te):
                    r['fold'] = fold
                    all_rows.append(r)
                    new = not os.path.exists(OUT)
                    with open(OUT, 'a', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=list(r.keys()))
                        if new:
                            w.writeheader()
                        w.writerow(r)
            print(f"  ne={ne} fold={fold} xong  ({time.time()-t0:.0f}s)", flush=True)

    # ---- DOI CHUNG + tong hop ----
    import collections
    print("\n" + "=" * 66)
    print("  Control: the clean PINN MAE must match the Experiment 1 rerun (~159/140/117)")
    print("=" * 66)
    for ne in ('50', '100', '150'):
        cl = [r['MAE_PINN'] for r in all_rows if str(r['n_early']) == ne and r['mode'] == 'clean']
        print(f"  ne={ne}: clean PINN = {np.mean(cl):.1f}")

    print("\n" + "=" * 66)
    print("  % increase in MAE relative to clean (lower is more robust)")
    print("=" * 66)
    print(f"  {'dieu kien':<14s}{'PINN %':>10s}{'XGBoost %':>12s}")
    for mode in ('sensor_bias', 'corr_drift'):
        pp, xx = [], []
        for ne in ('50', '100', '150'):
            for seed in (0, 1, 2):
                for fold in range(5):
                    cl = [r for r in all_rows if str(r['n_early']) == ne and r['seed'] == seed
                          and r['fold'] == fold and r['mode'] == 'clean']
                    md = [r for r in all_rows if str(r['n_early']) == ne and r['seed'] == seed
                          and r['fold'] == fold and r['mode'] == mode]
                    if cl and md:
                        pp.append((md[0]['MAE_PINN'] - cl[0]['MAE_PINN']) / cl[0]['MAE_PINN'] * 100)
                        xx.append((md[0]['MAE_XGB'] - cl[0]['MAE_XGB']) / cl[0]['MAE_XGB'] * 100)
        print(f"  {mode:<14s}{np.mean(pp):>9.1f}%{np.mean(xx):>11.1f}%")
    print(f"\nXong sau {(time.time()-t0)/60:.1f} phut", flush=True)


if __name__ == '__main__':
    main()
