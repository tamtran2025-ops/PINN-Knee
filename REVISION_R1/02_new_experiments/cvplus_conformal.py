"""CV+ conformal prediction (Barber et al., 2021, Annals of Statistics, Section 3).

Two problems with the split-conformal procedure used in the original submission:
  R3-5  the same calibration subset was used both for early stopping and for the
        nonconformity scores, so the predictor is not independent of the calibration
        data and the exchangeability argument breaks.
  A coverage of about 1.00 is not evidence of good calibration; it is evidence that
  the intervals are too wide.

CV+ instead trains K models, each leaving out one inner fold, and forms the interval
from out-of-fold residuals. No data is wasted and no point is used both to select the
model and to calibrate it.

Per-cell records are kept so that stratified coverage can be computed afterwards:
correct marginal coverage does not imply correct conditional coverage, since a
constant-width interval can over-cover short-lived cells and under-cover long-lived
ones while the average still looks right.

Output is written under REVISION_R1 and never overwrites the submitted results.
"""
import os, sys, csv, time, warnings
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE
from features import build_feature_matrix, normalize_features
from models import create_model
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from repeated_cv_architecture import kfold_split_all

OUT_CSV = os.path.join(HERE, 'cvplus_conformal.csv')
N_EARLY_LIST = [50, 100, 150]
N_OUTER, K_INNER = 5, 10
ALPHA = 0.05
SEEDS = [0]


def cvplus_interval(mu_test_per_k, resid, inner_of, alpha=ALPHA):
    """CV+ interval following Barber et al. (2021), Section 3.

    mu_test_per_k : (K, n_test)  prediction of the k-th model on the test set
    resid         : (n_train,)   out-of-fold residual for each training point
    inner_of      : (n_train,)   which inner fold point i belongs to
    """
    n_tr = len(resid)
    n_te = mu_test_per_k.shape[1]
    lo = np.empty(n_te)
    hi = np.empty(n_te)
    # for each test point, form n_train values mu_{-k(i)}(x) -/+ R_i
    for j in range(n_te):
        mu_i = mu_test_per_k[inner_of, j]          # (n_train,)
        low_vals = mu_i - resid
        high_vals = mu_i + resid
        # conservative quantile from the order statistic, without interpolation
        k_lo = int(np.floor(alpha * (n_tr + 1)))
        k_hi = int(np.ceil((1 - alpha) * (n_tr + 1)))
        k_lo = max(k_lo, 1)
        k_hi = min(k_hi, n_tr)
        lo[j] = np.sort(low_vals)[k_lo - 1]
        hi[j] = np.sort(high_vals)[k_hi - 1]
    return lo, hi


def run_outer(ne, seed, tr_cells, te_cells):
    np.random.seed(seed); torch.manual_seed(seed)
    Xtr, ytr, _, _ = build_feature_matrix(tr_cells, ne)
    Xte, yte, _, _ = build_feature_matrix(te_cells, ne)
    if Xtr.size == 0 or Xte.size == 0:
        return None
    Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)
    n_tr = len(ytr)

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_tr)
    inner_folds = np.array_split(order, K_INNER)
    inner_of = np.empty(n_tr, dtype=int)
    for k, idx in enumerate(inner_folds):
        inner_of[idx] = k

    resid = np.empty(n_tr)
    mu_test_per_k = np.empty((K_INNER, len(yte)))

    # one training run per inner fold, serving both purposes:
    #   (a) out-of-fold residual on that inner fold
    #   (b) predicting on the test set, which the CV+ formula needs
    for k, idx in enumerate(inner_folds):
        mask = np.ones(n_tr, dtype=bool); mask[idx] = False
        m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
        m, _ = train_pinn_knee(m, Xtr_n[mask], ytr[mask], None, None,
                               use_log_target=True)
        m.eval()
        with torch.no_grad():
            Xo = torch.tensor(Xtr_n[idx], dtype=torch.float32).to(DEVICE)
            Xe = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
            resid[idx] = np.abs(ytr[idx] -
                                np.atleast_1d(m.predict_raw(Xo).cpu().numpy().ravel()))
            mu_test_per_k[k] = np.atleast_1d(m.predict_raw(Xe).cpu().numpy().ravel())

    lo, hi = cvplus_interval(mu_test_per_k, resid, inner_of)
    point = mu_test_per_k.mean(axis=0)

    cover = float(np.mean((yte >= lo) & (yte <= hi)))
    width = float(np.mean(hi - lo))
    return {'n_early': ne, 'seed': seed, 'n_train': n_tr, 'n_test': len(yte),
            'K_inner': K_INNER, 'alpha': ALPHA,
            'PICP': round(cover, 4), 'MPIW': round(width, 1),
            'MAE': round(float(np.mean(np.abs(point - yte))), 2),
            # Keep per-cell detail so stratified coverage can be computed.
            # Correct marginal coverage does not imply correct conditional coverage:
            # an interval of the same width for a cell lasting 200 cycles and one lasting 1600
            # can over-cover the short-lived group and under-cover the long-lived one.
            'y_true': '|'.join(f'{v:.0f}' for v in yte),
            'y_pred': '|'.join(f'{v:.2f}' for v in point),
            'lower': '|'.join(f'{v:.2f}' for v in lo),
            'upper': '|'.join(f'{v:.2f}' for v in hi)}


def main():
    cells = load_paper_pool()
    print(f"Pool: {len(cells)} cells   K_inner={K_INNER}   alpha={ALPHA}")
    print(f"CV+ guarantee: coverage >= {1-2*ALPHA:.2f}  (Barber et al. 2021)\n")

    done = set()
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV, encoding='utf-8')):
            done.add((r['n_early'], r['seed'], r['fold']))

    splits = kfold_split_all(cells, N_OUTER, seed=42)
    total = len(N_EARLY_LIST) * N_OUTER * len(SEEDS)
    i, t0 = 0, time.time()
    for ne in N_EARLY_LIST:
        for fold, (tr, cal, te) in enumerate(splits):
            # CV+ needs no separate calibration split, so calibration is folded into training
            tr_all = tr + cal
            for seed in SEEDS:
                i += 1
                if (str(ne), str(seed), str(fold)) in done:
                    continue
                r = run_outer(ne, seed, tr_all, te)
                if r is None:
                    continue
                r['fold'] = fold
                new = not os.path.exists(OUT_CSV)
                with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(r.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(r)
                eta = (time.time() - t0) / i * (total - i) / 60
                print(f"  [{i}/{total}] ne={ne} f={fold}  PICP={r['PICP']:.3f}  "
                      f"MPIW={r['MPIW']:7.1f}  MAE={r['MAE']:6.1f}   ETA {eta:5.1f}p")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")


if __name__ == '__main__':
    main()
