"""Attribution ablation: does the interval advantage come from the conformal procedure or
from the architecture?

Comparing PINN-Knee with CV+ against a Gaussian process with Bayesian intervals
confounds the two. This script wraps CV+ around the Gaussian process itself, using the
identical procedure from cvplus_conformal.py, so the two effects separate.

Control: the point MAE of GP + CV+ must land near the Table 1 values of
162.5 / 147.7 / 116.4.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments')):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding='utf-8')

from features import build_feature_matrix, normalize_features   # noqa: E402
from models import create_model                                  # noqa: E402
from rerun_exp1_fixed import load_paper_pool                     # noqa: E402
from run_experiments import _kfold_split                         # noqa: E402
from cvplus_conformal import cvplus_interval, K_INNER, ALPHA     # noqa: E402

SEEDS = [0, 1, 2]


def run_outer_gp(ne, seed, tr_cells, te_cells):
    np.random.seed(seed)
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

    for k, idx in enumerate(inner_folds):
        mask = np.ones(n_tr, dtype=bool)
        mask[idx] = False
        m = create_model('GaussianProcess')
        m.fit(Xtr_n[mask], np.log1p(ytr[mask]))          # same log target
        resid[idx] = np.abs(ytr[idx] - np.expm1(m.predict(Xtr_n[idx])))
        mu_test_per_k[k] = np.expm1(m.predict(Xte_n))

    lo, hi = cvplus_interval(mu_test_per_k, resid, inner_of)
    point = mu_test_per_k.mean(axis=0)
    return dict(n_early=ne, seed=seed,
                PICP=float(np.mean((yte >= lo) & (yte <= hi))),
                MPIW=float(np.mean(hi - lo)),
                MAE=float(np.mean(np.abs(point - yte))))


def main():
    print(f"GP + CV+ (K_INNER={K_INNER}, alpha={ALPHA}) ...\n")
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    rows = []
    for ne in (50, 100, 150):
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                r = run_outer_gp(ne, seed, tr + cal, te)
                if r:
                    r['fold'] = fold
                    rows.append(r)
        print(f"  n_early={ne} done", flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(HERE, 'uq_gp_cvplus.csv'), index=False)

    agg = g.groupby('n_early').agg(PICP=('PICP', 'mean'), MPIW=('MPIW', 'mean'),
                                   MAE=('MAE', 'mean'))
    print("\nControl MAE (reference, Table 1: 162.5 / 147.7 / 116.4;")
    print("  note: GP+CV+ fits on tr+cal, so its MAE may be lower; this is expected)")
    for ne in (50, 100, 150):
        print(f"   n_early={ne:>3}: MAE={agg.loc[ne, 'MAE']:.1f}")

    cv = pd.read_csv(os.path.join(HERE, 'cvplus_conformal.csv'))
    cvg = cv.groupby('n_early').agg(PICP=('PICP', 'mean'), MPIW=('MPIW', 'mean'))
    mc = pd.read_csv(os.path.join(HERE, 'uq_matched_coverage.csv')).set_index('n_early')

    print("\n" + "=" * 84)
    print("FOUR CELLS: coverage and interval width")
    print("=" * 84)
    print(f"{'n_early':>8}{'method':>26}{'PICP':>9}{'MPIW':>9}{'vs PINN+CV+':>20}")
    print("-" * 84)
    for ne in (50, 100, 150):
        base = float(cvg.loc[ne, 'MPIW'])
        print(f"{ne:>8}{'PINN + CV+':>26}{cvg.loc[ne, 'PICP']:>9.3f}{base:>9.0f}{'(reference)':>20}")
        r = agg.loc[ne]
        print(f"{ne:>8}{'GP + CV+':>26}{r.PICP:>9.3f}{r.MPIW:>9.0f}"
              f"{f'{100*(r.MPIW/base-1):+.0f}%':>20}")
        gp_w = float(mc.loc[ne, 'gp_mpiw'])
        print(f"{ne:>8}{'GP + Bayesian (matched cov)':>26}"
              f"{mc.loc[ne, 'gp_picp']:>9.3f}{gp_w:>9.0f}"
              f"{f'{100 * (gp_w / base - 1):+.0f}%':>20}")
        print()


if __name__ == '__main__':
    main()
