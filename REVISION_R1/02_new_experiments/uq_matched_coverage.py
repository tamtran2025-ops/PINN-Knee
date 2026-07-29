"""Interval width compared at matched coverage.

The obvious objection to the previous table is that CV+ over-covers, at 95.8 to 97.4
per cent, and its intervals are wider than the Gaussian ones; comparing widths at
different coverage levels means little. The fair comparison inflates the Gaussian
interval until it attains the same empirical coverage as CV+, then compares widths.

Method: fit the Gaussian process once per budget, fold and seed, store the mean and
standard deviation, then sweep the nominal level from 0.90 to 0.9999 to find the level
at which it matches the CV+ coverage.

Control: the Gaussian process MAE must match Table 1 (162.5 / 147.7 / 116.4).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

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

SEEDS = [0, 1, 2]
TABLE1_GP = {50: 162.5, 100: 147.7, 150: 116.4}


def fit_all():
    """Fit the GP once and store mu, sd and y_true for each run."""
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    store = []
    for ne in (50, 100, 150):
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                np.random.seed(seed)
                Xtr, ytr, _, _ = build_feature_matrix(tr, ne)
                Xte, yte, _, _ = build_feature_matrix(te, ne)
                if Xtr.size == 0 or Xte.size == 0:
                    continue
                Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)
                m = create_model('GaussianProcess')
                m.fit(Xtr_n, np.log1p(ytr))
                mu, sd = m.predict(Xte_n, return_std=True)
                store.append(dict(n_early=ne, fold=fold, seed=seed,
                                  mu=mu, sd=sd, y=yte))
    return store


def picp_mpiw(store, ne, nominal):
    z = norm.ppf(0.5 + nominal / 2.0)
    ps, ws = [], []
    for r in store:
        if r['n_early'] != ne:
            continue
        lo = np.expm1(r['mu'] - z * r['sd'])
        hi = np.expm1(r['mu'] + z * r['sd'])
        ps.append(np.mean((r['y'] >= lo) & (r['y'] <= hi)))
        ws.append(np.mean(hi - lo))
    return float(np.mean(ps)), float(np.mean(ws))


def main():
    print("Fitting the GP once, then sweeping the nominal level ...\n")
    store = fit_all()

    print("Control: GP MAE")
    ok = True
    for ne in (50, 100, 150):
        maes = [np.mean(np.abs(r['y'] - np.expm1(r['mu'])))
                for r in store if r['n_early'] == ne]
        got = float(np.mean(maes))
        good = abs(got - TABLE1_GP[ne]) <= 0.15
        ok &= good
        print(f"   n_early={ne:>3}: {got:>6.1f}  vs Table 1 {TABLE1_GP[ne]:>6.1f}  "
              f"{'MATCH' if good else '>>> MISMATCH <<<'}")
    if not ok:
        sys.exit("\nCONTROL FAILED.")

    cv = pd.read_csv(os.path.join(HERE, 'cvplus_conformal.csv'))
    cvg = cv.groupby('n_early').agg(PICP=('PICP', 'mean'), MPIW=('MPIW', 'mean'))

    grid = np.concatenate([np.arange(0.90, 0.999, 0.002), np.arange(0.999, 0.99999, 0.0002)])
    print("\n" + "=" * 80)
    print("MATCHED COVERAGE: what nominal level does the GP need, and how wide is it then?")
    print("=" * 80)
    print(f"{'n_early':>8}{'CV+ PICP':>10}{'CV+ MPIW':>10}"
          f"{'GP level needed':>16}{'GP PICP':>10}{'GP MPIW':>10}{'verdict':>18}")
    print("-" * 80)
    rows = []
    for ne in (50, 100, 150):
        target = float(cvg.loc[ne, 'PICP'])
        cvw = float(cvg.loc[ne, 'MPIW'])
        hit = None
        for nom in grid:
            p, w = picp_mpiw(store, ne, nom)
            if p >= target:
                hit = (nom, p, w)
                break
        if hit is None:
            nom, p, w = grid[-1], *picp_mpiw(store, ne, grid[-1])[0:2]
            verd = 'GP cannot reach it'
        else:
            nom, p, w = hit
            verd = 'CV+ narrower' if cvw < w else 'GP narrower'
        rows.append(dict(n_early=ne, cv_picp=target, cv_mpiw=cvw,
                         gp_nominal=nom, gp_picp=p, gp_mpiw=w, verdict=verd))
        print(f"{ne:>8}{target:>10.3f}{cvw:>10.0f}{nom:>16.3%}{p:>10.3f}{w:>10.0f}{verd:>18}")

    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'uq_matched_coverage.csv'), index=False)
    print("\nWrote: uq_matched_coverage.csv")


if __name__ == '__main__':
    main()
