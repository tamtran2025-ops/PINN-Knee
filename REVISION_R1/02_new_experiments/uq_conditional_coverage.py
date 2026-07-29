"""Conditional coverage: do CV+ and the Gaussian process Bayesian interval hold up within
each lifespan group, or only on average?

Correct marginal coverage does not guarantee correct conditional coverage. An interval
of roughly constant width can over-cover short-lived cells and under-cover long-lived
ones while the average still looks good. This is the standard weak point of conformal
methods and the paper did not report it.

Stratification follows Table S9: terciles of the true knee over the whole pool.
CV+ intervals come from cvplus_conformal.csv, which stores the truth and both bounds
per cell. Control: the marginal CV+ coverage must reproduce 0.958 / 0.974 / 0.973
from Section 5.3.

The Gaussian process is deterministic, so a single seed is used; three would triple
the counts without changing anything.
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

SEEDS = [0]   # the GP is deterministic; three seeds would triple n without changing anything
REF_MARGINAL = {50: 0.958, 100: 0.974, 150: 0.973}


def parse(s):
    return np.array([float(v) for v in str(s).split('|') if v not in ('', 'nan')])


def cvplus_points():
    """(n_early, y_true, lower, upper) per cell, read from the existing file."""
    d = pd.read_csv(os.path.join(HERE, 'cvplus_conformal.csv'))
    rows = []
    for _, r in d.iterrows():
        y, lo, hi = parse(r['y_true']), parse(r['lower']), parse(r['upper'])
        if not (len(y) == len(lo) == len(hi)):
            continue
        for a, b, c in zip(y, lo, hi):
            rows.append((int(r['n_early']), a, b, c))
    return pd.DataFrame(rows, columns=['n_early', 'y', 'lo', 'hi'])


def gp_points(nominal=0.95):
    """Gaussian process Bayesian interval on the same outer split (tr+cal, as for CV+)."""
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    z = norm.ppf(0.5 + nominal / 2.0)
    rows = []
    for ne in (50, 100, 150):
        for tr, cal, te in splits:
            fit = tr + cal
            for seed in SEEDS:
                np.random.seed(seed)
                Xtr, ytr, _, _ = build_feature_matrix(fit, ne)
                Xte, yte, _, _ = build_feature_matrix(te, ne)
                if Xtr.size == 0 or Xte.size == 0:
                    continue
                Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)
                m = create_model('GaussianProcess')
                m.fit(Xtr_n, np.log1p(ytr))
                mu, sd = m.predict(Xte_n, return_std=True)
                lo, hi = np.expm1(mu - z * sd), np.expm1(mu + z * sd)
                for a, b, c in zip(yte, lo, hi):
                    rows.append((ne, a, b, c))
    return pd.DataFrame(rows, columns=['n_early', 'y', 'lo', 'hi'])


def strat(df, q33, q67):
    return np.where(df.y < q33, 'short', np.where(df.y < q67, 'medium', 'long'))


def main():
    print("Reading CV+ from file, recomputing the GP intervals ...\n")
    cv = cvplus_points()
    gp = gp_points()

    allq = np.concatenate([cv.y.values, gp.y.values])
    q33, q67 = np.quantile(allq, [1 / 3, 2 / 3])
    print(f"Strata (terciles of the true knee): short < {q33:.0f}, medium < {q67:.0f}, long >= {q67:.0f}")

    print("\nControl: marginal CV+ coverage (Section 5.3: 0.958 / 0.974 / 0.973)")
    ok = True
    for ne in (50, 100, 150):
        s = cv[cv.n_early == ne]
        got = float(((s.y >= s.lo) & (s.y <= s.hi)).mean())
        good = abs(got - REF_MARGINAL[ne]) <= 0.012
        ok &= good
        print(f"   n_early={ne:>3}: {got:.3f}  vs {REF_MARGINAL[ne]:.3f}  "
              f"{'MATCH' if good else '>>> MISMATCH <<<'}")
    if not ok:
        sys.exit("CONTROL FAILED. Do not use these results.")

    for name, df in (('PINN-Knee + CV+', cv), ('Gaussian process, Bayesian 95%', gp)):
        df = df.copy()
        df['st'] = strat(df, q33, q67)
        print("\n" + "=" * 78)
        print(f"{name}: coverage within each lifespan group")
        print("=" * 78)
        print(f"{'n_early':>8}{'group':>10}{'n cell':>9}{'coverage':>11}{'mean width':>13}")
        print("-" * 78)
        for ne in (50, 100, 150):
            s0 = df[df.n_early == ne]
            marg = float(((s0.y >= s0.lo) & (s0.y <= s0.hi)).mean())
            for st in ('short', 'medium', 'long'):
                s = s0[s0.st == st]
                if s.empty:
                    continue
                cov = float(((s.y >= s.lo) & (s.y <= s.hi)).mean())
                flag = '  <- THIEU' if cov < 0.90 else ''
                print(f"{ne:>8}{st:>10}{len(s):>9}{cov:>11.3f}{(s.hi - s.lo).mean():>13.0f}{flag}")
            print(f"{ne:>8}{'(bien)':>10}{len(s0):>9}{marg:>11.3f}"
                  f"{(s0.hi - s0.lo).mean():>13.0f}")
            print()

    # save for the table
    out = []
    for name, df in (('CV+', cv), ('GP_Bayes95', gp)):
        df = df.copy(); df['st'] = strat(df, q33, q67)
        for ne in (50, 100, 150):
            for st in ('short', 'medium', 'long', 'all'):
                s = df[(df.n_early == ne)] if st == 'all' else \
                    df[(df.n_early == ne) & (df.st == st)]
                if s.empty:
                    continue
                out.append(dict(method=name, n_early=ne, stratum=st, n=len(s),
                                coverage=float(((s.y >= s.lo) & (s.y <= s.hi)).mean()),
                                width=float((s.hi - s.lo).mean())))
    pd.DataFrame(out).to_csv(os.path.join(HERE, 'uq_conditional_coverage.csv'), index=False)
    print("Wrote uq_conditional_coverage.csv")


if __name__ == '__main__':
    main()
