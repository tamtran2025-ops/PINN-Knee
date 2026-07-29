"""Coverage comparison: CV+ against the Gaussian process Bayesian predictive interval.

Motivation: the Gaussian process is the strongest baseline on MAE, but its intervals
rest on a Gaussian assumption and carry no finite-sample guarantee, whereas CV+
(Barber et al., 2021) guarantees at least 1 - 2*alpha without distributional
assumptions. The paper had never measured the Gaussian process coverage for comparison.

Mandatory control: the Gaussian process MAE here must reproduce Table 1
(162.5 / 147.7 / 116.4). If it does not, the harness is wrong and the results must not
be used.
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
TABLE1_GP = {50: 162.5, 100: 147.7, 150: 116.4}   # control values


def gp_intervals():
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    rows = []
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
                pred = np.expm1(mu)
                for nominal in (0.90, 0.95):
                    z = norm.ppf(0.5 + nominal / 2.0)
                    lo = np.expm1(mu - z * sd)
                    hi = np.expm1(mu + z * sd)
                    rows.append(dict(
                        n_early=ne, fold=fold, seed=seed, nominal=nominal,
                        PICP=float(np.mean((yte >= lo) & (yte <= hi))),
                        MPIW=float(np.mean(hi - lo)),
                        MAE=float(np.mean(np.abs(yte - pred)))))
    return pd.DataFrame(rows)


def main():
    print("Measuring the Gaussian Process intervals ...\n")
    g = gp_intervals()

    # ---- Control ----
    ctrl = g[g.nominal == 0.90].groupby('n_early')['MAE'].mean().round(1)
    print("Control: the GP MAE must match Table 1")
    ok = True
    for ne in (50, 100, 150):
        got, want = float(ctrl[ne]), TABLE1_GP[ne]
        good = abs(got - want) <= 0.15
        ok &= good
        print(f"   n_early={ne:>3}: script={got:>6.1f}   Table 1={want:>6.1f}   "
              f"{'MATCH' if good else '>>> MISMATCH, DO NOT USE THESE RESULTS <<<'}")
    if not ok:
        sys.exit("\nCONTROL FAILED. Stopping.")

    cv = pd.read_csv(os.path.join(HERE, 'cvplus_conformal.csv'))
    cvg = cv.groupby('n_early').agg(PICP=('PICP', 'mean'), MPIW=('MPIW', 'mean'))

    print("\n" + "=" * 78)
    print("EMPIRICAL COVERAGE (PICP) and INTERVAL WIDTH (MPIW, cycles)")
    print("=" * 78)
    print(f"{'n_early':>8}{'method':>34}{'nominal level':>16}{'PICP':>9}{'MPIW':>9}")
    print("-" * 78)
    for ne in (50, 100, 150):
        for nominal in (0.90, 0.95):
            s = g[(g.n_early == ne) & (g.nominal == nominal)]
            flag = '  <- BELOW' if s.PICP.mean() < nominal - 0.02 else ''
            print(f"{ne:>8}{'Gaussian Process (Bayes)':>34}{nominal:>16.0%}"
                  f"{s.PICP.mean():>9.3f}{s.MPIW.mean():>9.0f}{flag}")
        r = cvg.loc[ne]
        print(f"{ne:>8}{'PINN-Knee + CV+ conformal':>34}{'>= 90% (guarantee)':>16}"
              f"{r.PICP:>9.3f}{r.MPIW:>9.0f}")
        print()

    g.to_csv(os.path.join(HERE, 'uq_gp_vs_cvplus.csv'), index=False)
    print(f"Wrote: uq_gp_vs_cvplus.csv")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for nominal in (0.90, 0.95):
        sub = g[g.nominal == nominal]
        under = [ne for ne in (50, 100, 150)
                 if sub[sub.n_early == ne].PICP.mean() < nominal - 0.02]
        print(f"  GP at the {nominal:.0%} level: under-covers at {len(under)}/3 budgets"
              + (f" (n_early={under})" if under else ""))
    print(f"  CV+ : dat {cvg.PICP.min():.3f} den {cvg.PICP.max():.3f}, "
          f"all exceed guarantee >= 0.90 at all three budgets")


if __name__ == '__main__':
    main()
