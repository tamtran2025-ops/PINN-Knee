"""Significance testing for median AE and within-100, which the paper tested only for MAE.

The paper claims the best median AE and the best within-100 fraction at all three
budgets, but Wilcoxon (S1), Friedman (S2) and the BCa intervals of Table 2 all run on
MAE. If those two metrics separate significantly, the claim is supported by a test
rather than by a point estimate.

Method: average the seeds within each fold to give five matched pairs per model and
budget, then a one-sided Wilcoxon signed-rank test (with n = 5 the smallest attainable
p is 1/2^5 = 0.031) and a paired BCa interval.

Important: rerun_exp1_fixed.csv also contains the three classical models in non-log
form, while Table 1 takes them from classical_log_full.csv under the log target.
Mixing the two would compare log against non-log, so the classical rows are excluded
from the neural file. Control: the mean MAE must reproduce Table 1 before any result
is read.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')

DL = ['PINN_UQ', 'Pure_NN', 'Ensemble_NN', 'Neural_ODE', 'LSTM', 'GRU',
      'Bayesian_LSTM', 'Transformer', 'Informer', 'PatchTST']
CLASSIC = ['GaussianProcess', 'RandomForest', 'XGBoost']
TOP = DL + CLASSIC
TABLE1 = {50: 159.2, 100: 139.6, 150: 117.4}
RNG = np.random.default_rng(0)


def load():
    a = pd.read_csv(os.path.join(HERE, 'rerun_exp1_fixed.csv'))
    b = pd.read_csv(os.path.join(HERE, 'classical_log_full.csv'))
    cols = ['model', 'n_early', 'fold', 'seed', 'MAE', 'MedianAE', 'Within_100']
    for d in (a, b):
        miss = [c for c in cols if c not in d.columns]
        if miss:
            sys.exit(f"thieu cot {miss}")
    # Important: rerun_exp1_fixed.csv also holds the three classical models, in NON-LOG form.
    # Table 1 takes the classical models from classical_log_full.csv under the log target.
    # Merging both would mix log with non-log, so the classical rows are dropped here.
    CLASSICAL = {'GaussianProcess', 'RandomForest', 'XGBoost'}
    a = a[~a.model.isin(CLASSICAL)]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    # average the seeds within each fold, giving one value per fold
    return d.groupby(['model', 'n_early', 'fold'], as_index=False).mean(numeric_only=True)


def bca(diff, n_boot=10000):
    """BCa 95% interval for the mean of `diff` (n = 5)."""
    n = len(diff)
    th = diff.mean()
    boots = np.array([RNG.choice(diff, n, replace=True).mean() for _ in range(n_boot)])
    z0 = np.percentile(boots, 50)
    from scipy.stats import norm
    p0 = (boots < th).mean()
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(p0)
    jack = np.array([np.delete(diff, i).mean() for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = num / den if den != 0 else 0.0
    out = []
    for q in (0.025, 0.975):
        zq = norm.ppf(q)
        adj = z0 + (z0 + zq) / (1 - acc * (z0 + zq))
        out.append(np.percentile(boots, 100 * norm.cdf(adj)))
    return out[0], out[1]


def main():
    d = load()

    REF = {'PINN_Knee': (159.2, 139.6, 117.4), 'GaussianProcess': (162.5, 147.7, 116.4),
           'RandomForest': (165.1, 144.6, 124.6), 'PINN_UQ': (168.2, 142.8, 124.8),
           'XGBoost': (194.2, 160.5, 138.4)}
    print("Control against Table 1 (every model in the top cluster):")
    ok = True
    for m, ref in REF.items():
        got = [d[(d.model == m) & (d.n_early == ne)].MAE.mean() for ne in (50, 100, 150)]
        good = all(abs(g - r) <= 0.15 for g, r in zip(got, ref))
        ok &= good
        print(f"   {m:<16} {got[0]:6.1f} {got[1]:6.1f} {got[2]:6.1f}   "
              f"Table 1: {ref[0]:6.1f} {ref[1]:6.1f} {ref[2]:6.1f}   "
              f"{'MATCH' if good else '>>> MISMATCH <<<'}")
    if not ok:
        sys.exit("CONTROL FAILED. Do not use these results.")

    for metric, better, sign in (('MAE', 'lower', +1),
                                 ('MedianAE', 'lower', +1),
                                 ('Within_100', 'higher', -1)):
        print("\n" + "=" * 88)
        print(f"{metric}  (PINN-Knee {better} = better).  "
              f"delta = PINN - baseline, multiplied by {sign:+d} so negative means PINN is better")
        print("=" * 88)
        print(f"{'n_early':>8}{'baseline':>18}{'PINN':>9}{'baseline':>10}"
              f"{'delta':>9}{'BCa 95%':>22}{'Wilcoxon':>11}{'':>4}")
        print("-" * 88)
        for ne in (50, 100, 150):
            p_ = d[(d.model == 'PINN_Knee') & (d.n_early == ne)].sort_values('fold')
            for b in TOP:
                q = d[(d.model == b) & (d.n_early == ne)].sort_values('fold')
                if len(q) != len(p_) or len(p_) < 5:
                    continue
                pv_, qv_ = p_[metric].values, q[metric].values
                diff = sign * (pv_ - qv_)          # am = PINN better
                lo, hi = bca(diff)
                try:
                    w = wilcoxon(pv_, qv_, alternative='less' if sign > 0 else 'greater').pvalue
                except Exception:
                    w = np.nan
                star = ' *' if hi < 0 else ''
                print(f"{ne:>8}{b:>18}{pv_.mean():>9.3f}{qv_.mean():>10.3f}"
                      f"{diff.mean():>+9.3f}   [{lo:>+7.3f}, {hi:>+7.3f}]"
                      f"{w:>11.4f}{star:>4}")
            print()
    print("* = the BCa 95% interval excludes 0 in favour of PINN-Knee")
    print("Note: with n = 5 folds the smallest attainable one-sided Wilcoxon p is 0.031.")

    # ---- export CSV so the Supplementary table is built from data, not retyped ----
    rows = []
    for metric, sign in (('MAE', +1), ('MedianAE', +1), ('Within_100', -1)):
        for ne in (50, 100, 150):
            p_ = d[(d.model == 'PINN_Knee') & (d.n_early == ne)].sort_values('fold')
            for b in TOP:
                q = d[(d.model == b) & (d.n_early == ne)].sort_values('fold')
                if len(q) != len(p_) or len(p_) < 5:
                    continue
                pv_, qv_ = p_[metric].values, q[metric].values
                diff = sign * (pv_ - qv_)
                lo, hi = bca(diff)
                try:
                    w = wilcoxon(pv_, qv_,
                                 alternative='less' if sign > 0 else 'greater').pvalue
                except Exception:
                    w = float('nan')
                rows.append(dict(metric=metric, n_early=ne, baseline=b,
                                 family='deep-learning' if b in DL else 'classical',
                                 pinn=pv_.mean(), base=qv_.mean(), delta=diff.mean(),
                                 ci_lo=lo, ci_hi=hi, wilcoxon_p=w,
                                 favours_pinn=bool(hi < 0)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, 'stats_median_within.csv'), index=False)
    print()
    print(f"Wrote stats_median_within.csv ({len(out)} rows)")
    for metric in ('MAE', 'MedianAE', 'Within_100'):
        for fam in ('deep-learning', 'classical'):
            sub = out[(out.metric == metric) & (out.family == fam)]
            print(f"   {metric:<11} vs {fam:<14}: "
                  f"{int(sub.favours_pinn.sum())}/{len(sub)} comparisons exclude 0")


if __name__ == '__main__':
    main()
