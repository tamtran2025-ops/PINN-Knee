"""Which test applies to repeated cross-validation: the ordinary paired t-test, or the
Nadeau-Bengio variance-corrected version?

Reports both, with the variance-ratio diagnostic used to decide. An inconclusive
result must not be read as "no difference" and must not be read as "a small
difference" either; it means the design does not resolve the question.
"""
import os, sys, csv, collections
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'repeated_cv_architecture.csv')

VARIANTS = ['full', 'arch_only', 'physics_only', 'mlp_matched', 'constant_median']


def nadeau_bengio(diff, n_train, n_test, alpha=0.05):
    """Two-sided confidence interval and p-value with the variance correction for repeated CV."""
    n = len(diff)
    m = float(np.mean(diff))
    s2 = float(np.var(diff, ddof=1))
    rho = n_test / n_train
    var_corr = (1.0 / n + rho) * s2
    se = np.sqrt(var_corr)
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    t = m / se if se > 0 else np.inf
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return m, m - tcrit * se, m + tcrit * se, t, p


def main():
    if not os.path.exists(CSV_PATH):
        print("No data yet:", CSV_PATH)
        return
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    print(f"Data: {len(rows)} design points "
          f"({len({r['split_seed'] for r in rows})} repetitions x "
          f"{len({r['fold'] for r in rows})} folds x "
          f"{len({r['model_seed'] for r in rows})} seeds)\n")

    # Grouped by (repetition, fold)  -  averaged over model seeds
    byfold = collections.defaultdict(dict)
    for r in rows:
        k = (r['split_seed'], r['fold'])
        for v in VARIANTS:
            byfold[k].setdefault(v, []).append(float(r[v]))
    keys = sorted(byfold)
    agg = {v: np.array([np.mean(byfold[k][v]) for k in keys]) for v in VARIANTS}
    n = len(keys)

    ntr = np.mean([float(r['n_train']) for r in rows])
    nte = np.mean([float(r['n_test']) for r in rows])
    print(f"n = {n} observed at fold level (n_train~{ntr:.0f}, n_test~{nte:.0f})\n")

    print(f"  {'variant':<18s}{'MAE':>10s}{'std':>9s}")
    print("  " + "-" * 37)
    for v in VARIANTS:
        print(f"  {v:<18s}{agg[v].mean():>10.1f}{agg[v].std(ddof=1):>9.1f}")

    print("\n" + "=" * 78)
    print("  PAIRED COMPARISON  (positive = left-hand side is better)")
    print("=" * 78)
    pairs = [
        ('arch_only', 'mlp_matched', 'Does the architecture help?'),
        ('full', 'mlp_matched', 'the published model vs a plain MLP'),
        ('arch_only', 'full', 'Do the five physics losses help?'),
        ('full', 'constant_median', 'vs a trivial floor'),
        ('constant_median', 'physics_only', 'Eq. (3) alone vs the floor'),
    ]
    for a, b, note in pairs:
        d = agg[b] - agg[a]
        m, lo, hi, t, p = nadeau_bengio(d, ntr, nte)
        # ordinary t-test for reference
        se_naive = np.std(d, ddof=1) / np.sqrt(n)
        tc = stats.t.ppf(0.975, n - 1)
        lo_n, hi_n = m - tc * se_naive, m + tc * se_naive
        _, p_naive = stats.ttest_rel(agg[b], agg[a])
        try:
            _, p_w = stats.wilcoxon(d)
        except Exception:
            p_w = float('nan')

        verdict = ("SIGNIFICANT" if lo > 0 else
                   ("OPPOSITE SIGN" if hi < 0 else "INCONCLUSIVE"))
        print(f"\n  {a}  vs  {b}      [{note}]")
        print(f"    mean difference : {m:+.1f} cycles  "
              f"({m/agg[b].mean()*100:+.1f}%)")
        print(f"    folds with the expected sign     : {int(np.sum(d > 0))}/{n}")
        print(f"    Nadeau-Bengio 95% CI  : [{lo:+.1f}, {hi:+.1f}]   p = {p:.4f}"
              f"   -> {verdict}")
        print(f"    (ordinary t-test 95% CI : [{lo_n:+.1f}, {hi_n:+.1f}]   p = {p_naive:.4f}"
              f"   <- optimistic, not valid for repeated CV)")
        print(f"    (Wilcoxon two-sided   : p = {p_w:.4f})")

    print("\n" + "=" * 78)
    print("  HOW TO READ")
    print("=" * 78)
    print("""  Draw conclusions only from the Nadeau-Bengio column. If the interval contains 0, report
  "INCONCLUSIVE" must not be read as "no difference",
  and must not be read as "a small difference" either.""")


if __name__ == '__main__':
    main()
