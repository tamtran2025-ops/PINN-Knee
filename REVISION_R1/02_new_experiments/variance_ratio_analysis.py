"""Ordinary paired t-test or Nadeau-Bengio for repeated cross-validation?

Thresholds come from a simulation of 400 runs per scenario at 20 repetitions.

Whatever the outcome, the recommendation for the manuscript is unchanged: report both
tests and describe the advantage as consistent but modest. The reason is Reviewer 2,
comment 7, on overlapping confidence intervals.
"""
import os, sys, csv, collections
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repeated_cv_analysis import nadeau_bengio

CSV_PATH = os.path.join(HERE, 'variance_ratio_20rep.csv')
K = 5

# Thresholds from a simulation of 400 runs per scenario at n_rep = 20:
#   unbiased : median 0.97, 10-90% range = [0.65, 1.34]
#   moderate bias   : median 0.59, 10-90% range = [0.38, 0.85]
HI, LO = 0.85, 0.65


def main():
    if not os.path.exists(CSV_PATH):
        print("No data yet."); return
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    d = np.array([float(r['diff']) for r in rows])
    reps = sorted({r['split_seed'] for r in rows}, key=int)
    rep_means = np.array([np.mean([float(r['diff']) for r in rows
                                   if r['split_seed'] == rp]) for rp in reps])
    n_te = np.mean([float(r['n_test']) for r in rows])
    n_tr = 117 - n_te

    var_fold = d.var(ddof=1)
    var_rep = rep_means.var(ddof=1)
    exp_rep = var_fold / K
    ratio = var_rep / exp_rep

    print("=" * 72)
    print(f"  {len(reps)} repetitions x {K} folds = {len(d)} observed")
    print("=" * 72)
    print(f"  mean difference (mlp - arch): {d.mean():+.2f} cycles")
    print(f"  folds with the expected sign               : {int((d>0).sum())}/{len(d)}")
    print(f"\n  variance between folds                 : {var_fold:8.2f}")
    print(f"  variance between repetitions (observed)   : {var_rep:8.2f}")
    print(f"  variance between repetitions (expected)    : {exp_rep:8.2f}")
    print(f"  RATIO                             : {ratio:8.3f}")

    print(f"\n  Thresholds from simulation (n_rep=20):")
    print(f"    ratio > {HI}  -> folds nearly INDEPENDENT  -> the ordinary t-test is valid")
    print(f"    ratio < {LO}  -> a persistent bias is present, so Nadeau-Bengio applies")

    if ratio > HI:
        verdict = "FOLDS NEARLY INDEPENDENT: use the ordinary paired t-test"
    elif ratio < LO:
        verdict = "PERSISTENT BIAS PRESENT: use Nadeau-Bengio"
    else:
        verdict = "AMBIGUOUS REGION: report both, choose neither"
    print(f"\n  >>> {verdict}")

    print("\n" + "=" * 72)
    print("  BOTH TESTS ON THE SAME DATA")
    print("=" * 72)
    t, p_naive = stats.ttest_1samp(d, 0)
    se = d.std(ddof=1) / np.sqrt(len(d))
    tc = stats.t.ppf(0.975, len(d) - 1)
    print(f"  ordinary paired t-test : {d.mean():+.2f} "
          f"CI [{d.mean()-tc*se:+.2f}, {d.mean()+tc*se:+.2f}]  p = {p_naive:.4f}")
    m, lo, hi, _, p_nb = nadeau_bengio(d, n_tr, n_te)
    print(f"  Nadeau-Bengio          : {m:+.2f} "
          f"CI [{lo:+.2f}, {hi:+.2f}]  p = {p_nb:.4f}")
    try:
        _, p_w = stats.wilcoxon(d)
        print(f"  Wilcoxon (distribution-free): p = {p_w:.4f}")
    except Exception:
        pass

    print("""
  Note: whatever the outcome, the recommendation for the manuscript is unchanged, namely
  to report both tests and describe the advantage as consistent but modest. Reason: R2-7
  objected to claiming superiority from overlapping confidence intervals.
""")


if __name__ == '__main__':
    main()
