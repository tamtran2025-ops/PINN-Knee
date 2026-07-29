"""Recompute every statistic quoted in the manuscript from the corrected runs.

Includes the BCa intervals over the 15 measurements (5 folds x 3 seeds) per model used
in Section 5.10. Results are printed and saved as JSON so the redline can be driven
from data rather than retyped.
"""
import os, sys, csv, json, collections, warnings
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
F_NN = os.path.join(HERE, 'rerun_exp1_fixed.csv')
F_CL = os.path.join(HERE, 'classical_log_table1.csv')
F_CLF = os.path.join(HERE, 'classical_log_full.csv')
F_AB = os.path.join(HERE, 'rerun_ablation_fixed.csv')
OUT = os.path.join(HERE, 'manuscript_stats.json')

CLASSICAL = {'XGBoost', 'RandomForest', 'GaussianProcess'}
NES = [50, 100, 150]
RES = {}


def load_rows():
    nn = list(csv.DictReader(open(F_NN, encoding='utf-8')))
    cl = list(csv.DictReader(open(F_CLF if os.path.exists(F_CLF) else F_CL,
                                  encoding='utf-8')))
    rows = []
    for r in nn:
        if r['model'] in CLASSICAL:
            continue                      # classical LOG models come from a separate file
        rows.append(r)
    rows += cl
    return rows, nn


def cell_stats(rows):
    """(model, ne) -> dict metric -> (mean, std) over the 15 measurements (5 folds x 3 seeds)."""
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r['model'], int(float(r['n_early'])))
        for m in ('MAE', 'MedianAE', 'Within_50', 'Within_100'):
            if m in r and r[m] not in ('', None):
                agg[key][m].append(float(r[m]))
    out = {}
    for k, d in agg.items():
        out[k] = {m: (float(np.mean(v)), float(np.std(v)), len(v))
                  for m, v in d.items()}
    return out


def fold_series(rows, metric='MAE'):
    """(model, ne) -> array of 5 fold values, averaged over 3 seeds."""
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r['model'], int(float(r['n_early'])))
        agg[key][int(float(r['fold']))].append(float(r[metric]))
    out = {}
    for k, d in agg.items():
        folds = sorted(d)
        out[k] = np.array([np.mean(d[f]) for f in folds])
    return out


def bca_ci(diffs, n_boot=10000, seed=0):
    res = stats.bootstrap((np.asarray(diffs),), np.mean, n_resamples=n_boot,
                          confidence_level=0.95, method='BCa',
                          random_state=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    rows, nn_all = load_rows()
    models = sorted({r['model'] for r in rows})
    print(f"models ({len(models)}): {models}\n")
    assert len(models) == 14

    # ============ TABLE 1 ============
    cs = cell_stats(rows)
    RES['table1'] = {}
    print("=" * 100)
    print("TABLE 1 (fair, log-target; classical from classical_log)")
    print("=" * 100)
    hdr = f"{'model':<16s}"
    for ne in NES:
        hdr += f" | {'MAEstd':>13s} {'MedAE':>6s} {'W100':>5s}"
    print(hdr)
    for m in models:
        line = f"{m:<16s}"
        for ne in NES:
            st = cs.get((m, ne), {})
            mae = st.get('MAE'); med = st.get('MedianAE'); w1 = st.get('Within_100')
            line += (f" | {mae[0]:6.1f}{mae[1]:<5.1f} {med[0]:6.0f} "
                     + (f"{w1[0]*100:4.0f}%" if w1 else "   ??"))
            RES['table1'][f"{m}|{ne}"] = {
                'MAE': round(mae[0], 1), 'MAE_std': round(mae[1], 1),
                'MedAE': round(med[0], 1),
                'W100': round(w1[0], 4) if w1 else None,
                'W50': round(st['Within_50'][0], 4) if 'Within_50' in st else None,
                'n': mae[2]}
        print(line)

    # ============ 5.1.1 WILCOXON (one-sided, 5 folds) ============
    fs = fold_series(rows)
    print("\n" + "=" * 100)
    print("5.1.1 WILCOXON one-sided (PINN better), n=5 fold seed-averaged  (p; * <=0.05)")
    print("=" * 100)
    RES['wilcoxon'] = {}
    for ne in NES:
        pinn = fs[('PINN_Knee', ne)]
        line = f"ne={ne:>3d}: "
        for m in models:
            if m == 'PINN_Knee':
                continue
            base = fs[(m, ne)]
            d = base - pinn                       # >0 => PINN better
            try:
                _, p = stats.wilcoxon(d, alternative='greater')
            except ValueError:
                p = 1.0
            RES['wilcoxon'][f"{m}|{ne}"] = round(float(p), 4)
            line += f"{m}:{p:.3f}{'*' if p <= 0.05 else ' '}  "
        print(line)

    # ============ 5.1.2 FRIEDMAN + mean ranks + NEMENYI ============
    print("\n" + "=" * 100)
    print("5.1.2 FRIEDMAN (rank per fold, 14 model) + NEMENYI vs PINN")
    print("=" * 100)
    RES['friedman'] = {}
    k = len(models)
    for ne in NES:
        mat = np.array([fs[(m, ne)] for m in models])   # (14, 5)
        nf = mat.shape[1]
        ranks = np.array([stats.rankdata(mat[:, f]) for f in range(nf)]).T  # (14,5)
        mean_rank = ranks.mean(axis=1)
        chi2, p = stats.friedmanchisquare(*[mat[i] for i in range(k)])
        order = np.argsort(mean_rank)
        top4 = [(models[i], round(mean_rank[i], 2)) for i in order[:4]]
        print(f"ne={ne:>3d}: chi2F={chi2:.1f}  p={p:.2e}  top4 ranks: {top4}")
        # Nemenyi: q = (Ri-Rj)/sqrt(k(k+1)/(12 nf)); p from the studentized range (q*sqrt2)
        se = np.sqrt(k * (k + 1) / (12.0 * nf))
        ipinn = models.index('PINN_Knee')
        nem = {}
        for j, m in enumerate(models):
            if m == 'PINN_Knee':
                continue
            qstat = abs(mean_rank[ipinn] - mean_rank[j]) / se
            pn = float(stats.studentized_range.sf(qstat * np.sqrt(2), k, np.inf))
            nem[m] = round(pn, 4)
        sig = {m: v for m, v in nem.items() if v <= 0.05}
        print(f"        Nemenyi p<=0.05 vs PINN: {sig}")
        RES['friedman'][str(ne)] = {
            'chi2': round(float(chi2), 1), 'p': float(p),
            'mean_rank': {m: round(float(mean_rank[i]), 2) for i, m in enumerate(models)},
            'nemenyi_vs_pinn': nem}

    # ============ 5.1.3 TABLE 2: BCa over the 5 fold differences ============
    print("\n" + "=" * 100)
    print("TABLE 2: BCa 95% CI on Delta_MAE = MAE(PINN) - MAE(baseline), 5 fold")
    print("=" * 100)
    RES['table2'] = {}
    for m in models:
        if m == 'PINN_Knee':
            continue
        line = f"{m:<16s}"
        for ne in NES:
            d = fs[('PINN_Knee', ne)] - fs[(m, ne)]   # am => PINN better
            lo, hi = bca_ci(d)
            sig = '*' if hi < 0 else ('t' if lo > 0 else 'ns')
            RES['table2'][f"{m}|{ne}"] = {'d': round(float(d.mean()), 1),
                                          'lo': round(lo, 1), 'hi': round(hi, 1),
                                          'sig': sig}
            line += f" | {d.mean():+7.1f} [{lo:+7.1f},{hi:+7.1f}] {sig:>2s}"
        print(line)

    # ============ 5.10: BCa over the 15 measurements per model ============
    print("\n" + "=" * 100)
    print("5.10 BCa CI of the mean MAE (15 measurements)  -  top model + Pure_NN")
    print("=" * 100)
    RES['bca_mean'] = {}
    per15 = collections.defaultdict(list)
    for r in rows:
        per15[(r['model'], int(float(r['n_early'])))].append(float(r['MAE']))
    for m in models:
        line = f"{m:<16s}"
        for ne in NES:
            v = per15[(m, ne)]
            lo, hi = bca_ci(v, seed=1)
            RES['bca_mean'][f"{m}|{ne}"] = {'mean': round(float(np.mean(v)), 1),
                                            'lo': round(lo, 1), 'hi': round(hi, 1)}
            line += f" | {np.mean(v):6.1f} [{lo:6.1f},{hi:6.1f}]"
        print(line)

    # ============ TABLE 3: per-loss ablation ============
    print("\n" + "=" * 100)
    print("TABLE 3 (per-loss ablation ne=100, full budget): Delta vs full + BCa")
    print("=" * 100)
    ab = list(csv.DictReader(open(F_AB, encoding='utf-8')))
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ab:
        per[r['config']][(int(r['fold']), int(r['seed']))].append(float(r['MAE']))
    full = per['full']
    keys = sorted(full)
    RES['table3'] = {}
    for cfg in ('full', 'no_ic', 'no_ode', 'no_knee_transition', 'no_sei',
                'no_monotonic', 'no_physics'):
        if cfg not in per:
            continue
        vals = np.array([np.mean(per[cfg][k]) for k in keys])
        fv = np.array([np.mean(full[k]) for k in keys])
        mae = vals.mean()
        if cfg == 'full':
            print(f"{cfg:<20s} MAE={mae:7.2f}")
            RES['table3'][cfg] = {'MAE': round(float(mae), 2)}
            continue
        d = vals - fv                              # >0 => removing the loss makes it worse
        # BCa over the 15 differences (fold x seed)
        lo, hi = bca_ci(d, seed=2)
        sig = 'sig' if lo > 0 or hi < 0 else 'ns'
        print(f"{cfg:<20s} MAE={mae:7.2f}  dMAE={d.mean():+6.2f} [{lo:+6.2f},{hi:+6.2f}] {sig}")
        RES['table3'][cfg] = {'MAE': round(float(mae), 2), 'd': round(float(d.mean()), 2),
                              'lo': round(lo, 2), 'hi': round(hi, 2), 'sig': sig}

    # ============ classical RAW, reference for the supplementary ============
    print("\n" + "=" * 100)
    print("Classical baselines on the RAW target (from rerun_exp1_fixed), for reference only")
    print("=" * 100)
    RES['classical_raw'] = {}
    craw = collections.defaultdict(list)
    for r in nn_all:
        if r['model'] in CLASSICAL:
            craw[(r['model'], int(float(r['n_early'])))].append(float(r['MAE']))
    for m in sorted(CLASSICAL):
        line = f"{m:<16s}"
        for ne in NES:
            v = craw[(m, ne)]
            line += f" | {np.mean(v):6.1f}"
            RES['classical_raw'][f"{m}|{ne}"] = round(float(np.mean(v)), 1)
        print(line)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(RES, f, indent=1)
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
