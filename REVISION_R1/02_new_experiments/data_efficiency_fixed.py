"""Data-efficiency curve recomputed after the H1 broadcast bug was fixed.

With the bug present, the plain neural network was trained on a silently broadcast
loss and its reported 241.3 cycles at n = 18 is inflated.

Control: XGBoost is unaffected by the bug and is deterministic given the seed, so its
curve must stay close to the original. Where the corrected curve differs from the
submitted one, only the internal comparison is meaningful and the text says so.
"""
import os, sys, csv, time, warnings
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from rerun_exp1_fixed import load_paper_pool, run_one

OUT = os.path.join(HERE, 'data_efficiency_fixed.csv')
FRACS = [0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
MODELS = ['PINN_Knee', 'XGBoost', 'Pure_NN']
NE = 100


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    # split 60/20/20 seed 42 nhu ban goc (data_loader.get_train_cal_test_split)
    from data_loader import get_train_cal_test_split
    tr_full, cal, te = get_train_cal_test_split(cells, train_frac=0.6,
                                                cal_frac=0.2, seed=42)
    print(f"train={len(tr_full)} cal={len(cal)} test={len(te)}", flush=True)

    rows, t0 = [], time.time()
    for frac in FRACS:
        n_keep = max(5, int(round(frac * len(tr_full))))
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            keep = rng.permutation(len(tr_full))[:n_keep]
            sub = [tr_full[i] for i in keep]
            for mn in MODELS:
                r = run_one(mn, NE, seed, sub, cal, te)
                if r is None:
                    continue
                rows.append({'model': mn, 'frac': frac, 'n_train_used': n_keep,
                             'seed': seed, 'MAE': r['MAE']})
                print(f"  frac={frac:.2f} n={n_keep:>2d} s={seed} {mn:<10s} "
                      f"MAE={r['MAE']:7.1f}  [{(time.time()-t0)/60:.1f}p]", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 60)
    print(f"  {'model':<10s}" + "".join(f"{f'n={max(5, int(round(fr*len(tr_full)))):d}':>10s}" for fr in FRACS))
    ctrl = {}
    for mn in MODELS:
        line = f"  {mn:<10s}"
        for fr in FRACS:
            v = [r['MAE'] for r in rows if r['model'] == mn and r['frac'] == fr]
            line += f"{np.mean(v):>10.1f}"
            if mn == 'XGBoost':
                ctrl[fr] = np.mean(v)
        print(line)
    print("\n  DOI CHUNG XGBoost goc: 198.8 / 239.0 / 181.2 / 177.2")
    print(f"  XGBoost moi          : " + " / ".join(f"{ctrl[f]:.1f}" for f in FRACS), flush=True)


if __name__ == '__main__':
    main()
