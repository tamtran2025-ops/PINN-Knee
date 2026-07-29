"""Cross-protocol transfer (Severson batches 1+2 -> batch 3) recomputed.

The pool used in the original submission could not be reproduced: the text reported
"102 cells in batches 1+2, 16 in batch 3", whereas the 117-cell cache contains 78 and
39. This script reruns the experiment on the 117-cell pool so that it is consistent
with every other experiment in the revision.

Internal controls: (i) the test split must contain exactly the 39 batch-3 cells;
(ii) the classical baselines and PINN-Knee, which are not affected by the H1 broadcast
bug, must fall in the plausible range of 100 to 400 cycles MAE.
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

OUT = os.path.join(HERE, 'batch_transfer_fixed.csv')
MODELS = ['XGBoost', 'RandomForest', 'GaussianProcess',
          'Pure_NN', 'Ensemble_NN', 'Neural_ODE', 'PINN_Knee']
NES = [50, 100, 150]
SEEDS = [0, 1, 2]


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    b12 = [c for c in cells if 'batch1' in c['name'] or 'batch2' in c['name']]
    b3 = [c for c in cells if 'batch3' in c['name']]
    assert len(b12) == 78 and len(b3) == 39, (len(b12), len(b3))

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(b12))
    n_cal = int(round(0.2 * len(b12)))
    cal = [b12[i] for i in idx[:n_cal]]
    tr = [b12[i] for i in idx[n_cal:]]
    print(f"train={len(tr)} cal={len(cal)} test(b3)={len(b3)}", flush=True)

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, encoding='utf-8')):
            done.add((r['model'], r['n_early'], r['seed']))
        print(f"Resume: {len(done)} run co san", flush=True)

    t0, i, total = time.time(), 0, len(MODELS) * len(NES) * len(SEEDS)
    for ne in NES:
        for mn in MODELS:
            for seed in SEEDS:
                i += 1
                if (mn, str(ne), str(seed)) in done:
                    continue
                try:
                    r = run_one(mn, ne, seed, tr, cal, b3)
                except Exception as e:
                    print(f"  [{i}/{total}] {mn} ne={ne} s={seed} LOI: {str(e)[:60]}",
                          flush=True)
                    continue
                if r is None:
                    continue
                row = {'model': mn, 'n_early': ne, 'seed': seed,
                       'n_test': r['n_test'], 'MAE': r['MAE'],
                       'MedianAE': r['MedianAE']}
                new = not os.path.exists(OUT)
                with open(OUT, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                eta = (time.time() - t0) / i * (total - i) / 60
                print(f"  [{i:>2d}/{total}] ne={ne:>3d} {mn:<16s} s={seed} "
                      f"MAE={r['MAE']:7.1f}  ETA {eta:5.1f}p", flush=True)

    # tong ket
    rows = list(csv.DictReader(open(OUT, encoding='utf-8')))
    print("\n" + "=" * 64)
    print(f"  {'model':<16s}" + "".join(f"{f'ne={ne}':>12s}" for ne in NES))
    for mn in MODELS:
        line = f"  {mn:<16s}"
        for ne in NES:
            v = [float(r['MAE']) for r in rows
                 if r['model'] == mn and r['n_early'] == str(ne)]
            line += f"{np.mean(v):>9.1f}{np.std(v):<2.0f}" if v else f"{'-':>12s}"
        print(line)
    print(flush=True)


if __name__ == '__main__':
    main()
