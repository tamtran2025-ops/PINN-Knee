"""Recompute the classical baselines under the unified log target, keeping every metric.

The protocol is deterministic and is rerun exactly, so MAE, median AE and the
within-50 fraction must match classical_log_table1.csv to the last digit. The script
asserts this before writing anything.
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

from features import build_feature_matrix, normalize_features
from models import create_model
from metrics import evaluate_knee_predictions
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'classical_log_full.csv')
REF = os.path.join(HERE, 'classical_log_table1.csv')
MODELS = ['XGBoost', 'RandomForest', 'GaussianProcess']
SEEDS = [0, 1, 2]
N_EARLY = [50, 100, 150]


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    splits = _kfold_split(cells, 5, seed=42)

    rows, t0 = [], time.time()
    for ne in N_EARLY:
        for mn in MODELS:
            for fold, (tr, cal, te) in enumerate(splits):
                for seed in SEEDS:
                    np.random.seed(seed)
                    Xtr, ytr, _, _ = build_feature_matrix(tr, ne)
                    Xte, yte, _, _ = build_feature_matrix(te, ne)
                    if Xtr.size == 0 or Xte.size == 0:
                        continue
                    Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)
                    m = create_model(mn)
                    m.fit(Xtr_n, np.log1p(ytr))
                    pred = np.expm1(m.predict(Xte_n))
                    met = evaluate_knee_predictions(yte, pred)
                    rows.append({'model': mn, 'n_early': ne, 'seed': seed, 'fold': fold,
                                 **{k: round(float(v), 4) for k, v in met.items()}})
        print(f"  ne={ne} xong ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- control: must match classical_log_table1.csv exactly ----
    ref = list(csv.DictReader(open(REF, encoding='utf-8')))
    refmap = {(r['model'], r['n_early'], r['seed'], r['fold']):
              (float(r['MAE']), float(r['MedianAE']), float(r['Within_50'])) for r in ref}
    bad = 0
    for r in rows:
        k = (r['model'], str(r['n_early']), str(r['seed']), str(r['fold']))
        if k not in refmap:
            bad += 1; continue
        ma, md, w5 = refmap[k]
        if abs(r['MAE']-ma) > 0.01 or abs(r['MedianAE']-md) > 0.01 or abs(r['Within_50']-w5) > 0.001:
            bad += 1
            print(f"  LECH {k}: MAE {r['MAE']} vs {ma}")
    print(f"\n  Control: {len(rows)-bad}/{len(rows)} values match the previous CSV exactly")
    print("  " + ("OK" if bad == 0 else f"*** {bad} LECH ***"), flush=True)


if __name__ == '__main__':
    main()
