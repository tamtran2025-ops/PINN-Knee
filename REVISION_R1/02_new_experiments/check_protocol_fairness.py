"""Diagnostic, not reported in the paper: is the Table 1 protocol unfair to PINN-Knee?

The neural models train on the training split and additionally use the calibration
split (20%) for early stopping, so they benefit from 80% of the non-test data while
the classical baselines see only 60%.

Question: if the classical baselines are refitted on train + calibration, which is
closer to an equal data allocation, does the Gaussian Process improve or degrade?
The answer decides whether the published ranking is generous to PINN-Knee.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments')):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding='utf-8')

from features import build_feature_matrix, normalize_features   # noqa: E402
from models import create_model                                  # noqa: E402
from metrics import evaluate_knee_predictions                    # noqa: E402
from rerun_exp1_fixed import load_paper_pool                     # noqa: E402
from run_experiments import _kfold_split                         # noqa: E402

MODELS = ['GaussianProcess', 'RandomForest', 'XGBoost']
SEEDS = [0, 1, 2]


def run(use_cal):
    cells = load_paper_pool()
    splits = _kfold_split(cells, 5, seed=42)
    out = {}
    for ne in (50, 100, 150):
        for mn in MODELS:
            maes = []
            for tr, cal, te in splits:
                fit_cells = (tr + cal) if use_cal else tr
                for seed in SEEDS:
                    np.random.seed(seed)
                    Xtr, ytr, _, _ = build_feature_matrix(fit_cells, ne)
                    Xte, yte, _, _ = build_feature_matrix(te, ne)
                    if Xtr.size == 0 or Xte.size == 0:
                        continue
                    Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)
                    m = create_model(mn)
                    m.fit(Xtr_n, np.log1p(ytr))
                    pred = np.expm1(m.predict(Xte_n))
                    met=evaluate_knee_predictions(yte, pred)
                    maes.append((met['MAE'],met['MedianAE'],met.get('Within_100',met.get('Within100',np.nan))))
            arr=np.array(maes,float); out[(ne, mn)] = arr.mean(axis=0)
    return out


def main():
    print("Running the classical baselines under both protocols (may take a few minutes)...\n")
    a = run(use_cal=False)   # as in the current Table 1
    b = run(use_cal=True)    # fairer: the classical baselines also get the calibration split
    pinn = {50: (159.2,79,0.57), 100: (139.6,70,0.62), 150: (117.4,59,0.65)}  # Table 1

    print(f"{'n_early':>8}{'model':>18}{'MAE tr':>10}{'MAE tr+cal':>12}{'MedAE tr+cal':>14}{'W100 tr+cal':>13}")
    print("-" * 72)
    for ne in (50, 100, 150):
        for mn in MODELS:
            x, y = a[(ne, mn)], b[(ne, mn)]
            print(f"{ne:>8}{mn:>18}{x[0]:>10.2f}{y[0]:>12.2f}{y[1]:>14.1f}{100*y[2]:>12.0f}%")
        pp=pinn[ne]
        print(f"{'':>8}{'PINN (Table 1)':>18}{pp[0]:>10.2f}{'':>12}{pp[1]:>14.1f}{100*pp[2]:>12.0f}%")
        bm=min(b[(ne,m)][0] for m in MODELS); bmed=min(b[(ne,m)][1] for m in MODELS)
        bw=max(b[(ne,m)][2] for m in MODELS)
        print(f"{'':>8}{'best classical':>18}{'':>10}{bm:>12.2f}{bmed:>14.1f}{100*bw:>12.0f}%")
        print(f"{'':>8}   PINN wins on MAE: {pp[0]<bm}   MedAE: {pp[1]<bmed}   W100: {pp[2]>bw}")
        print()


if __name__ == '__main__':
    main()
