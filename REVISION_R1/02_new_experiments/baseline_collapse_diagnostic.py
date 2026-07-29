"""Diagnostic: which baselines in Table 1 actually fitted the data?

Table 1 ranks fourteen models by MAE, and Sections 5.1.1 to 5.1.3 draw their
significance claims largely from the six sequence baselines. This script checks
a prior question: did those models learn anything at all?

Three diagnostics per model and budget, all computed from the stored per-cell
predictions in rerun_exp1_fixed.csv:

  spread   the LARGEST standard deviation of the predictions inside any single
           run. Taking the maximum rather than the average is the conservative
           choice: a model cannot be called collapsed on the strength of its
           quietest run. A model that has learned varies its output across
           cells; one that has collapsed emits the same number for every cell.
  |r|      correlation between prediction and label over the pooled test cells.
  floor    MAE of the optimal constant predictor (the median of each test
           split). A model that scores worse than this is worse than not
           looking at the input at all.

The cause is structural rather than a training accident. Section 4.4 feeds every
baseline the same 24-dimensional engineered feature vector, for fairness at the
feature-engineering level. models.py declares the sequence models with
input_size=1 and reshapes a (batch, 24) vector to (batch, 24, 1), so the 24
heterogeneous scalars are presented as a 24-step univariate sequence. That axis
carries no temporal ordering, so recurrent and attention architectures have
nothing to exploit along it, and with roughly 70 training cells per fold they
settle on the unconditional mean.

Control: the pooled MAE per model must reproduce Table 1 under the same
run-level averaging Table 1 uses.
"""
import collections
import csv
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'rerun_exp1_fixed.csv')
OUT = os.path.join(HERE, 'baseline_collapse_diagnostic.csv')

# non-log copies; Table 1 sources these three from classical_log_full.csv
CLASSICAL = {'XGBoost', 'RandomForest', 'GaussianProcess'}
TABLE1 = {('PINN_Knee', 50): 159.2, ('PINN_Knee', 100): 139.6, ('PINN_Knee', 150): 117.4,
          ('Pure_NN', 100): 161.5, ('Neural_ODE', 100): 150.4, ('Ensemble_NN', 100): 156.0}


def main():
    runs = collections.defaultdict(list)
    with open(SRC, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['model'] in CLASSICAL:
                continue
            yt = np.array([float(v) for v in r['y_true'].split('|')])
            yp = np.array([float(v) for v in r['y_pred'].split('|')])
            if len(yt) != len(yp):
                continue
            runs[(r['model'], int(r['n_early']))].append((yt, yp))

    print('Control against Table 1 (run-level averaging)')
    bad = 0
    for (mdl, ne), ref in TABLE1.items():
        got = float(np.mean([np.abs(a - b).mean() for a, b in runs[(mdl, ne)]]))
        ok = abs(got - ref) < 0.15
        bad += not ok
        print(f'  {mdl:<12} ne={ne:<4} {got:7.2f}  vs Table 1 {ref:6.1f}  '
              f'{"MATCH" if ok else ">>> MISMATCH <<<"}')
    if bad:
        sys.exit('CONTROL FAILED. Do not use these results.')

    rows = []
    for ne in (50, 100, 150):
        floor = float(np.mean([np.abs(yt - np.median(yt)).mean()
                               for yt, _ in runs[('PINN_Knee', ne)]]))
        print('\n' + '=' * 82)
        print(f'  n_early = {ne}     optimal constant predictor scores MAE {floor:.1f} cycles')
        print('=' * 82)
        print(f'  {"model":<16}{"MAE":>7}{"vs floor":>10}{"spread":>9}{"|r|":>7}   verdict')
        stat = []
        for (mdl, n), v in runs.items():
            if n != ne:
                continue
            mae = float(np.mean([np.abs(a - b).mean() for a, b in v]))
            spread = float(max(b.std() for _, b in v))
            yt = np.concatenate([a for a, _ in v])
            yp = np.concatenate([b for _, b in v])
            rho = abs(float(np.corrcoef(yt, yp)[0, 1]))
            stat.append((mae, mdl, spread, rho))
        for mae, mdl, spread, rho in sorted(stat):
            collapsed = spread < 1.0 and rho < 0.05
            print(f'  {mdl:<16}{mae:>7.1f}{mae - floor:>+10.1f}{spread:>9.3f}{rho:>7.3f}   '
                  + ('COLLAPSED to a constant' if collapsed else 'learned'))
            rows.append(dict(n_early=ne, model=mdl, MAE=round(mae, 2),
                             constant_floor=round(floor, 2),
                             delta_vs_floor=round(mae - floor, 2),
                             within_run_pred_std=round(spread, 5),
                             abs_corr_with_label=round(rho, 4),
                             collapsed=int(collapsed)))

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_col = len({r['model'] for r in rows if r['collapsed']})
    print(f'\n{n_col} of the 10 deep-learning baselines collapsed '
          f'(the eleventh neural model is PINN-Knee itself). '
          f'Wrote {os.path.basename(OUT)}.')


if __name__ == '__main__':
    main()
