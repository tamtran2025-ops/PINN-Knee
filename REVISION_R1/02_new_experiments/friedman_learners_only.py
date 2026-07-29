"""Does the rank test survive once the non-learning baselines are removed?

Six sequence baselines (LSTM, GRU, Bayesian LSTM, Transformer, Informer,
PatchTST) emit a single constant for every test cell within every run: the
per-run standard deviation of their predictions is exactly 0.00 cycles and the
correlation with the label is |r| < 0.01. They are also worse than the optimal
constant predictor by 58 to 109 cycles. They did not fit the data at n = 117
cells.

That matters for Section 5.1.2, because the Friedman omnibus test is computed
over all 14 models. Six mutually near-identical non-learners sitting at the
bottom of every fold ranking inflate the between-model rank spread, so a large
chi-square there is partly an artefact of including them. The honest test is
whether the omnibus still rejects, and whether PINN-Knee still takes the top
mean rank, among the models that actually learned.

Method: fold-level MAE, seeds averaged within fold, exactly as Section 5.1.2
does (n = 5 folds). Neural models come from rerun_exp1_fixed.csv; the classical
three come from classical_log_full.csv, which is the log-target protocol Table 1
reports for them.

Control, run first: the 14-model statistic must reproduce the chi-square values
already printed in the manuscript. If it does not, the pipeline is wrong and
nothing below it may be used.
"""
import collections
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
NEURAL = os.path.join(HERE, 'rerun_exp1_fixed.csv')
CLASSIC = os.path.join(HERE, 'classical_log_full.csv')

# chi-square values printed in Section 5.1.2 of the manuscript
PAPER_CHI2 = {50: 53.0, 100: 57.4, 150: 59.7}
CLASSICAL_MODELS = {'XGBoost', 'RandomForest', 'GaussianProcess'}
# the six baselines whose predictions are constant within every run
NON_LEARNERS = ['LSTM', 'GRU', 'Bayesian_LSTM', 'Transformer', 'Informer', 'PatchTST']


def fold_mae():
    """{(n_early, model): {fold: MAE}} with the 3 seeds averaged inside each fold.

    rerun_exp1_fixed.csv also carries the three classical models, but in NON-LOG
    form. Table 1 and Section 5.1.2 take them from classical_log_full.csv instead,
    so the non-log copies must be skipped or the two protocols get averaged
    together. Section 5.1.2's own script does the same.
    """
    acc = collections.defaultdict(list)
    for path in (NEURAL, CLASSIC):
        with open(path, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if path == NEURAL and r['model'] in CLASSICAL_MODELS:
                    continue
                acc[(int(r['n_early']), r['model'], int(r['fold']))].append(float(r['MAE']))
    out = collections.defaultdict(dict)
    for (ne, mdl, fold), v in acc.items():
        out[(ne, mdl)][fold] = float(np.mean(v))
    return out


def friedman(table, models, folds, ne):
    """Returns (chi2, p, {model: mean rank}). Rank 1 = best MAE within a fold."""
    ranks = np.zeros((len(folds), len(models)))
    for i, f in enumerate(folds):
        ranks[i] = stats.rankdata([table[(ne, m)][f] for m in models])
    rbar = ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[[table[(ne, m)][f] for f in folds] for m in models])
    return float(chi2), float(p), dict(zip(models, rbar))


def nemenyi_cd(k, n, q_alpha):
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


# critical values of the Studentised range / sqrt(2) at alpha = 0.05
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}


def main():
    table = fold_mae()
    ne_list = (50, 100, 150)
    allm = sorted({m for (ne, m) in table if ne == 100})
    folds = sorted(table[(100, 'PINN_Knee')])
    print(f'{len(allm)} models, {len(folds)} folds (the 3 seeds are averaged inside each fold)\n')

    # ---------------- control ----------------
    print('Control: reproduce the 14-model chi-square printed in Section 5.1.2')
    bad = 0
    for ne in ne_list:
        got, gp, _ = friedman(table, allm, folds, ne)
        ref = PAPER_CHI2[ne]
        ok = abs(got - ref) < 0.6
        bad += not ok
        print(f'  n_early={ne:<4} computed chi2 {got:6.2f}   manuscript prints {ref:5.1f}   '
              f'{"MATCH" if ok else ">>> MISMATCH <<<"}   (p = {gp:.2e})')
    if bad:
        sys.exit('CONTROL FAILED. Do not use the results below.')

    learners = [m for m in allm if m not in NON_LEARNERS]
    print(f'\nDropping the {len(NON_LEARNERS)} baselines that did not fit, {len(learners)} remain:')
    print('  ' + ', '.join(learners) + '\n')

    for ne in ne_list:
        print('=' * 78)
        print(f'  n_early = {ne}')
        print('=' * 78)
        for tag, mods in (('14 models (as published)', allm), (f'{len(learners)} models that fitted', learners)):
            chi2, p, rbar = friedman(table, mods, folds, ne)
            k, n = len(mods), len(folds)
            cd = nemenyi_cd(k, n, Q05[k])
            order = sorted(rbar, key=rbar.get)
            pos = order.index('PINN_Knee') + 1
            print(f'\n  {tag}:  chi2 = {chi2:.2f}, p = {p:.2e}, '
                  f'{"REJECTS" if p < 0.05 else "does NOT reject"} the null')
            print(f'    mean ranks: ' + ', '.join(f'{m} {rbar[m]:.2f}' for m in order[:4]))
            print(f'    PINN-Knee places {pos}/{k} (rank {rbar["PINN_Knee"]:.2f}), Nemenyi CD = {cd:.2f}')
            beat = [m for m in mods
                    if m != 'PINN_Knee' and rbar[m] - rbar['PINN_Knee'] > cd]
            print(f'    significantly better (Nemenyi) than {len(beat)}/{k - 1} models'
                  + (f': {", ".join(beat)}' if beat else ': none'))
        print()



if __name__ == '__main__':
    main()
