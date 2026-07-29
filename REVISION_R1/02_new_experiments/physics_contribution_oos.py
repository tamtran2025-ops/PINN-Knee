"""Out-of-sample version of the physics-contribution analysis of Section 5.5.1.

The original analysis fitted on a subset but reported per-cell figures for all cells,
so roughly 80% of the cells quoted were inside the training set. This version uses
5-fold cross-validation, so every cell is evaluated exactly once out of sample, and
reports the in-sample figure alongside it in the same run; a large by a factor of between the two
is direct evidence of overfitting.

Thresholds recorded in advance: if the physics-only prediction beats a constant
median predictor out of sample, the claim that the physics head is a useful prior
survives. Control: the in-sample MAE must land near 124.3, the value in the submitted
physics_contribution.csv; a discrepancy above 25 cycles means stop.
"""
import os, sys, csv, time, warnings
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import create_model
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'physics_contribution_oos.csv')
N_EARLY, SEEDS, N_FOLDS = 100, [0, 1, 2], 5


def extract(model, Xn):
    """Tra ve (n_phys, delta)  -  both of shape (N,)."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)
        nph, _ = model._physics_forward(Xt)
        dl = model._nn_delta(Xt)
    return nph.cpu().numpy().ravel(), dl.cpu().numpy().ravel()


def main():
    print(f"Writing results to: {OUT}", flush=True)                       # R3
    cells = load_paper_pool()
    assert len(cells) == 117, f"Expected 117 cells, got {len(cells)}"   # R4
    print(f"Pool {len(cells)} cell, n_early={N_EARLY}, "
          f"{N_FOLDS} fold x {len(SEEDS)} seed\n", flush=True)

    splits = _kfold_split(cells, N_FOLDS, seed=42)
    rows, t0 = [], time.time()
    for fold, (tr, cal, te) in enumerate(splits):
        tr_all = tr + cal                       # train on train+cal
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            Xtr, ytr, _, itr = build_feature_matrix(tr_all, N_EARLY)
            Xte, yte, _, ite = build_feature_matrix(te, N_EARLY)
            if Xtr.size == 0 or Xte.size == 0:
                continue
            Xtr_n, Xte_n, _, _ = normalize_features(Xtr, Xte)

            m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
            m, _ = train_pinn_knee(m, Xtr_n, ytr, tr_all, N_EARLY,
                                   physics_lambda=dict(PHYSICS_LAMBDA),
                                   use_log_target=True)

            for tag, Xn, yv, cl, ids in (('in', Xtr_n, ytr, tr_all, itr),
                                         ('out', Xte_n, yte, te, ite)):
                nph, dl = extract(m, Xn)
                pred = nph + dl
                for j in range(len(yv)):
                    rows.append({'fold': fold, 'seed': seed, 'sample': tag,
                                 'cell': cl[ids[j]]['name'],
                                 'true_knee': float(yv[j]),
                                 'n_phys': round(float(nph[j]), 3),
                                 'delta': round(float(dl[j]), 3),
                                 'knee_pred': round(float(pred[j]), 3),
                                 'y_train_median': round(float(np.median(ytr)), 1)})
            print(f"  fold={fold} seed={seed}  n_in={len(ytr)} n_out={len(yte)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------- analysis ----------------
    import collections
    print("\n" + "=" * 72)
    print("  IN-SAMPLE vs OUT-OF-SAMPLE")
    print("=" * 72)
    res = {}
    for tag in ('in', 'out'):
        s = [r for r in rows if r['sample'] == tag]
        y = np.array([r['true_knee'] for r in s])
        nph = np.array([r['n_phys'] for r in s])
        dl = np.array([r['delta'] for r in s])
        pred = np.array([r['knee_pred'] for r in s])
        med = np.array([r['y_train_median'] for r in s])
        rho = np.abs(dl) / (np.abs(pred) + 1e-9)
        from scipy import stats
        r_p, _ = stats.pearsonr(nph, y)
        res[tag] = dict(n=len(s), rho_med=float(np.median(rho)) * 100,
                        r=r_p, mae_nph=float(np.mean(np.abs(nph - y))),
                        mae_pred=float(np.mean(np.abs(pred - y))),
                        mae_const=float(np.mean(np.abs(med - y))))
        d = res[tag]
        print(f"\n  [{tag}-sample]  n = {d['n']}")
        print(f"    median rho (|delta|/|pred|) : {d['rho_med']:5.1f}%")
        print(f"    r(n_phys, knee that)          : {d['r']:+.3f}  (R2={d['r']**2:.3f})")
        print(f"    MAE n_phys                    : {d['mae_nph']:6.1f}")
        print(f"    MAE, full prediction          : {d['mae_pred']:6.1f}")
        print(f"    MAE, constant median(y_train)   : {d['mae_const']:6.1f}")
        v = "better than constant" if d['mae_nph'] < d['mae_const'] else "*** WORSE than constant ***"
        print(f"    -> n_phys {v} ({d['mae_nph']-d['mae_const']:+.1f})")

    print("\n" + "=" * 72)
    print("  CONCLUSION AGAINST THE PRE-RECORDED THRESHOLDS")
    print("=" * 72)
    drho = abs(res['in']['rho_med'] - res['out']['rho_med'])
    print(f"  (1) rho gap in/out = {drho:.1f} points  -> "
          f"{'the ~89% claim survives' if drho < 3 else 'the ~89% claim must be revised'}")
    ok2 = res['out']['mae_nph'] < res['out']['mae_const']
    print(f"  (2) n_phys out-of-sample {'thap' if ok2 else 'CAO'} than the constant -> "
          f"{'the physics-only statement survives' if ok2 else 'the physics-only statement must go'}")
    print(f"  (3) r: in={res['in']['r']:+.3f} vs out={res['out']['r']:+.3f}  "
          f"(gap {abs(res['in']['r']-res['out']['r']):.3f})")
    print(f"\n  Control: in-sample MAE of n_phys = {res['in']['mae_nph']:.1f}"
          f"  (as submitted: 124.3)  "
          f"{'OK' if abs(res['in']['mae_nph']-124.3) < 25 else '*** MISMATCH > 25 ***'}")


if __name__ == '__main__':
    main()
