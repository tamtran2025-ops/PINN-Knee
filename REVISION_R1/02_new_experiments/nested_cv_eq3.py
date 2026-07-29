"""R3-1: leakage-free re-estimation of the target-dependent constants in Eq. (3).

Three quantities were set from the scale of the full knee distribution: the offset
delta, delta_max and N_max. alpha, beta and gamma are unchanged, because the paper
derives them from the functional form in Appendix A rather than from the labels.

Leakage-free rule: inside each training fold the three quantities are re-estimated
from the training labels only, and the held-out fold is never touched.

Self-consistency check, run before anything else: if the rule is fed the whole
dataset it must return the published values, and the script asserts this.

Thresholds recorded in advance: a nested-versus-leaky difference beyond +10 cycles
would mean the leak had inflated the published result, in which case the nested
numbers must be the ones reported. Control: the leaky branch must reproduce the
Table 1 value of about 139.6 cycles at n_early = 100.
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
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'nested_cv_eq3.csv')
N_EARLY, SEEDS, N_FOLDS = 100, [0, 1, 2], 5

EQ3_CONST = 6.3000        # alpha*log(b0)+beta*log(d0)+gamma*c0 tai tanh=0
NEUTRAL_RATIO = 0.845     # neutral / median over the whole dataset
DMAX_RATIO = 1.8          # delta_max / median
NMAX_RATIO = 1.49         # N_max / max


def leakfree_params(y_train):
    """The three quantities, computed from the training-fold labels only."""
    med, mx = float(np.median(y_train)), float(np.max(y_train))
    return (float(np.log(NEUTRAL_RATIO * med) - EQ3_CONST),   # delta
            DMAX_RATIO * med,                                  # delta_max
            NMAX_RATIO * mx)                                   # N_max


def run(variant, fold, seed, tr, cal, te):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, _, _ = build_feature_matrix(tr, N_EARLY)
    Xc, yc, _, _ = build_feature_matrix(cal, N_EARLY)
    Xte, yte, _, _ = build_feature_matrix(te, N_EARLY)
    if Xtr.size == 0 or Xte.size == 0:
        return None
    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)

    np.random.seed(seed); torch.manual_seed(seed)
    m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)

    if variant == 'nested':
        d, dmax, nmax = leakfree_params(ytr)
        m.eq3_delta = d
        m._delta_scale = dmax
        m.max_cycle = nmax
    # variant == 'leaky' -> keep (-0.4, 800, 2500) in the paper

    m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, N_EARLY,
                           physics_lambda=dict(PHYSICS_LAMBDA), X_val=Xc_n,
                           y_val=yc if yc.size else None, use_log_target=True)
    m.eval()
    with torch.no_grad():
        pred = m.predict_raw(torch.tensor(Xte_n, dtype=torch.float32)
                             .to(DEVICE)).cpu().numpy().ravel()
    met = evaluate_knee_predictions(yte, pred)
    return {'variant': variant, 'fold': fold, 'seed': seed,
            'n_train': len(ytr), 'n_test': len(yte),
            'eq3_delta': round(m.eq3_delta, 4),
            'delta_max': round(float(m._delta_scale), 1),
            'N_max': round(float(m.max_cycle), 1),
            'MAE': round(met['MAE'], 4)}


def main():
    print(f"Writing results to: {OUT}", flush=True)

    # --- self-consistency check, before running anything else ---
    cells = load_paper_pool()
    assert len(cells) == 117, f"Expected 117 cells, got {len(cells)}"   # step R4
    y_all = np.array([c['knee_cycle'] for c in cells], float)
    d, dmax, nmax = leakfree_params(y_all)
    print(f"\nSelf-consistency check (feeding all labels must return the published values):")
    print(f"  delta     : {d:+.4f}   (paper: -0.4000)   "
          f"{'OK' if abs(d + 0.4) < 0.02 else '*** MISMATCH ***'}")
    print(f"  delta_max : {dmax:.1f}   (paper: 800.0)     "
          f"{'OK' if abs(dmax - 800) < 40 else '*** MISMATCH ***'}")
    print(f"  N_max     : {nmax:.1f}  (paper: 2500.0)    "
          f"{'OK' if abs(nmax - 2500) < 60 else '*** MISMATCH ***'}")
    if abs(d + 0.4) >= 0.02:
        print("\n  SELF-CONSISTENCY CHECK FAILED: the rule is wrong. Stopping.")
        return
    print(f"\nPool {len(cells)} cells, n_early={N_EARLY}, "
          f"{N_FOLDS} folds x {len(SEEDS)} seeds x 2 branches\n", flush=True)

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, encoding='utf-8')):
            done.add((r['variant'], r['fold'], r['seed']))

    splits = _kfold_split(cells, N_FOLDS, seed=42)
    total, i, t0 = 2 * N_FOLDS * len(SEEDS), 0, time.time()
    for variant in ('leaky', 'nested'):
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                i += 1
                if (variant, str(fold), str(seed)) in done:
                    continue
                try:
                    r = run(variant, fold, seed, tr, cal, te)
                except Exception as e:
                    print(f"  [{i}/{total}] {variant} f={fold} s={seed} "
                          f"ERROR: {str(e)[:60]}", flush=True)
                    continue
                if r is None:
                    continue
                new = not os.path.exists(OUT)
                with open(OUT, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(r.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(r)
                eta = (time.time() - t0) / i * (total - i) / 60
                print(f"  [{i:>2d}/{total}] {variant:<6s} f={fold} s={seed}  "
                      f"delta={r['eq3_delta']:+.3f} dmax={r['delta_max']:6.1f} "
                      f"MAE={r['MAE']:7.1f}   ETA {eta:4.1f}p", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
