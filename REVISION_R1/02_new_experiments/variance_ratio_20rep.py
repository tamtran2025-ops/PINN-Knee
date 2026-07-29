"""Twenty repetitions of cross-validation to decide whether the folds can be treated as
independent.

The question is whether repeated cross-validation admits an ordinary paired t-test or
requires the conservative Nadeau-Bengio correction.

Reasoning: every repetition covers all 117 cells, so the persistent per-cell component
is identical across repetitions and cancels out of the between-repetition variance,
leaving only noise. The variance ratio therefore diagnoses which regime applies.

Calibrated in advance by simulation, 400 runs per scenario: the two regimes overlap so
heavily that the observed ratio of 1.07 cannot by itself overturn the conclusion, which
is why the paper reports both tests. Multiple seeds do not shift the ratio, since they
enter the numerator and the denominator alike.

The matched MLP has 7,601 parameters against 7,591 for the architecture variant.
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
from repeated_cv_architecture import kfold_split_all
from ablation_architecture import MatchedMLP, train_plain

OUT = os.path.join(HERE, 'variance_ratio_20rep.csv')
N_EARLY, K, MODEL_SEED = 100, 5, 0
SPLIT_SEEDS = list(range(100, 120))          # 20 repetitions, independent of the previous run
ZERO_LAMBDA = {k: 0.0 for k in PHYSICS_LAMBDA}


def one_fold(seed, tr, cal, te):
    """Return the MAE of arch_only and mlp_matched on the same test set."""
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, _, _ = build_feature_matrix(tr, N_EARLY)
    Xc, yc, _, _ = build_feature_matrix(cal, N_EARLY)
    Xte, yte, _, _ = build_feature_matrix(te, N_EARLY)
    if Xtr.size == 0 or Xte.size == 0:
        return None
    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
    nf = Xtr_n.shape[1]
    Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)

    # arch_only: PINN_Knee architecture, five physics losses removed
    np.random.seed(seed); torch.manual_seed(seed)
    m1 = create_model('PINN_Knee', n_features=nf, device=DEVICE)
    m1, _ = train_pinn_knee(m1, Xtr_n, ytr, tr, N_EARLY,
                            physics_lambda=dict(ZERO_LAMBDA), X_val=Xc_n,
                            y_val=yc if yc.size else None, use_log_target=True)
    m1.eval()
    with torch.no_grad():
        p1 = m1.predict_raw(Xte_t).cpu().numpy().ravel()

    # mlp_matched: a plain MLP with a matched parameter count (7601 against 7591)
    np.random.seed(seed); torch.manual_seed(seed)
    m2 = MatchedMLP(nf).to(DEVICE)
    m2 = train_plain(m2, Xtr_n, np.log1p(ytr), Xc_n,
                     np.log1p(yc) if yc.size else None)
    with torch.no_grad():
        p2 = np.expm1(m2(Xte_t).cpu().numpy().ravel())

    return (evaluate_knee_predictions(yte, p1)['MAE'],
            evaluate_knee_predictions(yte, p2)['MAE'], len(yte))


def main():
    cells = load_paper_pool()
    print(f"Pool {len(cells)} cells | {len(SPLIT_SEEDS)} repetitions x {K} folds "
          f"x 1 seed x 2 variants = {len(SPLIT_SEEDS)*K*2} training runs\n")

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, encoding='utf-8')):
            done.add((r['split_seed'], r['fold']))
        print(f"Resume: already have {len(done)} points\n")

    total, i, t0 = len(SPLIT_SEEDS) * K, 0, time.time()
    for rs in SPLIT_SEEDS:
        splits = kfold_split_all(cells, K, seed=rs)
        for fold, (tr, cal, te) in enumerate(splits):
            i += 1
            if (str(rs), str(fold)) in done:
                continue
            try:
                out = one_fold(MODEL_SEED, tr, cal, te)
            except Exception as e:
                print(f"  [{i}/{total}] rs={rs} f={fold} ERROR: {str(e)[:60]}")
                continue
            if out is None:
                continue
            mae_arch, mae_mlp, n_te = out
            row = {'split_seed': rs, 'fold': fold, 'model_seed': MODEL_SEED,
                   'n_test': n_te, 'arch_only': round(mae_arch, 4),
                   'mlp_matched': round(mae_mlp, 4),
                   'diff': round(mae_mlp - mae_arch, 4)}
            new = not os.path.exists(OUT)
            with open(OUT, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
            eta = (time.time() - t0) / i * (total - i) / 60
            print(f"  [{i:>3d}/{total}] rs={rs} f={fold}  arch={mae_arch:6.1f}  "
                  f"mlp={mae_mlp:6.1f}  diff={row['diff']:+7.1f}   ETA {eta:5.1f}p")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {OUT}")
    print("Run variance_ratio_analysis.py for the conclusion.")


if __name__ == '__main__':
    main()
