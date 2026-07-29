"""Architecture ablation: what does each component of PINN-Knee actually buy?

Variants compared at a matched parameter count:
  full             the model as published (physics head + bounded correction + 5 losses)
  arch_only        same architecture, all five physics losses removed
  physics_only     trained in full, then Delta_NN forced to 0 at inference, so this
                   measures how far Eq. (3) alone gets inside the real model
  mlp_matched      a plain MLP with the same number of trainable parameters
  constant_median  predicts the training median; the floor every variant must beat

Reading the contrasts:
  full vs arch_only        do the five physics losses change accuracy?
  arch_only vs mlp_matched does the Residual Physics architecture itself help?
  physics_only             how far does Eq. (3) get on its own?

Note on the parameter match: h=64 gives 3,713 parameters, roughly half of PINN-Knee,
which would bias the comparison in our favour. We use h=100 and the script prints the
realised parameter counts so the match can be checked.

Usage: python ablation_architecture.py [n_early]
"""
import os, sys, csv, json, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, N_EPOCHS, LEARNING_RATE, PHYSICS_LAMBDA, HIDDEN_SIZE
from features import build_feature_matrix, normalize_features
from models import create_model, count_parameters
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from run_experiments import _kfold_split
from rerun_exp1_fixed import load_paper_pool

OUT_CSV = os.path.join(HERE, 'ablation_architecture.csv')
SEEDS, N_FOLDS = [0, 1, 2], 5
# Run a single budget quickly: python ablation_architecture.py 100
N_EARLY_LIST = ([int(a) for a in sys.argv[1:]] if len(sys.argv) > 1
                else [50, 100, 150])
ZERO_LAMBDA = {k: 0.0 for k in PHYSICS_LAMBDA}


class MatchedMLP(nn.Module):
    """Plain MLP with a parameter count matched to PINN_Knee and no physics head.

    PINN_Knee (n_features=24, h=min(128,64)=64):
        physics_head 24->64->32->6 = 1600 + 2080 +  198 = 3878
        nn_head      24->64->32->1 = 1600 + 2080 +   33 = 3713
        total                                           = 7591

    MatchedMLP(hidden=h): 25h + h^2/2 + h + 1
        h=64  -> 3713 parameters, about HALF, which would bias the comparison towards PINN_Knee
        h=100 -> 7601  (matched)

    We use h=100. The script prints the realised parameter counts so the match can be checked.
    """

    def __init__(self, n_features, hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Dropout(0.15),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_plain(model, Xtr, ytr_log, Xv, yv_log, n_epochs=N_EPOCHS):
    """Cng optimizer/scheduler/early-stopping nh train_pinn_knee."""
    Xt = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(ytr_log, dtype=torch.float32).to(DEVICE)
    has_v = Xv is not None and yv_log is not None and len(yv_log) > 0
    if has_v:
        Xvv = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
        yvv = torch.tensor(yv_log, dtype=torch.float32).to(DEVICE)

    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=80, min_lr=1e-6)
    best, best_state, wait, patience = float('inf'), None, 0, 200

    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xt).reshape(-1), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        vl = loss.item()
        if has_v:
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(model(Xvv).reshape(-1), yvv).item()
            model.train()
        sch.step(vl)
        if vl < best - 1e-4:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_variant(variant, n_early, seed, tr, cal, te):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Xtr, ytr, _, _ = build_feature_matrix(tr, n_early)
    Xc, yc, _, _ = build_feature_matrix(cal, n_early)
    Xte, yte, _, _ = build_feature_matrix(te, n_early)
    if Xtr.size == 0 or Xte.size == 0:
        return None
    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
    nf = Xtr_n.shape[1]
    Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
    npar = 0

    if variant == 'constant_median':
        pred = np.full(len(yte), float(np.median(ytr)))

    elif variant == 'mlp_matched':
        m = MatchedMLP(nf).to(DEVICE)
        npar = count_parameters(m)
        m = train_plain(m, Xtr_n, np.log1p(ytr), Xc_n,
                        np.log1p(yc) if yc.size else None)
        with torch.no_grad():
            pred = np.expm1(m(Xte_t).cpu().numpy().ravel())

    else:
        # arch_only: drop the five physics losses, to measure what those losses are worth.
        # physics_only: train the full published model, then force Delta_NN = 0 at
        #   inference, to measure how far Eq. (3) gets inside the real model.
        #   (Training with lambda = 0 would no longer be the published model.)
        lam = ZERO_LAMBDA if variant == 'arch_only' else dict(PHYSICS_LAMBDA)
        m = create_model('PINN_Knee', n_features=nf, device=DEVICE)
        npar = count_parameters(m)
        m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, n_early,
                               physics_lambda=lam, X_val=Xc_n,
                               y_val=yc if yc.size else None,
                               use_log_target=True)
        if variant == 'physics_only':
            m._delta_scale = 0.0          # p _NN = 0 -> ch cn Eq.(3)
        m.eval()
        with torch.no_grad():
            pred = m.predict_raw(Xte_t).cpu().numpy().ravel()

    met = evaluate_knee_predictions(yte, pred)
    return {'variant': variant, 'n_early': n_early, 'seed': seed,
            'n_train': len(ytr), 'n_test': len(yte), 'n_params': npar,
            **{k: round(v, 4) for k, v in met.items()}}


def main():
    cells = load_paper_pool()
    print(f"Pool: {len(cells)} cell   device={DEVICE}\n")
    splits = _kfold_split(cells, n_folds=N_FOLDS, seed=42)
    variants = ['full', 'arch_only', 'physics_only', 'mlp_matched', 'constant_median']
    total = len(N_EARLY_LIST) * len(variants) * N_FOLDS * len(SEEDS)

    rows, t0, i = [], time.time(), 0
    for ne in N_EARLY_LIST:
        for v in variants:
            for fold, (tr, cal, te) in enumerate(splits):
                for seed in SEEDS:
                    i += 1
                    try:
                        r = run_variant(v, ne, seed, tr, cal, te)
                    except Exception as e:
                        print(f"  [{i}/{total}] {v} ne={ne} f={fold} s={seed} ERROR: {str(e)[:60]}")
                        continue
                    if r is None:
                        continue
                    r['fold'] = fold
                    rows.append(r)
                    new = not os.path.exists(OUT_CSV)
                    with open(OUT_CSV, 'a', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=list(r.keys()))
                        if new:
                            w.writeheader()
                        w.writerow(r)
                    eta = (time.time() - t0) / i * (total - i) / 60
                    print(f"  [{i:>3d}/{total}] ne={ne:>3d} {v:<16s} f={fold} s={seed}  "
                          f"MAE={r['MAE']:7.1f}   ETA {eta:5.1f}p")

    print("\n" + "=" * 76)
    print("  ABLATION KIEN TRUC")
    print("=" * 76)
    print(f"  {'variant':<18s}" + "".join(f"{f'ne={n}':>12s}" for n in N_EARLY_LIST) + f"{'#params':>10s}")
    print("  " + "-" * 72)
    for v in variants:
        line = f"  {v:<18s}"
        npar = 0
        for ne in N_EARLY_LIST:
            vals = [r['MAE'] for r in rows if r['variant'] == v and r['n_early'] == ne]
            npar = max(npar, max([r['n_params'] for r in rows
                                  if r['variant'] == v] or [0]))
            line += f"{np.mean(vals):>8.1f}{np.std(vals):<3.0f}" if vals else f"{'-':>12s}"
        print(line + f"{npar:>10d}")
    print("""
  HOW TO READ:
    full vs arch_only        do the five physics losses change accuracy?
    arch_only vs mlp_matched does the Residual Physics architecture itself help?
    physics_only             how far does Eq. (3) get on its own?
    constant_median          the floor; every variant must clearly beat it to mean anything.
""")
    print(f"Done in {(time.time()-t0)/60:.1f} min -> {OUT_CSV}")


if __name__ == '__main__':
    main()
