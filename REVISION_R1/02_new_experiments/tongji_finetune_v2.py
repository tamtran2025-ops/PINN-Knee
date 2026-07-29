"""Cross-chemistry fine-tuning on the Tongji NCM cells, second attempt.

In the first attempt the correction head was so far out of distribution that its
output overwhelmed the physics term and softplus pushed every prediction to about 1,
making a learning rate of 1e-4 over 100 epochs a no-op: 57 of 57 runs were unchanged.

This version uses a larger learning rate and 300 epochs, chosen from train-side
convergence on the fitted cell rather than from test performance. The physics
parameters and the correction magnitude are recorded before and after as mechanistic
evidence, alongside a constant-prediction baseline that uses only the one available label.
"""
import os, sys, csv, time, copy, pickle, warnings
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, RESULTS_DIR, MAX_CYCLE_LIFE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import PINN_Knee
from train import train_pinn_knee

OUT = os.path.join(HERE, 'tongji_finetune_v2.csv')
NE, REF_ZS, TOL = 150, 316.79, 20.0
BASE_SEEDS, FT_SEEDS = list(range(5)), [0, 1, 2]
FT_EPOCHS, FT_LR, FT_WD = 300, 1e-2, 1e-5


def load_pools():
    with open(os.path.join(RESULTS_DIR, '_severson_cache.pkl'), 'rb') as f:
        sev = pickle.load(f)
    sev = [c for c in sev if c.get('has_knee_point') and c.get('knee_cycle') is not None]
    with open(os.path.join(RESULTS_DIR, '_tongji_cache.pkl'), 'rb') as f:
        tj = pickle.load(f)
    tj = [c for c in tj if c.get('has_knee_point') and c.get('knee_cycle') is not None
          and c.get('knee_cycle', 0) > NE]
    return sev, tj


def predict(model, X_n):
    model.eval()
    with torch.no_grad():
        return model.predict_raw(torch.tensor(X_n, dtype=torch.float32)
                                 .to(DEVICE)).cpu().numpy().ravel()


def parts(model, X_n):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_n, dtype=torch.float32).to(DEVICE)
        kp, _ = model._physics_forward(Xt)
        d = model._nn_delta(Xt)
    return float(kp.mean()), float(d.mean())


def finetune(base, X_fit, y_fit, mode):
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.requires_grad = True
    if mode == 'freeze_physics':
        for p in m.physics_head.parameters():
            p.requires_grad = False
    elif mode == 'freeze_nn':
        for p in m.nn_head.parameters():
            p.requires_grad = False
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=FT_LR, weight_decay=FT_WD)
    X = torch.tensor(X_fit, dtype=torch.float32).to(DEVICE)
    y_log = torch.tensor(np.log1p(y_fit), dtype=torch.float32).to(DEVICE)
    m.train()
    for _ in range(FT_EPOCHS):
        opt.zero_grad()
        loss = m.compute_loss(X, y_log, knee_max=MAX_CYCLE_LIFE,
                              physics_lambda=PHYSICS_LAMBDA)
        loss.backward()
        opt.step()
    m.eval()
    return m


def main():
    print(f"Writing to: {OUT}", flush=True)
    sev, tj = load_pools()
    assert len(sev) == 113
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(sev))
    sev_tr = [sev[i] for i in idx[22:]]
    X_tr, y_tr, _, _ = build_feature_matrix(sev_tr, NE)
    X_te, y_te, _, tev = build_feature_matrix(tj, NE)
    tj = [tj[i] for i in tev]
    assert len(tj) == 19
    _, X_te_n, _, _ = normalize_features(X_tr, X_te, X_te)
    y_te = np.asarray(y_te, float)
    nf = X_tr.shape[1]

    print("[1/3] Train 5-seed base...", flush=True)
    t0 = time.time()
    bases = []
    for seed in BASE_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = PINN_Knee(n_features=nf, hidden=128, layers=3, dropout=0.1).to(DEVICE)
        m, _ = train_pinn_knee(m, X_tr, y_tr, sev_tr, NE, use_log_target=True,
                               verbose=False)
        bases.append(m)
    ens = np.mean([predict(m, X_te_n) for m in bases], axis=0)
    ens_mae = float(np.mean(np.abs(ens - y_te)))
    print(f"[2/3] Control: zero-shot ensemble = {ens_mae:.2f} (goc {REF_ZS} {TOL})",
          flush=True)
    if abs(ens_mae - REF_ZS) > TOL:
        print("*** MISMATCH: STOPPING ***", flush=True)
        return
    print("  OK", flush=True)

    print("[3/3] LOO 19 fold x 3 seed, lr=1e-2 x 300ep...", flush=True)
    rows = []
    n = len(tj)
    for fit_i in range(n):
        te_i = [i for i in range(n) if i != fit_i]
        X_fit, y_fit = X_te_n[fit_i:fit_i+1], y_te[fit_i:fit_i+1]
        X_h, y_h = X_te_n[te_i], y_te[te_i]
        const_mae = float(np.mean(np.abs(y_h - y_fit[0])))
        for seed in FT_SEEDS:
            base = bases[seed]
            zs = float(np.mean(np.abs(predict(base, X_h) - y_h)))
            kp0, d0 = parts(base, X_fit)
            row = {'fold': fit_i, 'fit_cell': tj[fit_i]['name'], 'seed': seed,
                   'knee_fit': int(y_fit[0]), 'n_test': len(te_i),
                   'zeroshot': round(zs, 2), 'const_1cell': round(const_mae, 2),
                   'phys_before': round(kp0, 1), 'delta_before': round(d0, 1)}
            for mode in ('freeze_physics', 'freeze_nn', 'full'):
                np.random.seed(seed); torch.manual_seed(seed)
                mf = finetune(base, X_fit, y_fit, mode)
                row[mode] = round(float(np.mean(np.abs(predict(mf, X_h) - y_h))), 2)
                if mode == 'full':
                    kp1, d1 = parts(mf, X_fit)
                    row['phys_after_full'] = round(kp1, 1)
                    row['delta_after_full'] = round(d1, 1)
            np.random.seed(seed); torch.manual_seed(seed)
            ms = PINN_Knee(n_features=nf, hidden=128, layers=3, dropout=0.1).to(DEVICE)
            ms = finetune(ms, X_fit, y_fit, 'full')
            row['from_scratch'] = round(float(np.mean(np.abs(predict(ms, X_h) - y_h))), 2)
            rows.append(row)
        if fit_i % 4 == 0:
            print(f"  fold {fit_i}/{n} ({(time.time()-t0)/60:.1f}p)", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 66)
    print("  TONGJI v2 (lr=1e-2 x 300ep; mean over 19 folds x 3 seeds)")
    print("=" * 66)
    for k in ('zeroshot', 'const_1cell', 'from_scratch',
              'freeze_physics', 'freeze_nn', 'full'):
        v = np.array([r[k] for r in rows], float)
        print(f"  {k:16s} MAE = {v.mean():7.1f}  {v.std():5.1f}")
    db = np.array([r['delta_before'] for r in rows], float)
    da = np.array([r['delta_after_full'] for r in rows], float)
    pb = np.array([r['phys_before'] for r in rows], float)
    pa = np.array([r['phys_after_full'] for r in rows], float)
    print(f"\n  CO CHE (cell fit): delta {db.mean():+.0f} -> {da.mean():+.0f} ; "
          f"physics {pb.mean():.0f} -> {pa.mean():.0f}")
    print(f"  Done {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
