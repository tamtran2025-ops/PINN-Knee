"""
R3-3 v2: log vs raw target  -  dung Pure_NN (target transform BEN NGOAI)
=====================================================================
v1 dung PINN_Knee bi hong: PINN_Knee.compute_loss HARDCODE log1p(pred)
(models.py:789), nen tat use_log_target tao mismatch scale -> raw MAE 3020 (artifact).

Pure_NN.forward = MLP thuan, target transform ap BEN NGOAI -> test log vs raw
CONG BANG. Cung split, cung model, chi khac target preprocessing.

  log: train MSE tren log1p(y), predict = expm1(output)
  raw: train MSE tren y tho,     predict = output

Control: the log-target Pure_NN MAE at n_early=100 must be near the Experiment 1 value (~161.5).
"""
import os, sys, csv, warnings, copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from features import build_feature_matrix, normalize_features
from models import create_model
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'r3_3_logvsraw.csv')
NE, SEEDS, PATIENCE, MIN_DELTA = 100, [0, 1, 2], 150, 0.01


def train_mlp(model, Xtr, ytr_t, Xv, yv_t):
    """Train Pure_NN tren target da transform (ytr_t). .squeeze() (H1 fix)."""
    Xt = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(ytr_t, dtype=torch.float32).to(DEVICE)
    has_v = Xv is not None and yv_t is not None and len(yv_t) > 0
    if has_v:
        Xvv = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
        yvv = torch.tensor(yv_t, dtype=torch.float32).to(DEVICE)
    opt = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50, min_lr=1e-6)
    best, best_state, wait = float('inf'), None, 0
    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xt).squeeze(-1), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        vl = loss.item()
        if has_v:
            model.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(model(Xvv).squeeze(-1), yvv).item()
            model.train()
        sch.step(vl)
        if vl < best - MIN_DELTA:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    print(f"Ghi: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    splits = _kfold_split(cells, 5, seed=42)
    all_y = np.array([c['knee_cycle'] for c in cells], float)
    q33, q67 = np.quantile(all_y, [1/3, 2/3])
    print(f"Doi cell: short<{q33:.0f}, med<{q67:.0f}, long>={q67:.0f}\n", flush=True)

    rows = []
    for fold, (tr, cal, te) in enumerate(splits):
        Xtr, ytr, _, _ = build_feature_matrix(tr, NE)
        Xc, yc, _, _ = build_feature_matrix(cal, NE)
        Xte, yte, _, _ = build_feature_matrix(te, NE)
        if Xtr.size == 0 or Xte.size == 0:
            continue
        Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
        Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
        for seed in SEEDS:
            for tag, use_log in (('log', True), ('raw', False)):
                np.random.seed(seed); torch.manual_seed(seed)
                m = create_model('Pure_NN', n_features=Xtr_n.shape[1], device=DEVICE)
                ytr_t = np.log1p(ytr) if use_log else ytr.astype(float)
                yc_t = (np.log1p(yc) if use_log else yc.astype(float)) if yc.size else None
                m = train_mlp(m, Xtr_n, ytr_t, Xc_n, yc_t)
                with torch.no_grad():
                    o = m(Xte_t).squeeze(-1).cpu().numpy()
                pred = np.expm1(o) if use_log else o
                err = np.abs(pred - yte)
                strata = np.where(yte < q33, 'short', np.where(yte < q67, 'medium', 'long'))
                for st in ('all', 'short', 'medium', 'long'):
                    mask = np.ones(len(yte), bool) if st == 'all' else (strata == st)
                    if mask.sum() == 0:
                        continue
                    rows.append({'fold': fold, 'seed': seed, 'target': tag, 'stratum': st,
                                 'MAE': round(float(np.mean(err[mask])), 2),
                                 'RMSE': round(float(np.sqrt(np.mean((pred[mask]-yte[mask])**2))), 2),
                                 'MedAE': round(float(np.median(err[mask])), 2)})
        print(f"  fold {fold} xong", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 56)
    print("  LOG vs RAW target (Pure_NN, trung binh)")
    print("=" * 56)
    print(f"  {'stratum':<9s}{'metric':<7s}{'LOG':>9s}{'RAW':>9s}")
    for st in ('all', 'short', 'medium', 'long'):
        for mt in ('MAE', 'RMSE', 'MedAE'):
            lg = [r[mt] for r in rows if r['target'] == 'log' and r['stratum'] == st]
            rw = [r[mt] for r in rows if r['target'] == 'raw' and r['stratum'] == st]
            if lg and rw:
                print(f"  {st:<9s}{mt:<7s}{np.mean(lg):>9.1f}{np.mean(rw):>9.1f}")
        print()
    lg = [r['MAE'] for r in rows if r['target'] == 'log' and r['stratum'] == 'all']
    print(f"  DOI CHUNG: Pure_NN log MAE={np.mean(lg):.1f} (Exp1 Pure_NN ~161.5)")
    print(f"  {'OK' if abs(np.mean(lg)-161.5) < 20 else '*** LECH ***'}", flush=True)


if __name__ == '__main__':
    main()
