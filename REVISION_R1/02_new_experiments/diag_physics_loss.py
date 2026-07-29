"""Diagnostic: are the five physics losses coded wrongly, or can they simply not reduce error?

Two measurements on the trained model (n_early = 100, split seed 42):
  (1) the magnitude of each loss term, to see whether the physics terms are so small
      that they are effectively inactive;
  (2) the variation of the physics-head knee prediction across cells, to see whether
      the head does per-cell work or behaves almost as a constant.
"""
import os
import sys
import warnings
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments')):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter('ignore')

from config import DEVICE, PHYSICS_LAMBDA                       # noqa: E402
from features import build_feature_matrix, normalize_features   # noqa: E402
from models import create_model                                 # noqa: E402
from train import train_pinn_knee                               # noqa: E402
from rerun_exp1_fixed import load_paper_pool                    # noqa: E402
from run_experiments import _kfold_split                        # noqa: E402

NE = 100


def main():
    cells = load_paper_pool()
    tr, cal, te = _kfold_split(cells, 5, seed=42)[0]
    np.random.seed(0); torch.manual_seed(0)
    Xtr, ytr, _, _ = build_feature_matrix(tr, NE)
    Xc, yc, _, _ = build_feature_matrix(cal, NE)
    Xte, yte, _, _ = build_feature_matrix(te, NE)
    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc)

    m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
    m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, NE, physics_lambda=dict(PHYSICS_LAMBDA),
                           X_val=Xc_n, y_val=yc, use_log_target=True)
    m.eval()

    X = torch.tensor(Xtr_n, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.log1p(ytr), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        knee_phys, params = m._physics_forward(X)
        delta = m._nn_delta(X)
        knee_raw = torch.nn.functional.softplus(knee_phys + delta - 1.0) + 1.0
        loss_data = torch.mean(((torch.log1p(knee_raw) - y) / 8.0) ** 2).item()
        loss_phys, _ = m._physics_losses(knee_phys, knee_phys, dict(PHYSICS_LAMBDA)) \
            if False else m._physics_losses(params, knee_phys, dict(PHYSICS_LAMBDA))
        loss_phys = loss_phys.item()
        lam_corr = PHYSICS_LAMBDA.get('correction_penalty', 0.1)
        loss_corr = (lam_corr * torch.mean((delta / m.max_cycle) ** 2)).item()

    print("=" * 72)
    print("(1) MAGNITUDE OF EACH TERM in total = data + 0.05*physics + correction")
    print("=" * 72)
    terms = {'loss_data': loss_data, '0.05 * loss_physics': 0.05 * loss_phys,
             'loss_correction': loss_corr}
    tot = sum(terms.values())
    for k, v in terms.items():
        print(f"   {k:<22} = {v:.6f}   ({100*v/tot:5.1f}% cua total)")
    print(f"\n   -> physics {'LON HON' if 0.05*loss_phys > loss_data else 'NHO HON'} data "
          f"gap {max(0.05*loss_phys, loss_data)/min(0.05*loss_phys, loss_data):.1f} lan")
    print("   => the physics losses are NOT neutralised by a too-small weight.")

    print()
    print("=" * 72)
    print("(2) Does the physics head do per-cell work?")
    print("=" * 72)
    with torch.no_grad():
        kp = knee_phys.cpu().numpy().ravel()
        dl = delta.cpu().numpy().ravel()
        pr = knee_raw.cpu().numpy().ravel()
    for nm, v in (('knee_physics', kp), ('delta_NN', dl), ('du doan cuoi', pr)):
        print(f"   {nm:<14} trung binh {v.mean():8.1f}   do lech {v.std():7.1f}   "
              f"he so bien thien {abs(v.std()/v.mean()):.3f}")
    print(f"\n   nhan that: trung binh {ytr.mean():.1f}, do lech {ytr.std():.1f}")
    print()
    for nm, v in (('a', params['a']), ('b', params['b']), ('c', params['c']),
                  ('d', params['d']), ('s', params['s'])):
        vv = v.detach().cpu().numpy().ravel()
        print(f"   tham so {nm}: trung binh {vv.mean():.5f}  do lech {vv.std():.5f}  "
              f"he so bien thien {abs(vv.std()/vv.mean()):.4f}")


if __name__ == '__main__':
    main()
