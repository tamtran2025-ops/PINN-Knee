"""R1-7, done correctly: sensitivity of PINN-Knee to the number of hidden layers.

An earlier attempt (network_depth_results.csv) used a plain unbounded SiLU MLP, whose
expm1 output diverged and produced meaningless MAE values between 223 and 2487. That
was not PINN-Knee at all.

Done correctly: ablate the hidden depth of both heads inside the real PINN_Knee model,
keeping everything else fixed. Control: depth 2, the default, must give about 139
cycles at n_early = 100, matching the published model.
"""
import os, sys, csv, warnings
import numpy as np
import torch
import torch.nn as nn
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import create_model, count_parameters
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool
from run_experiments import _kfold_split

OUT = os.path.join(HERE, 'r1_7_depth.csv')
NE, SEEDS = 100, [0, 1, 2]


def build_heads(nf, depth, h=64):
    """Build physics_head (nf->...->6) and nn_head (nf->...->1) with `depth` hidden layers."""
    def stack(out_dim, dropout):
        layers, d_in = [], nf
        widths = [h] + [h // 2] * (depth - 1) if depth > 1 else [h]
        for w in widths:
            layers += [nn.Linear(d_in, w), nn.Tanh()]
            if dropout:
                layers.append(nn.Dropout(0.15))
            d_in = w
        layers.append(nn.Linear(d_in, out_dim))
        return nn.Sequential(*layers)
    return stack(6, False), stack(1, True)


def make_pinn(nf, depth):
    """The real PINN_Knee; for depth != 2 both heads are swapped for that depth."""
    m = create_model('PINN_Knee', n_features=nf, device=DEVICE)
    if depth != 2:
        ph, nh = build_heads(nf, depth)
        m.physics_head = ph.to(DEVICE)
        m.nn_head = nh.to(DEVICE)
        with torch.no_grad():          # keep the final nn_head initialised near zero, as in the original
            m.nn_head[-1].weight.mul_(0.01)
            m.nn_head[-1].bias.zero_()
    return m


def main():
    print(f"Writing to: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    splits = _kfold_split(cells, 5, seed=42)

    rows = []
    for depth in (1, 2, 3):
        maes, meds, npar = [], [], 0
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                Xtr, ytr, _, _ = build_feature_matrix(tr, NE)
                Xc, yc, _, _ = build_feature_matrix(cal, NE)
                Xte, yte, _, _ = build_feature_matrix(te, NE)
                if Xtr.size == 0 or Xte.size == 0:
                    continue
                Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
                np.random.seed(seed); torch.manual_seed(seed)
                m = make_pinn(Xtr_n.shape[1], depth)
                npar = count_parameters(m)
                m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, NE,
                                       physics_lambda=dict(PHYSICS_LAMBDA), X_val=Xc_n,
                                       y_val=yc if yc.size else None, use_log_target=True)
                m.eval()
                with torch.no_grad():
                    pred = m.predict_raw(torch.tensor(Xte_n, dtype=torch.float32)
                                         .to(DEVICE)).cpu().numpy().ravel()
                maes.append(np.mean(np.abs(pred - yte)))
                meds.append(np.median(np.abs(pred - yte)))
        rows.append({'depth': depth, 'n_params': npar,
                     'MAE': round(np.mean(maes), 2), 'MAE_std': round(np.std(maes), 2),
                     'MedianAE': round(np.mean(meds), 2)})
        print(f"  depth={depth}: MAE={np.mean(maes):.1f} params={npar}", flush=True)

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 56)
    print("  R1-7 DEPTH ABLATION (PINN-Knee that)")
    print("=" * 56)
    print(f"  {'depth':>6s}{'n_params':>10s}{'MAE':>9s}{'MedianAE':>10s}")
    for r in rows:
        mark = "  <- default" if r['depth'] == 2 else ""
        print(f"  {r['depth']:>6d}{r['n_params']:>10d}{r['MAE']:>9.1f}{r['MedianAE']:>10.1f}{mark}")
    d2 = [r['MAE'] for r in rows if r['depth'] == 2][0]
    print(f"\n  Control: depth=2 MAE={d2:.1f} (PINN_Knee that ~139)")
    print(f"  {'OK' if abs(d2-139) < 12 else '*** MISMATCH  -  rebuild is wrong ***'}", flush=True)


if __name__ == '__main__':
    main()
