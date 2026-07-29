"""R2-3, done correctly: do the physics-head parameters a, b, c, d, s hit their bounds?

An earlier attempt (parameter_bounds_results.csv) inspected alpha, beta, gamma and
delta, which are the fixed coefficients of Eq. (3) rather than the learned parameters,
and reported wrong values for them as well.

Done correctly: forward the trained PINN_Knee model, read a, b, c, d, s per cell
through tanh(z) before the bounding transform, and report the activation rate for each
parameter at each budget. Control: the mean values must match Section 5.5.
"""
import os, sys, csv, warnings
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

OUT = os.path.join(HERE, 'r2_3_bounds.csv')
SEEDS = [0, 1, 2]
ACTIVE_THR = 0.99


def raw_tanh(model, Xn):
    """Read tanh(z_i) for the five parameters from physics_head, before the bounding transform."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)
        out = model.physics_head(Xt).cpu().numpy()   # (N,6) raw
    return np.tanh(out[:, :5])                        # tanh(z0..z4)


def transformed(th):
    """th = tanh(z) (N,5) -> a,b,c,d,s after the bounding transform."""
    a = 0.9 + 0.2 * th[:, 0]
    b = np.exp(-6.0 + 1.5 * th[:, 1])
    c = 0.15 + 0.1 * th[:, 2]
    d = np.exp(-4.5 + 1.5 * th[:, 3])
    s = 2.5 + 1.5 * th[:, 4]
    return a, b, c, d, s


def main():
    print(f"Writing to: {OUT}", flush=True)
    cells = load_paper_pool()
    assert len(cells) == 117
    splits = _kfold_split(cells, 5, seed=42)
    rows = []
    agg = {ne: {p: [] for p in 'abcds'} for ne in (50, 100, 150)}
    vals = {ne: {p: [] for p in 'abcds'} for ne in (50, 100, 150)}

    for ne in (50, 100, 150):
        for fold, (tr, cal, te) in enumerate(splits):
            for seed in SEEDS:
                np.random.seed(seed); torch.manual_seed(seed)
                Xtr, ytr, _, _ = build_feature_matrix(tr, ne)
                Xc, yc, _, _ = build_feature_matrix(cal, ne)
                Xte, yte, _, _ = build_feature_matrix(te, ne)
                if Xtr.size == 0:
                    continue
                Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
                m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
                m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, ne,
                                       physics_lambda=dict(PHYSICS_LAMBDA), X_val=Xc_n,
                                       y_val=yc if yc.size else None, use_log_target=True)
                th = raw_tanh(m, Xte_n)            # tanh on the TEST set
                a, b, c, d, s = transformed(th)
                for j, nm in enumerate('abcds'):
                    agg[ne][nm].append(np.mean(np.abs(th[:, j]) > ACTIVE_THR) * 100)
                for nm, v in zip('abcds', (a, b, c, d, s)):
                    vals[ne][nm].extend(v.tolist())
        print(f"  ne={ne} done", flush=True)

    print("\n" + "=" * 66)
    print(f"  % OF CELLS WITH AN ACTIVE BOUND (|tanh|>{ACTIVE_THR})")
    print("=" * 66)
    print(f"  {'param':<6s}{'ne=50':>10s}{'ne=100':>10s}{'ne=150':>10s}")
    for nm in 'abcds':
        line = f"  {nm:<6s}"
        for ne in (50, 100, 150):
            line += f"{np.mean(agg[ne][nm]):>9.1f}%"
        print(line)
    overall = np.mean([np.mean(agg[ne][nm]) for ne in (50, 100, 150) for nm in 'abcds'])
    print(f"\n  Mean per parameter and budget: {overall:.1f}% active")
    print(f"  => {100-overall:.1f}% of samples have every bound INACTIVE (strictly interior)")

    print("\n" + "=" * 66)
    print("  Control: mean values (Section 5.5: a~1.0, b~0.005, d/b~3.0)")
    print("=" * 66)
    for ne in (50, 100, 150):
        a = np.mean(vals[ne]['a']); b = np.mean(vals[ne]['b'])
        d = np.mean(vals[ne]['d']); s = np.mean(vals[ne]['s'])
        print(f"  ne={ne}: a={a:.3f}  b={b:.5f}  d={d:.5f}  d/b={d/b:.2f}  s={s:.2f}")

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['n_early', 'param', 'pct_active', 'mean_value'])
        for ne in (50, 100, 150):
            for nm in 'abcds':
                w.writerow([ne, nm, round(np.mean(agg[ne][nm]), 2),
                            round(np.mean(vals[ne][nm]), 5)])
    print(f"\nWrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
