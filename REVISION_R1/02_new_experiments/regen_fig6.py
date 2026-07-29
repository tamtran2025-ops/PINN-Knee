"""Regenerate Figure 6, the learned physics parameter distributions.

Controls fixed before plotting: a near 1.0, b near 0.005, d near 0.015, s near 3.5.
The older physics_params_per_cell_agg.csv reports d/b = 2.845 and belongs to a
different run; it is not used.
"""
import os, sys, csv, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT = os.path.join(HERE, 'figures_new', 'image6.png')
CSV = os.path.join(HERE, 'fig6_physics_params.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
NE = 100

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "figure.dpi": 200, "font.family": "sans-serif",
})


def transform(th):
    """th = tanh(z) (N,5) -> a,b,c,d,s after the bounding transform (theo models.py)."""
    a = 0.9 + 0.2 * th[:, 0]
    b = np.exp(-6.0 + 1.5 * th[:, 1])
    c = 0.15 + 0.10 * th[:, 2]
    d = np.exp(-4.5 + 1.5 * th[:, 3])
    s = 2.5 + 1.5 * th[:, 4]
    return a, b, c, d, s


def main():
    cells = load_paper_pool()
    print(f"pool = {len(cells)} cell", flush=True)
    splits = _kfold_split(cells, 5, seed=42)
    tr, cal, te = splits[0]

    Xtr, ytr, _, _ = build_feature_matrix(tr, NE)
    Xc, yc, _, _ = build_feature_matrix(cal, NE)
    Xall, yall, _, _ = build_feature_matrix(cells, NE)      # the WHOLE pool
    Xtr_n, Xall_n, Xc_n, _ = normalize_features(Xtr, Xall, Xc)
    print(f"forward on {len(yall)} cell (ne={NE})", flush=True)

    np.random.seed(0); torch.manual_seed(0)
    m = create_model('PINN_Knee', n_features=Xtr_n.shape[1], device=DEVICE)
    m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, NE, physics_lambda=dict(PHYSICS_LAMBDA),
                           X_val=Xc_n, y_val=yc, use_log_target=True)
    m.eval()
    Xt = torch.tensor(Xall_n, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        raw = m.physics_head(Xt).cpu().numpy()
        th = np.tanh(raw[:, :5])
        nphys, _ = m._physics_forward(Xt)
        delta = m._nn_delta(Xt)
    a, b, c, d, s = transform(th)
    nphys = nphys.cpu().numpy().ravel(); delta = delta.cpu().numpy().ravel()
    y = np.asarray(yall, float)

    # ---------- Control ----------
    dob = d / b
    print("\n=== Control: must match Section 5.5 ===")
    print(f"  a = {a.mean():.3f}   (paper ~1.0)")
    print(f"  b = {b.mean():.5f}  (paper ~0.005)")
    print(f"  d = {d.mean():.5f}  (paper ~0.015)")
    print(f"  s = {s.mean():.2f}     (paper ~3.5)")
    print(f"  d/b median = {np.median(dob):.3f}  (paper 3.00)")
    print(f"  %cell d>b  = {100*np.mean(d > b):.1f}%  (paper 100%)")
    ok = (abs(np.median(dob) - 3.0) < 0.15 and np.mean(d > b) > 0.99
          and abs(a.mean() - 1.0) < 0.1)
    print(f"  {'OK, proceeding to plot' if ok else '*** MISMATCH: stopping, nothing plotted ***'}", flush=True)
    if not ok:
        return

    with open(CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['a', 'b', 'c', 'd', 's', 'd_over_b', 'n_phys', 'delta', 'true_knee'])
        for i in range(len(y)):
            w.writerow([a[i], b[i], c[i], d[i], s[i], dob[i], nphys[i], delta[i], y[i]])

    # ---------- VE ----------
    fig, axes = plt.subplots(2, 3, figsize=(12.83, 7.60))
    fig.suptitle(r"PINN-Knee Physics Parameter Validation (Severson, $n_{\mathrm{early}}=100$)",
                 fontsize=13, y=0.985)
    P = [('(a)', 'Learned $b$: SEI-type $\\sqrt{t}$ rate', '$b$ (SEI rate)'),
         ('(b)', 'Learned $d$: post-knee rate', '$d$ (post-knee decay)'),
         ('(c)', 'ODE constraint: post > pre', '$d/b$'),
         ('(d)', 'Learned $a$: initial condition', '$a$ (initial capacity)'),
         ('(e)', 'Learned $s$: knee transition', '$s$ (knee sharpness)'),
         ('(f)', 'Physics vs truth (color = $\\Delta_{\\mathrm{NN}}$)', 'True knee (cycles)')]

    def hist(ax, v, color, lines=()):
        ax.hist(v, bins=28, color=color, edgecolor='black', linewidth=0.6)
        for xv, cl, lb in lines:
            ax.axvline(xv, ls='--', c=cl, lw=1.8, label=lb)
        if lines:
            ax.legend(fontsize=8)
        ax.set_ylabel("Count")
        ax.grid(axis='y', ls=':', alpha=0.4); ax.set_axisbelow(True)

    ax = axes[0, 0]; hist(ax, b, '#1f3b57',
                          [(0.004, 'red', '$b^{(1)}_{target}=0.004$'),
                           (0.006, 'orange', '$b^{(2)}_{target}=0.006$')])
    ax = axes[0, 1]; hist(ax, d, '#4CAF7D')
    ax = axes[0, 2]; hist(ax, dob, '#C4756B', [(3.0, 'black', 'target $d/b=3$')])
    ax = axes[1, 0]; hist(ax, a, '#8C7BC4', [(1.0, 'black', 'target $a=1.0$')])
    ax = axes[1, 1]; hist(ax, s, '#C9B370', [(3.5, 'black', 'target $s=3.5$')])
    ax = axes[1, 2]
    sc = ax.scatter(y, nphys, c=delta, cmap='RdBu_r', vmin=-400, vmax=400,
                    s=34, edgecolors='black', linewidths=0.4)
    lim = [0, max(y.max(), nphys.max()) * 1.05]
    ax.plot(lim, lim, ls='--', c='gray', lw=1.2)
    ax.set_ylabel(r"$n_{\mathrm{phys}}$ (physics-only prediction)")
    ax.grid(ls=':', alpha=0.4); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$\Delta_{\mathrm{NN}}$ (cycles)")

    for k, (axx, (lab, ttl, xlab)) in enumerate(zip(axes.ravel(), P)):
        axx.set_title(f"{lab} {ttl}", fontsize=11.5, loc='left')
        axx.set_xlabel(xlab)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(OUT).size
    print(f"\nSaved {OUT}: {w}x{h} (AR={w/h:.3f}; original 1.687)", flush=True)


if __name__ == '__main__':
    main()
