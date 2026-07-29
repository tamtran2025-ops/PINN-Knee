"""Regenerate Figure 7, the physics-versus-correction decomposition.

Left panel: relative correction magnitude against the true knee. Right panel: the
physics-only prediction and the full prediction against the true knee.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, 'Paper_Knee', 'results', 'physics_contribution.csv')
OUT = os.path.join(HERE, 'figures_new', 'image7.png')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 200, "font.family": "serif",
})

rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
true = np.array([float(r['true_knee']) for r in rows])
nphys = np.array([float(r['n_phys']) for r in rows])
pred = np.array([float(r['knee_pred']) for r in rows])
ratio = np.array([float(r['ratio_abs_delta_over_pred']) for r in rows])
strata = np.array([r['strata'] for r in rows])

print(f"n={len(rows)} cell | median rho={np.median(ratio)*100:.1f}% "
      f"| >30%={100*np.mean(ratio>0.3):.1f}%")

COL = {'near': '#2CA02C', 'mid': '#4C72B0', 'far': '#C44E52'}
LAB = {'near': 'near median', 'mid': 'mid', 'far': 'far from median'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.25, 5.0))  # AR ~2.65 (goc 2.671)

# ---- (a) correction ratio ----
for k in ('near', 'mid', 'far'):
    m = strata == k
    ax1.scatter(true[m], ratio[m], s=42, c=COL[k], alpha=0.85,
                edgecolors='black', linewidths=0.5, label=LAB[k])
ax1.axhline(0.30, ls='--', c='black', lw=1.4, label='30% threshold')
ax1.set_xlabel("True knee (cycles)")
ax1.set_ylabel(r"$|\Delta_{\mathrm{NN}}|\,/\,|\hat{n}_{\mathrm{knee}}|$")
ax1.set_title("(a) Correction ratio vs true knee", loc='left')
ax1.grid(ls=':', alpha=0.5); ax1.set_axisbelow(True)
ax1.legend(loc='upper right')

# ---- (b) physics-only vs physics+correction ----
ax2.scatter(true, nphys, s=42, c='#8DA0CB', alpha=0.8, edgecolors='black',
            linewidths=0.4, label=r"$n_{\mathrm{phys}}$")
ax2.scatter(true, pred, s=42, c='#E8A0A0', alpha=0.8, edgecolors='black',
            linewidths=0.4, label=r"$n_{\mathrm{phys}}+\Delta_{\mathrm{NN}}$")
lim = [0, max(true.max(), pred.max()) * 1.05]
ax2.plot(lim, lim, ls='--', c='gray', lw=1.2)
ax2.set_xlim(lim); ax2.set_ylim(bottom=0)
ax2.set_xlabel("True knee (cycles)")
ax2.set_ylabel("Predicted knee (cycles)")
ax2.set_title("(b) Physics-only vs physics+correction", loc='left')
ax2.grid(ls=':', alpha=0.5); ax2.set_axisbelow(True)
ax2.legend(loc='upper left')

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close(fig)
from PIL import Image
w, h = Image.open(OUT).size
print(f"Saved {OUT}: {w}x{h} (AR={w/h:.3f}; original 1.966)")
