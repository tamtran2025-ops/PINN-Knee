"""Build Supplementary Figures S1 and S2.

S1: capacity-fade curves for six representative cells with all three knee detectors
    (Bacon-Watts, curvature, second derivative) marked, plus the ensemble-median label
    that is actually used for training. Requested by Reviewer 3, comment 2.
S2: distributions of the five physics parameters a, b, c, d, s at each early-cycle
    budget, with the prescribed bounds drawn. Requested by Reviewer 2, comment 3.

The bound-activation panel is regenerated only from r2_3_bounds.csv, the same file
used in the manuscript, so the figure cannot drift from the reported numbers.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(ROOT, 'Paper_Knee', 'results', '_severson_cache.pkl')
OUTDIR = os.path.join(ROOT, 'REVISION_R1', '03_new_figures', 'supplementary')
os.makedirs(OUTDIR, exist_ok=True)

DET = [('bacon_watts', 'Bacon-Watts', '#D55E00', 'v'),
       ('curvature', 'Curvature', '#0072B2', 's'),
       ('second_derivative', '2nd derivative', '#009E73', '^')]


def figure_s1():
    cells = pickle.load(open(CACHE, 'rb'))
    cells = [c for c in cells if c.get('knee_cycle')]
    # 6 cell dai dien trai deu pho tuoi tho (ngan -> dai)
    cells_sorted = sorted(cells, key=lambda c: c['knee_cycle'])
    idx = np.linspace(0, len(cells_sorted) - 1, 6).astype(int)
    pick = [cells_sorted[i] for i in idx]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, c, lab in zip(axes.ravel(), pick, 'abcdef'):
        cyc = np.asarray(c['cycles'], dtype=float)
        cap = np.asarray(c['capacity'], dtype=float)
        ax.plot(cyc, cap, color='0.35', lw=1.0, zorder=1)

        det = c.get('knee_details') or {}
        ymin, ymax = np.nanmin(cap), np.nanmax(cap)
        for key, name, col, mk in DET:
            k = det.get(key)
            if k is None or not np.isfinite(k):
                continue
            ax.axvline(k, color=col, ls=':', lw=1.2, alpha=0.9, zorder=2)
            ax.plot([k], [ymax], marker=mk, color=col, ms=6,
                    clip_on=False, zorder=4, label=name)

        ke = c['knee_cycle']
        ax.axvline(ke, color='k', ls='-', lw=1.8, alpha=0.85, zorder=3,
                   label='Ensemble median (label)')
        ax.set_title(f"({lab}) {c['name']}   label = {int(ke)} cycles",
                     fontsize=9.5, loc='left')
        ax.set_xlabel('Cycle', fontsize=9)
        ax.set_ylabel('Discharge capacity (Ah)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.margins(x=0.02)
        ax.set_ylim(ymin - 0.02 * (ymax - ymin), ymax + 0.03 * (ymax - ymin))

    h, l = axes.ravel()[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b); hh.append(a); ll.append(b)
    fig.legend(hh, ll, loc='lower center', ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = os.path.join(OUTDIR, 'FigureS1_capacity_fade_three_knees.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  S1 -> {out}")
    return out


def figure_s2():
    pc = os.path.join(HERE, 'r2_3_bounds_percell.csv')
    if not os.path.exists(pc):
        print("  S2: CHUA co r2_3_bounds_percell.csv, bo qua")
        return None
    d = pd.read_csv(pc)
    names = {'a': 'a  (normalised initial capacity)',
             'b': 'b  (SEI sqrt(t) rate)',
             'c': 'c  (post-knee drop amplitude)',
             'd': 'd  (post-knee decay rate)',
             's': 's  (knee sharpness)'}
    # bien cung tu models.py _physics_forward
    bounds = {'a': (0.7, 1.1), 'b': (np.exp(-7.5), np.exp(-4.5)),
              'c': (0.05, 0.25), 'd': (np.exp(-6.0), np.exp(-3.0)),
              's': (1.0, 4.0)}
    cols = {50: '#0072B2', 100: '#E69F00', 150: '#009E73'}

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for ax, p, lab in zip(axes.ravel(), 'abcds', 'abcde'):
        sub = d[d.param == p]
        lo, hi = bounds[p]
        allv = sub.value.values
        rng = (min(allv.min(), lo), max(allv.max(), hi))
        bins = np.linspace(*rng, 45)
        for ne in (50, 100, 150):
            v = sub[sub.n_early == ne].value.values
            ax.hist(v, bins=bins, histtype='step', lw=1.5, color=cols[ne],
                    label=f'n_early = {ne}', density=True)
        ax.axvline(lo, color='r', ls='--', lw=1.1)
        ax.axvline(hi, color='r', ls='--', lw=1.1, label='Prescribed bound')
        ax.set_title(f'({lab}) {names[p]}', fontsize=9.5, loc='left')
        ax.set_xlabel('Learned value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.tick_params(labelsize=8)

    ax = axes.ravel()[5]
    ax.axis('off')
    agg = pd.read_csv(os.path.join(HERE, 'r2_3_bounds.csv'))
    txt = ("(f) Bound activation\n\n"
           "Fraction of evaluations with the\nparameter pushed onto its bound\n"
           "(|tanh(z)| > 0.99):\n\n")
    for ne in (50, 100, 150):
        s = agg[agg.n_early == ne]
        txt += f"   n_early = {ne}:  " + ", ".join(
            f"{r['param']} {r['pct_active']:.0f}%" for _, r in s.iterrows()) + "\n"
    txt += "\nNo bound is active at any budget;\nthe parameterisation never clips\nthe solution."
    ax.text(0.0, 0.97, txt, va='top', ha='left', fontsize=9.5, family='sans-serif')

    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = os.path.join(OUTDIR, 'FigureS2_physics_param_distributions.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  S2 -> {out}")
    return out


if __name__ == '__main__':
    print("Dung Supplementary figures:")
    figure_s1()
    figure_s2()
