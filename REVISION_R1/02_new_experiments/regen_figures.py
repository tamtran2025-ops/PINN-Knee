"""Regenerate the main-text figures from the corrected result files.

Aspect ratios are kept identical to the originals so the images are not stretched when
embedded in the document. Classical baselines are read from classical_log_full.csv and
the neural models from rerun_exp1_fixed.csv. Seeds are averaged within each fold and
budget, giving the 15 points per model quoted in the captions.
"""
import os, sys, csv, collections, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_new')
os.makedirs(OUT, exist_ok=True)
F_NN = os.path.join(HERE, 'rerun_exp1_fixed.csv')
F_CL = os.path.join(HERE, 'classical_log_full.csv')
F_DE = os.path.join(HERE, 'data_efficiency_fixed.csv')

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 200, "font.family": "serif",
})
COLORS = {
    "PINN_Knee": "#C44E52", "XGBoost": "#4C72B0", "RandomForest": "#8172B3",
    "GaussianProcess": "#64B5CD", "Pure_NN": "#55A868", "Ensemble_NN": "#CCB974",
    "Neural_ODE": "#2CA02C", "LSTM": "#9467BD", "GRU": "#7F7F7F",
    "Transformer": "#8C564B", "Informer": "#BCBD22", "Bayesian_LSTM": "#17BECF",
    "PatchTST": "#E377C2", "PINN_UQ": "#AEC7E8",
}
PRETTY = {"PINN_Knee": "PINN Knee", "Pure_NN": "Pure NN", "Ensemble_NN": "Ensemble NN",
          "Neural_ODE": "Neural ODE", "Bayesian_LSTM": "Bayesian LSTM", "PINN_UQ": "PINN UQ"}
MODELS = list(COLORS.keys())
CLASSICAL = {"XGBoost", "RandomForest", "GaussianProcess"}
NES = [50, 100, 150]


def load_merged():
    """Rows: classical baselines from classical_log_full, the rest from rerun_exp1_fixed."""
    rows = []
    for r in csv.DictReader(open(F_NN, encoding='utf-8')):
        if r.get('status', 'ok') != 'ok':
            continue
        if r['model'] in CLASSICAL:
            continue
        rows.append(r)
    for r in csv.DictReader(open(F_CL, encoding='utf-8')):
        rows.append(r)
    return rows


ROWS = load_merged()


def cellstat(ne):
    """model -> (mean_MAE, std_MAE) tren 15 do."""
    agg = collections.defaultdict(list)
    for r in ROWS:
        if int(float(r['n_early'])) == ne:
            agg[r['model']].append(float(r['MAE']))
    return {m: (np.mean(v), np.std(v)) for m, v in agg.items()}


def fold_matrix(ne):
    """DataFrame (5 fold x 14 model) MAE trung binh theo seed."""
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ROWS:
        if int(float(r['n_early'])) == ne:
            d[int(float(r['fold']))][r['model']].append(float(r['MAE']))
    folds = sorted(d)
    data = {m: [np.mean(d[f][m]) for f in folds] for m in MODELS}
    return pd.DataFrame(data, index=folds)[MODELS]


def critical_difference(k, n):
    q05 = {14: 3.354}
    return q05.get(k, 3.354) * np.sqrt(k * (k + 1) / (6.0 * n))


# ===================== FIG 1: 3-panel MAE bars =====================
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.60))  # AR 3.26
    panel = ['(a) ', '(b) ', '(c) ']
    for pi, (ax, ne) in enumerate(zip(axes, NES)):
        st = cellstat(ne)
        order = sorted(st, key=lambda m: st[m][0])       # tang dan MAE
        y = np.arange(len(order))[::-1]                  # tot nhat o TREN
        means = [st[m][0] for m in order]
        stds = [st[m][1] for m in order]
        cols = [COLORS[m] for m in order]
        ax.barh(y, means, xerr=stds, color=cols, edgecolor='black',
                linewidth=0.6, error_kw=dict(ecolor='black', capsize=2, lw=0.8))
        ax.set_yticks(y)
        labels = [PRETTY.get(m, m) for m in order]
        ax.set_yticklabels(labels)
        for tick, m in zip(ax.get_yticklabels(), order):
            if m == 'PINN_Knee':
                tick.set_color(COLORS['PINN_Knee']); tick.set_fontweight('bold')
        ax.set_xlabel("MAE (cycles)")
        ax.set_title(panel[pi] + r"$n_{\mathrm{early}}=%d$" % ne, loc='left')
        ax.grid(axis='x', ls=':', alpha=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'image1.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[Fig1] xong", flush=True)


# ===================== FIG 2-4: CD diagrams =====================
def cd_figs():
    for ne, name in zip(NES, ('image2.png', 'image3.png', 'image4.png')):
        piv = fold_matrix(ne)
        data = piv.values
        stat, pval = friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
        nem = sp.posthoc_nemenyi_friedman(data)
        nem.index = list(piv.columns); nem.columns = list(piv.columns)
        ranks = pd.DataFrame(data, columns=list(piv.columns)).rank(axis=1).mean(axis=0)
        ranks = ranks.sort_values()
        nem = nem.reindex(index=ranks.index, columns=ranks.index)
        fig, ax = plt.subplots(figsize=(10.0, 3.07))     # AR 3.26
        sp.critical_difference_diagram(
            ranks=ranks, sig_matrix=nem, ax=ax,
            label_fmt_left="{label} ({rank:.2f})  ",
            label_fmt_right="  ({rank:.2f}) {label}")
        cd = critical_difference(data.shape[1], data.shape[0])
        ax.set_title(r"CD diagram at $n_{\mathrm{early}}=%d$ (Friedman $p=%.2e$, CD$=%.2f$)"
                     % (ne, pval, cd))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, name), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"[CD ne={ne}] chi2={stat:.1f} p={pval:.2e} rank_min={ranks.min():.2f}", flush=True)


# ===================== FIG 5: early prediction (3 budget) =====================
def fig5():
    sub = ['PINN_Knee', 'GaussianProcess', 'Pure_NN']
    mk = {'PINN_Knee': 'o', 'GaussianProcess': 's', 'Pure_NN': '^'}
    fig, ax = plt.subplots(figsize=(9.0, 5.95))          # AR 1.512
    for m in sub:
        ys = [cellstat(ne)[m][0] for ne in NES]
        es = [cellstat(ne)[m][1] for ne in NES]
        ax.errorbar(NES, ys, yerr=es, marker=mk[m], color=COLORS[m], lw=2,
                    markersize=8, capsize=3, label=PRETTY.get(m, m))
    ax.set_xlabel(r"Number of early cycles ($n_{\mathrm{early}}$)")
    ax.set_ylabel("Test MAE (cycles)")
    ax.set_title("Early prediction performance")
    ax.set_xticks(NES)
    ax.grid(ls=':', alpha=0.5); ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'image5.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[Fig5] xong", flush=True)


# ===================== FIG 8: data efficiency =====================
def fig8():
    rows = list(csv.DictReader(open(F_DE, encoding='utf-8')))
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        d[r['model']][int(r['n_train_used'])].append(float(r['MAE']))
    sub = ['PINN_Knee', 'XGBoost', 'Pure_NN']
    mk = {'PINN_Knee': 'o', 'XGBoost': 's', 'Pure_NN': '^'}
    fig, ax = plt.subplots(figsize=(7.0, 4.78))          # AR 1.463
    for m in sub:
        ns = sorted(d[m])
        ys = [np.mean(d[m][n]) for n in ns]
        es = [np.std(d[m][n]) for n in ns]
        ax.errorbar(ns, ys, yerr=es, marker=mk[m], color=COLORS[m], lw=2,
                    markersize=8, capsize=3, label=PRETTY.get(m, m))
    ax.set_xlabel("Training set size (cells)")
    ax.set_ylabel("Test MAE (cycles)")
    ax.set_title("Data efficiency")
    ax.grid(ls=':', alpha=0.5); ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'image8.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[Fig8] xong", flush=True)


# ===================== FIG 12: per-fold boxplot =====================
def fig12():
    # average the seeds within each (fold, budget), giving 15 points per model as in the caption
    tmp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ROWS:
        key = (int(float(r['n_early'])), int(float(r['fold'])))
        tmp[r['model']][key].append(float(r['MAE']))
    perfold = {m: [np.mean(v) for v in d.values()] for m, d in tmp.items()}
    order = sorted(MODELS, key=lambda m: np.median(perfold[m]))
    fig, ax = plt.subplots(figsize=(8.0, 8.47))          # AR 0.944
    data = [perfold[m] for m in order]
    bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.4),
                    medianprops=dict(color='black', lw=1.2))
    for patch, m in zip(bp['boxes'], order):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.85)
    ax.set_yticklabels([PRETTY.get(m, m) for m in order])
    for tick, m in zip(ax.get_yticklabels(), order):
        if m == 'PINN_Knee':
            tick.set_color(COLORS['PINN_Knee']); tick.set_fontweight('bold')
    ax.set_xlabel("MAE (cycles) across all folds and budgets")
    ax.set_title("Per-fold MAE distribution (14 models)")
    ax.grid(axis='x', ls=':', alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'image13.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[Fig12] xong", flush=True)


if __name__ == '__main__':
    fig1(); cd_figs(); fig5(); fig8(); fig12()
    print("\nXONG 7 figure ->", OUT, flush=True)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        from PIL import Image
        w, h = Image.open(p).size
        print(f"  {f}: {w}x{h} (AR={w/h:.3f})")
