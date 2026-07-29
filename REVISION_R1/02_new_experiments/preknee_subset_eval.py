"""R3-2: re-evaluate using only genuinely pre-knee cells (n_knee > n_early).

Within each run, test cells whose true knee falls inside the observation window are
dropped and the MAE is recomputed, then averaged over the 15 runs (5 folds x 3 seeds)
exactly as Table 1 aggregates. The "full" column must reproduce Table 1, which is the
internal control.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _split(s, cast=float):
    return [cast(x) for x in str(s).split('|') if x != '' and x != 'nan']


def per_run_metrics(df):
    """Return a DataFrame: model, n_early, MAE_full, MAE_preknee, n_drop_mean, n_test_mean."""
    out = []
    for (model, ne), g in df.groupby(['model', 'n_early']):
        full, pre, drops, ntest = [], [], [], []
        for _, r in g.iterrows():
            yt = np.array(_split(r['y_true']))
            yp = np.array(_split(r['y_pred']))
            if len(yt) == 0 or len(yt) != len(yp):
                continue
            err = np.abs(yt - yp)
            full.append(err.mean())
            keep = yt > ne                      # keep only genuinely pre-knee cells
            drops.append(int((~keep).sum()))
            ntest.append(len(yt))
            if keep.sum() > 0:
                pre.append(err[keep].mean())
        if not full:
            continue
        out.append(dict(model=model, n_early=ne,
                        MAE_full=np.mean(full), MAE_preknee=np.mean(pre) if pre else np.nan,
                        delta=np.mean(pre) - np.mean(full) if pre else np.nan,
                        n_drop_mean=np.mean(drops), n_test_mean=np.mean(ntest),
                        n_runs=len(full)))
    return pd.DataFrame(out)


def main():
    frames = []
    for fn in ('rerun_exp1_fixed.csv', 'classical_log_full.csv'):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            print(f"  skipped (not found): {fn}")
            continue
        d = pd.read_csv(p)
        if not {'test_cells', 'y_true', 'y_pred'}.issubset(d.columns):
            print(f"  skipped (per-cell columns missing): {fn}")
            continue
        frames.append(d[['model', 'n_early', 'y_true', 'y_pred']])
        print(f"  read {fn}: {len(d)} runs, {d.model.nunique()} models")

    df = pd.concat(frames, ignore_index=True)
    res = per_run_metrics(df).sort_values(['n_early', 'MAE_full'])

    print("\n" + "=" * 92)
    print("MAE on ALL test cells vs ONLY genuinely pre-knee cells (n_knee > n_early)")
    print("=" * 92)
    for ne in (50, 100, 150):
        sub = res[res.n_early == ne]
        if sub.empty:
            continue
        drop = sub.n_drop_mean.mean()
        print(f"\n n_early = {ne}   (mean {drop:.2f} cells dropped per run, "
              f"on ~{sub.n_test_mean.mean():.1f} test cells)")
        print(f"   {'model':<18}{'MAE full':>11}{'MAE pre-knee':>14}{'delta':>9}")
        for _, r in sub.iterrows():
            print(f"   {r.model:<18}{r.MAE_full:>11.2f}{r.MAE_preknee:>14.2f}{r.delta:>+9.2f}")

    out = os.path.join(HERE, 'preknee_subset_eval.csv')
    res.to_csv(out, index=False)
    print(f"\nWrote: {out}")

    # internal control: the 'full' column must match Table 1
    chk = res[res.model == 'PINN_Knee'].set_index('n_early')['MAE_full'].round(1).to_dict()
    print(f"\nCheck: PINN_Knee MAE_full = {chk}  (Table 1: 50->159.2, 100->139.6, 150->117.4)")


if __name__ == '__main__':
    main()
