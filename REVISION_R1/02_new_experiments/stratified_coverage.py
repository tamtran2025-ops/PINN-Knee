"""Stratified coverage: is the interval equally trustworthy across lifespan groups?

Cells that die at 200 cycles and cells that last 1600 have very different error scales.
A single marginal coverage figure can hide over-coverage in one group and
under-coverage in another.

Two stratifications are reported: by the true knee, and by the predicted value, the
latter being the one available at deployment time. Exact conditional coverage is not
attainable without further assumptions (Barber et al., 2021, Section 4), so an even
split across groups is reported as a songth and an uneven one is reported plainly
rather than hidden.
"""
import os, sys, csv
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, 'cvplus_conformal.csv')
ALPHA = 0.05


def load():
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    if not rows or 'y_true' not in rows[0]:
        print("The CSV has no per-cell columns (y_true/lower/upper).")
        print("-> rerun cvplus_conformal.py so the per-cell detail is stored.")
        return None
    out = []
    for r in rows:
        yt = np.array([float(v) for v in r['y_true'].split('|')])
        yp = np.array([float(v) for v in r['y_pred'].split('|')])
        lo = np.array([float(v) for v in r['lower'].split('|')])
        hi = np.array([float(v) for v in r['upper'].split('|')])
        out.append((r['n_early'], int(r['fold']), yt, yp, lo, hi))
    return out


def report(tag, y, lo, hi, strat, labels):
    print(f"\n  --- stratified by {tag} ---")
    print(f"    {'group':<22s}{'n':>5s}{'PICP':>9s}{'MPIW':>10s}{'knee TB':>10s}")
    picps = []
    for k, lab in enumerate(labels):
        m = strat == k
        if m.sum() == 0:
            continue
        p = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        w = float(np.mean(hi[m] - lo[m]))
        picps.append(p)
        print(f"    {lab:<22s}{int(m.sum()):>5d}{p:>9.3f}{w:>10.0f}{y[m].mean():>10.0f}")
    if picps:
        print(f"    -> largest PICP by a factor of across strata: "
              f"{max(picps)-min(picps):.3f}")
    return picps


def main():
    data = load()
    if data is None:
        return
    print("=" * 72)
    print(f"  STRATIFIED COVERAGE  -  CV+ (alpha={ALPHA}, target {1-ALPHA:.2f},")
    print(f"  theoretical guarantee >= {1-2*ALPHA:.2f})")
    print("=" * 72)

    for ne in ('50', '100', '150'):
        sub = [d for d in data if d[0] == ne]
        if not sub:
            continue
        y = np.concatenate([d[2] for d in sub])
        p = np.concatenate([d[3] for d in sub])
        lo = np.concatenate([d[4] for d in sub])
        hi = np.concatenate([d[5] for d in sub])
        cover = float(np.mean((y >= lo) & (y <= hi)))
        print(f"\n{'='*72}\n  n_early = {ne}   (n={len(y)} cell predictions)")
        print(f"  PICP marginal = {cover:.3f}   MPIW = {np.mean(hi-lo):.0f}")

        # terciles of the TRUE lifespan
        q = np.quantile(y, [1/3, 2/3])
        strat = np.digitize(y, q)
        report("TRUE lifespan (labels used)", y, lo, hi, strat,
               ["short-lived (bottom third)", "medium-lived", "long-lived (top third)"])

        # terciles of the PREDICTED value, which is what is available at deployment time
        qp = np.quantile(p, [1/3, 2/3])
        stratp = np.digitize(p, qp)
        report("PREDICTED (no labels used)", y, lo, hi, stratp,
               ["low prediction", "medium prediction", "high prediction"])

    print("""
=======================================================================
  HOW TO READ
=======================================================================
  Conformal only guarantees MARGINAL coverage. Conditional coverage la
  is not attainable without further assumptions (Barber et al., 2021, Section 4);
  Vovk 2012; Lei & Wasserman 2014). Therefore:

  - If the spread across strata is below about 0.05, report it plainly, citing the known
    a theoretical limit. Reporting it plainly is the complete and honest answer.
  - If the imbalance is large it must not be hidden. Report it, and if needed use
    locally adaptive or normalised conformal variants.
""")


if __name__ == '__main__':
    main()
