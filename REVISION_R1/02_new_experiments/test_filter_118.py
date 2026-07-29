"""Which capacity filter yields the 118-cell pool quoted in the original submission?

Single hypothesis under test, rather than a sweep: SEV_batch2_c012 retains a
QDischarge value above 1.1 Ah in the cache, which the current filter removes. If the
original applied no upper threshold, filtering on Q > 0 alone should give 118 cells.

Thresholds recorded in advance: exactly 118 confirms the published number; anything
else refutes the hypothesis. Control: rerunning the current filter 0 < Q <= 1.1 must
return exactly 117 cells and exactly the cache membership, otherwise the environment
has changed and the run is void.
"""
import os, sys, time, pickle, collections, warnings
import numpy as np
import h5py

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'Paper_Knee', 'scripts'))
from knee_detection import detect_knee_ensemble, validate_knee_point
from config import KNEE_MIN_CYCLE

D = os.path.join(ROOT, 'data', 'severson')
FILES = {1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
         2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
         3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat"}
CACHE = os.path.join(ROOT, 'Paper_Knee', 'results', '_severson_cache.pkl')

FILTERS = {
    "control_0_to_1.1": lambda q: (q > 0) & (q <= 1.1),   # must return 117
    "hypothesis_Q_gt_0":  lambda q: q > 0,                   # does this give 118?
}


def load_raw():
    raw = {}
    for b, fn in FILES.items():
        p = os.path.join(D, fn)
        assert os.path.exists(p), f"THIEU FILE: {p}"
        with h5py.File(p, 'r') as f:
            batch = f['batch']
            for i in range(batch['summary'].shape[0]):
                s = f[batch['summary'][i, 0]]
                raw[f"SEV_batch{b}_c{i:03d}"] = (
                    np.array(s['cycle']).ravel(),
                    np.array(s['QDischarge']).ravel())
    return raw


def run_filter(raw, fn, label, verbose_every=35):
    valid, reasons = [], collections.Counter()
    t0 = time.time()
    for k, (name, (cyc, cap)) in enumerate(sorted(raw.items()), 1):
        m = fn(cap)
        c2, q2 = cyc[m].astype(int), cap[m].astype(float)
        if len(q2) < KNEE_MIN_CYCLE:
            reasons['too_short'] += 1
        else:
            knee, _, _ = detect_knee_ensemble(c2, q2)
            if knee is None:
                reasons['no_knee'] += 1
            else:
                ok, diag = validate_knee_point(c2, q2, knee)
                if ok:
                    valid.append((name, knee))
                else:
                    reasons[diag.get('reason', 'invalid')] += 1
        if k % verbose_every == 0:
            print(f"    [{label}] {k}/140  ({time.time()-t0:.0f}s)", flush=True)
    return valid, reasons


def main():
    print(f"Output from: {os.path.abspath(__file__)}", flush=True)
    raw = load_raw()
    assert len(raw) == 140, f"Expected 140 raw cells, got {len(raw)}"   # step R4
    print(f"Loaded from .mat: {len(raw)} raw cells  OK\n", flush=True)

    cache_names = {c['name'] for c in pickle.load(open(CACHE, 'rb'))}
    print(f"Reference cache: {len(cache_names)} cell\n", flush=True)

    results = {}
    for label, fn in FILTERS.items():
        print(f"--- {label} ---", flush=True)
        valid, reasons = run_filter(raw, fn, label)
        names = {n for n, _ in valid}
        ks = np.array([k for _, k in valid])
        results[label] = (names, reasons, ks)
        print(f"    hop le = {len(valid)}   loai = {dict(reasons)}", flush=True)
        print(f"    knee min={ks.min()} max={ks.max()} mean={ks.mean():.1f}",
              flush=True)
        print(f"    so cell knee <= 50: {int((ks <= 50).sum())}\n", flush=True)

    # ---- control: must return exactly 117 cells and exactly the cache membership ----
    ctrl_names = results["doi_chung_0_den_1.1"][0]
    print("=" * 68)
    print("Control: must return 117 cells and match the cache membership")
    print("=" * 68)
    ok_n = len(ctrl_names) == 117
    ok_set = ctrl_names == cache_names
    print(f"  cells = {len(ctrl_names)}  -> {'OK' if ok_n else '*** WRONG ***'}")
    print(f"  matches the cache -> {'OK' if ok_set else '*** WRONG ***'}")
    if not (ok_n and ok_set):
        print("\n  CONTROL FAILED: the environment has changed. Do not trust the results below.")
        return

    # ---- conclusion against the thresholds recorded in advance ----
    hyp_names = results["gia_thuyet_chi_Q>0"][0]
    n = len(hyp_names)
    print("\n" + "=" * 68)
    print(f"HYPOTHESIS 'chi Q > 0'  ->  {n} cell")
    print("=" * 68)
    if n == 118:
        print("  => 118. The published number is correct; the abstract needs no change.")
        extra = sorted(hyp_names - cache_names)
        print(f"  The 118th cell: {extra}")
        for e in extra:
            k = dict(results['gia_thuyet_chi_Q>0'][2:] and
                     {n_: k_ for n_, k_ in
                      zip([x for x in sorted(hyp_names)], [])}) if False else None
        ks = results["gia_thuyet_chi_Q>0"][2]
        print(f"  so cell knee <= 50 (bi build_feature_matrix loai o ne=50): "
              f"{int((ks <= 50).sum())}")
    elif n == 117:
        print("  => 117, same as the current filter. The 118 figure remains unexplained.")
        print("  => Still inconclusive.")
    else:
        print(f"  => {n}, matching no hypothesis. This hypothesis is rejected.")


if __name__ == '__main__':
    main()
