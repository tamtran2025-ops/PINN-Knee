"""R1-5: export the extracted feature matrix so others need not reprocess the .mat files.

One CSV per early-cycle budget, one row per cell, produced by the same
build_feature_matrix routine used by every experiment in the paper, so the row counts
match the evaluation pools of 115, 111 and 106 cells.

The raw Severson .mat files are not redistributed here. Place them as described in
REPRODUCE.md, section 1, and rerun; the cache is built on the first run.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments')):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding='utf-8')

from features import build_feature_matrix          # noqa: E402
from rerun_exp1_fixed import load_paper_pool       # noqa: E402


def main():
    try:
        cells = load_paper_pool()
    except FileNotFoundError:
        sys.exit(
            "results/_severson_cache.pkl not found. This script needs the raw data, "
            "which are not redistributed in this repository. Place the .mat files as described "
            "in REPRODUCE.md section 1 and rerun; the cache is built on the first run. "
            "Ma tran dac trung da trich san co tai features_n_early_50/100/150.csv."
        )
    print(f"Pool: {len(cells)} cell")
    for ne in (50, 100, 150):
        X, y, names, idx = build_feature_matrix(cells, ne)
        used = [cells[i]['name'] for i in idx]
        df = pd.DataFrame(X, columns=names)
        df.insert(0, 'knee_cycle', y.astype(int))
        df.insert(0, 'cell_name', used)
        out = os.path.join(HERE, f'features_n_early_{ne}.csv')
        df.to_csv(out, index=False)
        print(f"  n_early={ne:>3}: {df.shape[0]} cell x {len(names)} dac trung -> "
              f"{os.path.basename(out)}  (knee nho nhat {int(y.min())})")


if __name__ == '__main__':
    main()
