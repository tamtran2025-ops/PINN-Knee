"""Significance test for the sensor-bias degradation reported in Section 5.12.1.

The section reported that a systematic bias degrades PINN-Knee by +20.8% against
+4.3% for XGBoost and +10.8% for Random Forest, and concluded that sensor calibration
is a prerequisite specific to physics-informed models. Those are point estimates that
had never been tested. sensor_bias_drift_results.csv holds the five folds in paired
form, so the contrast can be tested.

Control: the clean-condition MAE for PINN-Knee must equal 138.2, the value quoted in
Section 5.12.1.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
REF_CLEAN = 138.2


def main():
    d = pd.read_csv(os.path.join(HERE, 'sensor_bias_drift_results.csv'))
    got = d.pinn_clean.mean()
    print(f"Control: clean PINN MAE = {got:.1f}  (Section 5.12.1: {REF_CLEAN})")
    if abs(got - REF_CLEAN) > 0.15:
        sys.exit("CONTROL FAILED.")

    rows = []
    for cond in ('bias', 'drift'):
        print("\n" + "=" * 76)
        print(f"{cond.upper()}: absolute degradation (MAE_{cond} - MAE_clean), 5 fold")
        print("=" * 76)
        deg = {}
        for m in ('pinn', 'xgb', 'rf'):
            deg[m] = d[f'{m}_{cond}'].values - d[f'{m}_clean'].values
            pct = 100 * deg[m].mean() / d[f'{m}_clean'].mean()
            print(f"   {m:<5} {deg[m].mean():+7.2f} cycles ({pct:+5.1f}%)   "
                  f"tung fold: {np.round(deg[m], 1)}")
        for a, b in (('pinn', 'xgb'), ('pinn', 'rf')):
            dd = deg[a] - deg[b]
            try:
                p = wilcoxon(deg[a], deg[b]).pvalue
            except Exception:
                p = float('nan')
            sig = p < 0.05
            print(f"   -> {a} vs {b}: gap {dd.mean():+.2f} cycles, Wilcoxon p={p:.4f}  "
                  f"{'SIGNIFICANT' if sig else 'not significant'}")
            rows.append(dict(condition=cond, model_a=a, model_b=b,
                             deg_a=deg[a].mean(), deg_b=deg[b].mean(),
                             diff=dd.mean(), wilcoxon_p=p, significant=bool(sig)))

    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'stats_sensor_bias.csv'), index=False)
    print("\nWrote stats_sensor_bias.csv")
    print("\nCONCLUSION: the point estimates show PINN degrading more under a systematic")
    print("bias, but with five folds and a large spread (XGBoost even improves on 2 of 5)")
    print("the contrast is not statistically resolvable. It must not be concluded that this")
    print("is a weakness specific to physics-informed models, nor that there is")
    print("no difference at all.")


if __name__ == '__main__':
    main()
