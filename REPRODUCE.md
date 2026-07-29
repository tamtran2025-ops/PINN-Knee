# Reproduction guide

Every table and figure in the revised manuscript is reproducible from this repository.
All neural runs use the corrected trainer (the `(N,1)` vs `(N,)` broadcast bug in the
original `train_nn_model` is fixed by the explicit `.squeeze(-1)` in `rerun_exp1_fixed.py`).

## 0. Environment

```bash
pip install -r requirements.txt      # Python 3.11 tested
cd REVISION_R1/02_new_experiments    # all scripts are run from here
```

## 1. Data placement

The raw datasets are public and not redistributed. Place them as:

```
data/severson/2017-05-12_batchdata_updated_struct_errorcorrect.mat   (batch 1)
data/severson/2017-06-30_batchdata_updated_struct_errorcorrect.mat   (batch 2)
data/severson/2018-04-12_batchdata_updated_struct_errorcorrect.mat   (batch 3)
data/nasa/   ... B0005, B0006, B0007
data/tongji/ ... Dataset_2_NCM
```

On first run the knee-detection pipeline (`Paper_Knee/scripts/knee_detection.py`, `Paper_Knee/scripts/data_loader.py`)
produces `results/_severson_cache.pkl`. The retained pool is **117 cells**: the filter
keeps discharge-capacity points with `0 < Q <= 1.1`, requires >= 30 valid cycles, runs a
three-detector ensemble (Bacon-Watts, curvature, second-derivative), takes the median as
the knee label, and applies a post-knee acceleration plausibility check. Knee cycles
range 63-1681 (mean 450, median 432).

Cells whose knee falls inside the observation window are not valid early-prediction targets
and are dropped at that budget: 0 / 4 / 9 cells at n_early = 50 / 100 / 150, leaving
117 / 113 / 108 valid cells with a minimum knee of 63 / 102 / 208 cycles.

> Note: an earlier draft reported 118 cells; that count could not be reproduced from the
> raw `.mat` files under any capacity filter, so the revision uses the reproducible 117.

## 2. Script -> manuscript output map

| Script | Produces | Manuscript |
|---|---|---|
| `rerun_exp1_fixed.py` | `rerun_exp1_fixed.csv` (630 runs, 14 models) | Table 1 (NN/seq rows), Fig 1, 5, 12 |
| `classical_log_w100.py` | `classical_log_full.csv` | Table 1 (classical rows, log-target) |
| `recompute_stats_for_manuscript.py` | `manuscript_stats.json` | Tables 1, 2, 3; S1, S2, S5, S8 |
| `ablation_architecture.py` | `ablation_architecture.csv` | Section 5.4.1 (single-split ablation) |
| `repeated_cv_architecture.py` | `repeated_cv_architecture.csv` | Section 5.4.1 (5x repeated CV) |
| `variance_ratio_20rep.py` + `variance_ratio_analysis.py` | `variance_ratio_20rep.csv` | Section 5.4.1 (+8.40 cycles; paired-t + Nadeau-Bengio) |
| `nested_cv_eq3.py` | leakage test | Section on target-leakage (R3-1) |
| `cvplus_conformal.py` | `cvplus_conformal.csv` | Section 5.3, Table S4 |
| `r1_7_depth_correct.py` | `r1_7_depth.csv` | Section 5.4.1 depth ablation |
| `r2_1_threshold_correct.py` | `r2_1_threshold.csv` | Section 5.1.4 |
| `r2_3_bounds_correct.py` | `r2_3_bounds.csv` | Section 5.5 (bound activation) |
| `r3_3_logvsraw_v2.py` | `r3_3_logvsraw.csv` | Table S9 |
| `batch_transfer_fixed.py` | `batch_transfer_fixed.csv` | Section 5.8, Table 4 |
| `data_efficiency_fixed.py` | `data_efficiency_fixed.csv` | Section 5.6, Fig 8 |
| `tongji_finetune_v2.py` | `tongji_finetune_v2.csv` | Limitations (NCM probe) |
| `preknee_subset_eval.py` | `preknee_subset_eval.csv` | Section 4.1 (pre-knee-only evaluation, R3-2) |
| `export_features.py` | `features_n_early_{50,100,150}.csv` | Extracted features (R1-5) |
| `build_supp_figures.py` | `FigureS1/S2*.png` | Supplementary Figures S1-S2 |
| `uq_gp_vs_cvplus.py` | `uq_gp_vs_cvplus.csv` | Table S10 (GP Bayesian coverage) |
| `uq_matched_coverage.py` | `uq_matched_coverage.csv` | Table S10 (matched-coverage widths) |
| `uq_gp_cvplus_ablation.py` | `uq_gp_cvplus.csv` | Table S10 (CV+ on GP, attribution) |
| `check_protocol_fairness.py` / `check_protocol_pinn.py` | `check_protocol_pinn.csv` | Section 4.3 (data-allocation check) |
| `stats_median_within.py` | `stats_median_within.csv` | Table S11 (paired tests on median AE) |
| `regen_figures.py` | `figures_new/*.png` | Figures 1-5, 8, 12 |
| `build_supplementary.py` | `Supplementary_Material.docx` | Supplementary S1-S9 |

## 2b. Which scripts need the raw data, and which do not

The repository ships every result file, so the tables and statistics can be reproduced
without downloading anything. Scripts fall into three groups:

**Run immediately, no raw data needed** (they read the shipped CSVs):

```bash
python recompute_stats_for_manuscript.py   # Tables 1, 2, 3 and S1, S2, S5, S8
python preknee_subset_eval.py              # pre-knee-only evaluation (Section 4.1)
python variance_ratio_analysis.py          # architecture vs matched MLP (Section 5.4.1)
python stats_median_within.py              # Table S11 paired tests on median AE
python build_supp_figures.py               # Supplementary Figures S1-S2
python build_supplementary.py              # Supplementary Tables S1-S10 + Figures S1-S2
```

`build_supp_figures.py` needs `results/_severson_cache.pkl` for Figure S1; the two
figures are also shipped in `REVISION_R1/03_new_figures/supplementary/` so that
`build_supplementary.py` reproduces the submitted document exactly either way. If the
figures are missing it prints a warning and emits a tables-only document.

**Need the raw datasets** (they retrain models): `rerun_exp1_fixed.py`,
`classical_log_w100.py`, `variance_ratio_20rep.py`, `nested_cv_eq3.py`,
`cvplus_conformal.py`, `ablation_architecture.py`, `r1_7_depth_correct.py`,
`r2_1_threshold_correct.py`, `r2_3_bounds_correct.py`, `r3_3_logvsraw_v2.py`,
`batch_transfer_fixed.py`, `data_efficiency_fixed.py`, `sensor_drift.py`,
`tongji_finetune_v2.py`, `export_features.py`, `uq_gp_vs_cvplus.py`,
`uq_matched_coverage.py`, `uq_gp_cvplus_ablation.py`, `check_protocol_fairness.py`,
`check_protocol_pinn.py`, `diag_physics_loss.py`.

Without the raw files these exit with a message pointing back to Section 1; they do not
fail with an unexplained traceback.

**Verification built into the scripts.** Every script added during the revision checks
itself against a number already published in the paper before reporting anything new,
and stops if the check fails. For example `preknee_subset_eval.py` must reproduce the
Table 1 MAE of PINN-Knee (159.2 / 139.6 / 117.4) and `uq_matched_coverage.py` must
reproduce the Table 1 MAE of the Gaussian Process (162.5 / 147.7 / 116.4).

## 3. One-command reproduction of the statistics

After `rerun_exp1_fixed.csv` and `classical_log_full.csv` exist:

```bash
python recompute_stats_for_manuscript.py
```

This prints Table 1 (ranked), the Wilcoxon table (S1), Friedman/Nemenyi (S2), the paired
BCa CIs (Table 2), the per-model BCa means (S8), and the per-loss ablation (Table 3/S5),
and writes `manuscript_stats.json`.

## 4. Notes on honesty of the reported numbers

- The architecture-vs-MLP advantage is reported with **both** a paired test and the
  conservative Nadeau-Bengio correction; the paper describes it as *consistent but
  modest*, not as statistical dominance.
- The physics-loss ablation (`recompute_stats_for_manuscript.py`, Table 3) shows the
  losses are **not** an accuracy lever; they regularize the physics parameters.
- The Tongji NCM fine-tune (`tongji_finetune_v2.py`) is highly seed-dependent and is
  **not** reported as a result; it appears only as a limitation.
