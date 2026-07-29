# PINN-Knee: Physics-Constrained Deep Learning with Conformal Prediction for Early Knee-Point Prediction in Lithium-Ion Batteries

Official open-source repository for the paper:
**Physics-Constrained Deep Learning with Conformal Prediction for Early Knee-Point Prediction in Lithium-Ion Batteries**
*Journal of Energy Storage (Elsevier), Manuscript Ref: EST-D-26-06495 (major revision)*

---

## Overview

PINN-Knee is a **Residual Physics** architecture for early knee-point prediction in
lithium-ion batteries. A physics head produces five interpretable degradation
parameters `(a, b, c, d, s)` that drive a closed-form log-knee formula (Eq. 3), and a
bounded neural correction head contributes a small data-driven adjustment. The trained
estimator is wrapped in **CV+ conformal prediction** (Barber et al., 2021) for
distribution-free, finite-sample-valid uncertainty quantification.

This repository contains the full, corrected evaluation pipeline used in the revised
manuscript. All numbers below are reproducible from the scripts and result files here.

## Headline results (honest summary)

Evaluated on **117 Severson LFP cells**, 5-fold CV, 3 seeds per fold, under a **unified
log-target protocol for all 14 models**:

- **Point accuracy:** PINN-Knee attains the **lowest MAE at n_early = 50 and 100** and
  is within **1.0 cycle** of the best model (a Gaussian Process) at n_early = 150; it
  attains the **lowest Median AE and Within-100 fraction at all three budgets**.
  Differences within the top cluster (PINN-Knee, PINN-UQ, Gaussian Process, Random
  Forest) are **not statistically significant** (overlapping BCa intervals; we do not
  claim uniform dominance).
- **Architecture value:** over a 20-repeat 5-fold CV the physics-structured architecture
  beats a parameter-matched MLP by a mean of **8.40 cycles** (74/100 folds). We report
  **both** tests: paired t p = 1.3e-4 and Wilcoxon p < 1e-4 (supported by a
  variance-ratio fold-independence diagnostic), **and** the conservative Nadeau-Bengio
  correction p = 0.43. We therefore describe the advantage as **consistent but modest**,
  not as statistical dominance.
- **Physics losses** do not change point accuracy (removing all five: -0.5 cycles, n.s.);
  their role is to regularize the physics parameters toward physically meaningful values.
  The **bounded correction head is essential** (removing it worsens MAE by ~40 cycles).
- **Uncertainty:** CV+ conformal achieves **95.8-97.4% empirical coverage** against its
  >=90% finite-sample guarantee, with interval width narrowing 1038 -> 731 cycles.
- **Transfer:** single-cell fine-tuning to NASA LiCoO2 reduces MAE from 127.4 (zero-shot)
  to 37.9-46.2 cycles. Transfer to the more distant NCM chemistry (Tongji) is poor
  (MAE ~317), reported transparently as a limitation.

## Repository structure

The layout mirrors the directory structure the scripts expect, so every script runs
unchanged after a clone (no path editing required).

```
├── Paper_Knee/
│   ├── scripts/                      # Core model + training code
│   │   ├── models.py                 #   PINN_Knee and all baselines
│   │   ├── train.py                  #   training loop (incl. train_pinn_knee)
│   │   ├── features.py               #   24-feature extraction
│   │   ├── config.py                 #   hyperparameters, PHYSICS_LAMBDA
│   │   ├── metrics.py                #   MAE / MedianAE / Within-k
│   │   ├── knee_detection.py         #   three-detector knee ensemble
│   │   ├── data_loader.py            #   dataset loading and knee validation
│   │   └── _experiments/             #   run_experiments.py, sensitivity_eq3.py, nasa_finetune.py
│   └── results/                      # Inputs consumed by the revision scripts
│                                     #   (sensitivity_eq3_summary.json, lambda_sensitivity.csv,
│                                     #    nasa_finetune.csv, physics_contribution.csv, ...)
├── REVISION_R1/02_new_experiments/   # Revision scripts and their result files
│   ├── rerun_exp1_fixed.py           # Table 1: 630-run benchmark (broadcast bug fixed)
│   ├── classical_log_w100.py         # Table 1: classical baselines under the log target
│   ├── recompute_stats_for_manuscript.py  # Tables 1, 2, 3 and S1, S2, S5, S8
│   ├── ablation_architecture.py      # Architecture ablation (full / arch-only / physics-only / matched MLP)
│   ├── repeated_cv_architecture.py   # 5x repeated 5-fold CV of the ablation
│   ├── variance_ratio_20rep.py       # 20-repeat CV: architecture vs matched MLP (+8.40 cycles)
│   ├── variance_ratio_analysis.py    #   paired t, Nadeau-Bengio, fold-independence diagnostic
│   ├── nested_cv_eq3.py              # Leakage-free nested CV of the Eq. (3) constants (Section 5.1.5)
│   ├── cvplus_conformal.py           # CV+ conformal prediction (Section 5.3, Table S4)
│   ├── stratified_coverage.py        # Lifespan-stratified coverage
│   ├── r1_7_depth_correct.py         # Head-depth ablation (Section 5.4.1)
│   ├── r2_1_threshold_correct.py     # Knee-label sensitivity (Section 5.1.4)
│   ├── r2_3_bounds_correct.py        # Physics-parameter bound activation (Section 5.5)
│   ├── r3_3_logvsraw_v2.py           # Log vs. raw target, stratified (Table S9)
│   ├── batch_transfer_fixed.py       # Cross-protocol transfer (Section 5.8)
│   ├── data_efficiency_fixed.py      # Data efficiency (Section 5.6)
│   ├── sensor_drift.py               # Sensor bias and correlated drift (Section 5.12.1)
│   ├── tongji_finetune_v2.py         # Cross-chemistry NCM probe (Limitations)
│   ├── preknee_subset_eval.py        # R3-2: metrics on genuine pre-knee cells only
│   ├── export_features.py            # R1-5: writes the extracted 24-feature matrices
│   ├── features_n_early_*.csv        #   the extracted features themselves (117/113/108 cells)
│   ├── build_supp_figures.py         # Supplementary Figures S1-S2
│   ├── regen_figures.py              # Regenerate Figures 1-5, 8, 12
│   ├── build_supplementary.py        # Build Supplementary Tables S1-S9
│   └── *.csv, manuscript_stats.json  # All verified result files
├── requirements.txt
├── REPRODUCE.md                      # Step-by-step reproduction
└── LICENSE                           # MIT
```

## Quick start

```bash
git clone https://github.com/tamtran2025-ops/PINN-Knee.git
cd PINN-Knee
pip install -r requirements.txt
```

All commands are run from `REVISION_R1/02_new_experiments`:

```bash
cd REVISION_R1/02_new_experiments
```

The statistics of Tables 1, 2, 3 and S1, S2, S5, S8 can be recomputed directly from the
result files shipped in this repository, without re-running any training:

```bash
python recompute_stats_for_manuscript.py
```

To re-run the training itself (this requires the raw datasets, see Data below):

```bash
python rerun_exp1_fixed.py                # 630-run benchmark
python classical_log_w100.py              # classical baselines under the log target
python variance_ratio_20rep.py && python variance_ratio_analysis.py
python nested_cv_eq3.py                   # leakage-free protocol (Section 5.1.5)
python cvplus_conformal.py                # CV+ conformal (Section 5.3)
```

Figures and the supplementary tables:

```bash
python regen_figures.py                   # Figures 1-5, 8, 12
python build_supplementary.py             # Supplementary Tables S1-S9
```

See **REPRODUCE.md** for the full mapping from each script to the manuscript
table/figure it produces, and for the raw-data placement (Severson / NASA / Tongji).

## Data

The datasets are public and are **not** redistributed here:

- **Severson LFP** (primary): Toyota Research Institute, https://data.matr.io/1/
- **NASA LiCoO2** (cross-chemistry): NASA Ames Prognostics Center of Excellence
- **Tongji NCM** (cross-chemistry probe): Tongji University degradation dataset

Place the raw files as described in `REPRODUCE.md`; a cached knee-annotated pickle is
produced on first run.

**The extracted features are shipped here** so that the results can be reproduced without
re-processing the raw `.mat` files: `features_n_early_50.csv`, `features_n_early_100.csv`
and `features_n_early_150.csv` each hold one row per valid cell (117 / 113 / 108 rows) with
the cell name, the knee label, and the 24 extracted features.

## Citation

```bibtex
@article{trang2026pinnknee,
  title={Physics-Constrained Deep Learning with Conformal Prediction for Early Knee-Point Prediction in Lithium-Ion Batteries},
  author={Tran, Thanh Trang and Tran, Nhut Tam},
  journal={Journal of Energy Storage},
  year={2026},
  note={Under review, EST-D-26-06495}
}
```

## License

MIT License - see [LICENSE](LICENSE).
