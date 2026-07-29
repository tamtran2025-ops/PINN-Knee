"""Package the public GitHub repository, preserving the directory layout so scripts run.

An earlier version flattened everything into scripts/ and results/, which broke the
16 of 20 scripts that resolve paths relative to their own location. This version
mirrors the original tree and does not modify a single line of any script.

Included beyond the core code: the scripts cited by name in the manuscript, the
Response or REPRODUCE.md, the uncertainty comparisons behind Tables S10 and S12, the
physics-loss diagnostic behind Section 3.3.3, the sensor-bias test behind Section
5.12.1, and the two supplementary figures so the Supplementary can be rebuilt exactly
as submitted.
"""
import os, shutil, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(ROOT, 'REVISION_R1', 'github_repo')
EXP_SRC = os.path.join(ROOT, 'REVISION_R1', '02_new_experiments')
SC_SRC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
RES_SRC = os.path.join(ROOT, 'Paper_Knee', 'results')

# clear the old layout (src/ scripts/ results/) before rebuilding
for old in ('src', 'scripts', 'results'):
    p = os.path.join(REPO, old)
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"  removing old directory: {old}/")

D_SC = os.path.join(REPO, 'Paper_Knee', 'scripts')
D_SCE = os.path.join(D_SC, '_experiments')
D_SCA = os.path.join(D_SC, '_analysis')
D_RES = os.path.join(REPO, 'Paper_Knee', 'results')
D_EXP = os.path.join(REPO, 'REVISION_R1', '02_new_experiments')
for d in (D_SC, D_SCE, D_SCA, D_RES, D_EXP):
    os.makedirs(d, exist_ok=True)

CORE = ['models.py', 'train.py', 'features.py', 'config.py', 'metrics.py',
        'knee_detection.py', 'data_loader.py']
CORE_EXP = ['run_experiments.py', 'sensitivity_eq3.py', 'nasa_finetune.py',
            # BAT BUOC: sensitivity_eq3/nasa_finetune import severson_only;
            # lambda_sensitivity is cited by REPRODUCE.md and by the Response (Table S6)
            'severson_only.py', 'lambda_sensitivity.py']
CORE_ANA = ['uncertainty.py']   # train.py + run_experiments.py import module nay
SCRIPTS = [
    'rerun_exp1_fixed.py', 'classical_log_w100.py', 'recompute_stats_for_manuscript.py',
    'ablation_architecture.py', 'repeated_cv_architecture.py', 'variance_ratio_20rep.py',
    'variance_ratio_analysis.py', 'repeated_cv_analysis.py', 'nested_cv_eq3.py',
    'cvplus_conformal.py', 'stratified_coverage.py', 'r1_7_depth_correct.py',
    'r2_1_threshold_correct.py', 'r2_3_bounds_correct.py', 'r3_3_logvsraw_v2.py',
    'batch_transfer_fixed.py', 'data_efficiency_fixed.py', 'sensor_drift.py',
    'tongji_finetune_v2.py', 'regen_figures.py', 'regen_fig7.py', 'regen_fig6.py',
    'build_supplementary.py', 'package_github_repo_v3.py',
    # added after the 2026-07-23 review: scripts cited by name in the paper or Response
    'preknee_subset_eval.py', 'export_features.py', 'build_supp_figures.py',
    'r2_3_bounds_percell.py', 'check_robust_metrics_multisplit.py',
    'physics_contribution_oos.py', 'test_filter_118.py',
    # uncertainty comparison, CV+ against Bayesian, plus the protocol check (Table S10, Section 4.3)
    'uq_gp_vs_cvplus.py', 'uq_matched_coverage.py', 'uq_gp_cvplus_ablation.py',
    'check_protocol_fairness.py', 'check_protocol_pinn.py',
    'diag_physics_loss.py',   # magnitude of each loss term (Section 3.3.3)
    'stats_median_within.py', # paired testing for median AE and within-100 (Table S11)
    'uq_conditional_coverage.py',  # coverage within each lifespan group (Table S12)
    'stats_sensor_bias.py',        # significance test for the degradation gap (Section 5.12.1)
]
CSVS = ['rerun_exp1_fixed.csv', 'classical_log_full.csv', 'classical_log_table1.csv',
        'manuscript_stats.json', 'ablation_architecture.csv',
        'repeated_cv_architecture.csv', 'variance_ratio_20rep.csv',
        'cvplus_conformal.csv', 'nested_cv_eq3.csv', 'r1_7_depth.csv',
        'r2_1_threshold.csv', 'r2_3_bounds.csv', 'r3_3_logvsraw.csv',
        'batch_transfer_fixed.csv', 'data_efficiency_fixed.csv',
        'sensor_bias_drift_results.csv', 'tongji_finetune_v2.csv',
        'preknee_subset_eval.csv', 'r2_3_bounds_percell.csv', 'robust_metrics_5splits.csv',
        # R1-5: ma tran dac trung DA TRICH, de nguoi khac khoi xu ly lai .mat
        'features_n_early_50.csv', 'features_n_early_100.csv', 'features_n_early_150.csv',
        'uq_gp_vs_cvplus.csv', 'uq_matched_coverage.csv', 'uq_gp_cvplus.csv',
        'check_protocol_pinn.csv', 'stats_median_within.csv', 'uq_conditional_coverage.csv', 'stats_sensor_bias.csv']
RES_FILES = ['sensitivity_eq3_summary.json', 'lambda_sensitivity.csv',
             'nasa_finetune.csv', 'cross_chem_tongji_results.csv',
             'physics_contribution.csv', 'physics_params_per_cell_agg.csv',
             'shap_importance.json']


def cp(src_dir, names, dst_dir, label):
    n, miss = 0, []
    for f in names:
        s = os.path.join(src_dir, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, f)); n += 1
        else:
            miss.append(f)
    print(f"  {label}: {n} file" + (f"  (thieu: {miss})" if miss else ""))


cp(SC_SRC, CORE, D_SC, 'Paper_Knee/scripts')
cp(os.path.join(SC_SRC, '_experiments'), CORE_EXP, D_SCE, 'Paper_Knee/scripts/_experiments')
cp(os.path.join(SC_SRC, '_analysis'), CORE_ANA, D_SCA, 'Paper_Knee/scripts/_analysis')
cp(RES_SRC, RES_FILES, D_RES, 'Paper_Knee/results')
cp(EXP_SRC, SCRIPTS, D_EXP, 'REVISION_R1/02_new_experiments (script)')
cp(EXP_SRC, CSVS, D_EXP, 'REVISION_R1/02_new_experiments (ket qua)')

# supplementary figures, shipped so the Supplementary rebuilds exactly as submitted
FIG_SRC = os.path.join(ROOT, 'REVISION_R1', '03_new_figures', 'supplementary')
D_FIG = os.path.join(REPO, 'REVISION_R1', '03_new_figures', 'supplementary')
os.makedirs(D_FIG, exist_ok=True)
cp(FIG_SRC, ['FigureS1_capacity_fade_three_knees.png',
             'FigureS2_physics_param_distributions.png'], D_FIG,
   'REVISION_R1/03_new_figures/supplementary')

# .gitignore
open(os.path.join(REPO, '.gitignore'), 'w', encoding='utf-8').write(
    "__pycache__/\n*.pyc\n*.pkl\nfigures_new/\n.ipynb_checkpoints/\n.DS_Store\n")
print("  .gitignore: da tao")
print(f"\nXong. Repo: {REPO}")
