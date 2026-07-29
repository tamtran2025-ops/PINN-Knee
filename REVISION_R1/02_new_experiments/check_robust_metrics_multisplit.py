"""Robustness of the headline metrics across several outer splits.

Each row is written to the CSV as soon as it is computed so that progress can be
followed in real time and a crash does not lose completed work.
"""
import os
import sys
import csv
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

sys.stdout.reconfigure(encoding='utf-8')
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SC = os.path.join(ROOT, 'Paper_Knee', 'scripts')
for p in (SC, os.path.join(SC, '_experiments'), os.path.join(SC, '_analysis')):
    sys.path.insert(0, p)

from config import DEVICE, PHYSICS_LAMBDA
from features import build_feature_matrix, normalize_features
from models import create_model
from metrics import evaluate_knee_predictions
from train import train_pinn_knee
from rerun_exp1_fixed import load_paper_pool, train_nn_model_FIXED
from repeated_cv_architecture import kfold_split_all

OUT = os.path.join(HERE, 'robust_metrics_5splits.csv')
FAST_MODELS = [
    'PINN_Knee', 'RandomForest', 'PINN_UQ', 'Ensemble_NN',
    'Pure_NN', 'GaussianProcess', 'Neural_ODE', 'XGBoost'
]
SPLIT_SEEDS = [42, 100, 101, 102, 103]
EARLY_BUDGETS = [50, 100, 150]
K = 5
MODEL_SEED = 0

def run_one_model(m_name, tr, cal, te, ne, split_seed, fold):
    np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MODEL_SEED)

    Xtr, ytr, _, _ = build_feature_matrix(tr, ne)
    Xc, yc, _, _ = build_feature_matrix(cal, ne)
    Xte, yte, _, _ = build_feature_matrix(te, ne)
    if Xtr.size == 0 or Xte.size == 0:
        return None

    Xtr_n, Xte_n, Xc_n, _ = normalize_features(Xtr, Xte, Xc if Xc.size else None)
    nf = Xtr_n.shape[1]

    if m_name == 'PINN_Knee':
        m = create_model('PINN_Knee', n_features=nf, device=DEVICE)
        m, _ = train_pinn_knee(m, Xtr_n, ytr, tr, ne, physics_lambda=PHYSICS_LAMBDA,
                               X_val=Xc_n, y_val=yc if yc.size else None, use_log_target=True)
        m.eval()
        Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = m.predict_raw(Xte_t).cpu().numpy().ravel()
    elif m_name in ('PINN_UQ', 'Pure_NN', 'Neural_ODE', 'Ensemble_NN'):
        m = create_model(m_name, n_features=nf, device=DEVICE)
        m, _ = train_nn_model_FIXED(m, Xtr_n, np.log1p(ytr), Xc_n, np.log1p(yc) if yc.size else None)
        m.eval()
        Xte_t = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = np.expm1(m(Xte_t).cpu().numpy().ravel())
    elif m_name == 'RandomForest':
        m = RandomForestRegressor(n_estimators=100, random_state=MODEL_SEED)
        m.fit(Xtr_n, np.log1p(ytr))
        preds = np.expm1(m.predict(Xte_n))
    elif m_name == 'XGBoost':
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=MODEL_SEED, n_jobs=-1)
        m.fit(Xtr_n, np.log1p(ytr))
        preds = np.expm1(m.predict(Xte_n))
    elif m_name == 'GaussianProcess':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        m = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=MODEL_SEED)
        m.fit(Xtr_n, np.log1p(ytr))
        preds = np.expm1(m.predict(Xte_n))

    metrics = evaluate_knee_predictions(yte, preds)
    return {
        'split_seed': split_seed,
        'fold': fold,
        'n_early': ne,
        'model': m_name,
        'MAE': round(metrics['MAE'], 2),
        'MedianAE': round(metrics['MedianAE'], 2),
        'Within_50': round(metrics['Within_50'], 4)
    }

def main():
    cells = load_paper_pool()
    total = len(SPLIT_SEEDS) * K * len(EARLY_BUDGETS) * len(FAST_MODELS)
    print(f"Bt u kim tra 15 pht: {total} lt hun luyn ({len(SPLIT_SEEDS)} splits x {K} folds x 3 budgets x 8 models)")

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT, encoding='utf-8')):
            done.add((r['split_seed'], r['fold'], r['n_early'], r['model']))
        print(f"Resume:  c {len(done)} lt kt qu trong CSV.\n")

    t0, idx = time.time(), len(done)
    for ss in SPLIT_SEEDS:
        splits = kfold_split_all(cells, K, seed=ss)
        for fold, (tr, cal, te) in enumerate(splits):
            for ne in EARLY_BUDGETS:
                for m_name in FAST_MODELS:
                    key = (str(ss), str(fold), str(ne), m_name)
                    if key in done:
                        continue
                    res = run_one_model(m_name, tr, cal, te, ne, ss, fold)
                    if res:
                        new_file = not os.path.exists(OUT)
                        with open(OUT, 'a', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=list(res.keys()))
                            if new_file:
                                w.writeheader()
                            w.writerow(res)
                        idx += 1
                        if idx % 10 == 0:
                            print(f"  [{idx:>3d}/{total}] ss={ss} f={fold} ne={ne:<3d} m={m_name:<15s} MAE={res['MAE']:>6.2f} MedianAE={res['MedianAE']:>6.2f}")

    print(f"\nXong sau {(time.time()-t0)/60:.1f} pht -> {OUT}")

if __name__ == '__main__':
    main()
