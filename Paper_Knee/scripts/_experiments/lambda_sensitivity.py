"""
Lambda_phys sensitivity study for PINN_Knee.

Runs PINN_Knee with different physics loss weights to show that the
main result is not cherry-picked to a single hyperparameter choice.

We sweep the physics loss multiplier lambda_phys in {0.01, 0.05, 0.1, 0.2}
at n_early=100. All 5 physics loss components are scaled together.

Outputs:
    results/lambda_sensitivity.csv
    results/lambda_sensitivity_summary.json
"""
import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from severson_only import load_severson_only
from data_loader import get_train_cal_test_split
from features import build_feature_matrix
from models import create_model
from train import train_pinn_knee
from metrics import evaluate_knee_predictions
from run_experiments import normalize_features
from config import PHYSICS_LAMBDA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, "results")

N_EARLY = 100
SEEDS = [42, 123, 2024]
# Lambda_phys sweep  -  we scale ALL 5 physics components uniformly
# by multiplying each of the weights in PHYSICS_LAMBDA. The multiplier
# represents the physics loss weight scale; default=1.0.
SCALES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]


def scale_physics_lambda(base, scale):
    out = copy.deepcopy(base)
    # Scale the 5 physics loss keys (not the correction_penalty)
    phys_keys = ['monotonic_decay', 'sei_sqrt_t', 'knee_transition',
                 'degradation_ode', 'initial_condition']
    for k in phys_keys:
        if k in out:
            out[k] = out[k] * scale
    return out


def main():
    print("[1/2] Loading cells...")
    cells = load_severson_only()
    train, cal, test = get_train_cal_test_split(
        cells, train_frac=0.6, cal_frac=0.2, seed=42,
    )
    print(f"  train={len(train)}, cal={len(cal)}, test={len(test)}")

    X_tr, y_tr, _, _ = build_feature_matrix(train, N_EARLY)
    X_cal, y_cal, _, _ = build_feature_matrix(cal, N_EARLY)
    X_te, y_te, _, _ = build_feature_matrix(test, N_EARLY)

    X_tr_n, X_te_n, X_cal_n, _ = normalize_features(X_tr, X_te, X_cal)
    n_features = X_tr_n.shape[1]

    rows = []
    t0 = time.time()
    total = len(SCALES) * len(SEEDS)
    idx = 0

    print(f"\n[2/2] Running {total} experiments...")
    for scale in SCALES:
        phys_lambda = scale_physics_lambda(PHYSICS_LAMBDA, scale)
        for seed in SEEDS:
            idx += 1
            np.random.seed(seed)
            torch.manual_seed(seed)

            try:
                model = create_model("PINN_Knee", n_features=n_features,
                                     device="cpu")
                model, _ = train_pinn_knee(
                    model, X_tr_n, y_tr,
                    X_val=X_cal_n, y_val=y_cal,
                    n_early=N_EARLY, physics_lambda=phys_lambda,
                    verbose=False,
                )
                model.eval()
                with torch.no_grad():
                    X_te_t = torch.tensor(X_te_n, dtype=torch.float32)
                    y_pred_log = model(X_te_t).numpy().squeeze()
                    y_pred = np.expm1(y_pred_log)

                metrics = evaluate_knee_predictions(y_te, y_pred)
                rows.append({
                    "scale": scale, "seed": seed,
                    "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
                    "MedianAE": metrics["MedianAE"],
                    "status": "ok",
                })
                print(f"  [{idx}/{total}] scale={scale:<4} seed={seed}: "
                      f"MAE={metrics['MAE']:.1f}  "
                      f"elapsed {(time.time()-t0)/60:.1f}min")
            except Exception as e:
                print(f"  [{idx}/{total}] scale={scale} seed={seed}: FAIL ({e})")
                rows.append({
                    "scale": scale, "seed": seed,
                    "MAE": None, "status": "error",
                })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, "lambda_sensitivity.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Summary
    summary = {}
    ok = df[df["status"] == "ok"]
    for scale in SCALES:
        sub = ok[ok["scale"] == scale]
        if len(sub) == 0:
            continue
        summary[scale] = {
            "MAE_mean": float(sub["MAE"].mean()),
            "MAE_std": float(sub["MAE"].std()),
            "n_seeds": int(len(sub)),
        }
    with open(os.path.join(RESULT_DIR, "lambda_sensitivity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  LAMBDA_PHYS SENSITIVITY")
    print("=" * 60)
    print(f"  {'scale':>8s}  {'MAE':>14s}")
    print(f"  {'-'*8}  {'-'*14}")
    for scale in SCALES:
        if scale not in summary:
            continue
        s = summary[scale]
        note = "  (physics OFF)" if scale == 0.0 else ""
        print(f"  {scale:>8.2f}  {s['MAE_mean']:>6.1f}+/-{s['MAE_std']:>4.1f}{note}")
    print(f"\nRuntime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
