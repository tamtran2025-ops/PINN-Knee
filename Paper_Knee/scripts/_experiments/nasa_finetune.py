"""
Few-shot cross-chemistry transfer via fine-tuning.

Approach:
  1. Train base PINN_Knee on Severson (LFP).
  2. For each NASA cell (leave-one-out):
     a. Load base weights.
     b. FREEZE the physics head (preserves physics prior).
     c. Fine-tune ONLY the NN correction head on 1 target cell
        for 100 epochs at lr=1e-4.
     d. Evaluate on the remaining 2 NASA cells.

Baselines for comparison:
  - Pure_NN (full fine-tune): continue training on target cell.
  - XGBoost (retrain): fit XGBoost on Severson + 1 NASA cell.

This experiment demonstrates that the Residual Physics architecture
supports cheap few-shot cross-chemistry transfer.
"""
import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from severson_only import load_severson_only
from features import build_feature_matrix
from models import create_model
from train import train_pinn_knee
from run_experiments import normalize_features, train_nn_model
from config import DEVICE, MAX_CYCLE_LIFE, PHYSICS_LAMBDA

BASE = os.path.dirname(SCRIPTS_DIR)
RES = os.path.join(BASE, "results")

print(f"Using device: {DEVICE}")

N_EARLY = 50          # NASA cells too short for larger n_early
FINETUNE_EPOCHS = 100
FINETUNE_LR = 1e-4
SEEDS = [42, 123, 2024]


def load_nasa_cells_with_80pct_knee():
    """Load NASA cells, assign knee = 80%-capacity cycle."""
    import importlib.util

    p7_scripts = os.path.normpath(os.path.join(
        os.path.dirname(BASE), "scripts"))
    saved_config = sys.modules.get("config")
    saved_path = list(sys.path)
    try:
        if p7_scripts not in sys.path:
            sys.path.insert(0, p7_scripts)
        cfg_path = os.path.join(p7_scripts, "config.py")
        spec_cfg = importlib.util.spec_from_file_location("paper7_config", cfg_path)
        p7_config = importlib.util.module_from_spec(spec_cfg)
        spec_cfg.loader.exec_module(p7_config)
        sys.modules["config"] = p7_config
        dl_path = os.path.join(p7_scripts, "data_loader.py")
        spec_dl = importlib.util.spec_from_file_location("paper7_data_loader", dl_path)
        p7_dl = importlib.util.module_from_spec(spec_dl)
        spec_dl.loader.exec_module(p7_dl)
    finally:
        if saved_config is not None:
            sys.modules["config"] = saved_config
        elif "config" in sys.modules:
            del sys.modules["config"]
        sys.path[:] = saved_path

    nasa_cells = p7_dl.load_nasa_cells()
    valid = []
    for cell in nasa_cells:
        cap = np.asarray(cell["capacity"])
        cyc = np.asarray(cell["cycles"])
        Q0 = cap[0]
        below = np.where(cap < 0.80 * Q0)[0]
        if len(below) == 0:
            continue
        knee = int(cyc[below[0]])
        if knee < 20:
            continue
        cell["knee_cycle"] = knee
        print(f"  {cell['name']}: knee at cycle {knee} "
              f"(total {len(cap)}, Q0={Q0:.3f})")
        valid.append(cell)
    return valid


def finetune_pinn_head(base_model, X_fit, y_fit, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    """Freeze physics head, fine-tune only nn_head on (X_fit, y_fit).

    y_fit is raw knee cycle; internally uses log target.
    """
    model = copy.deepcopy(base_model)

    # Freeze physics head
    for p in model.physics_head.parameters():
        p.requires_grad = False

    # Only nn_head params are trainable
    trainable = [p for p in model.nn_head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)

    X = torch.tensor(X_fit, dtype=torch.float32).to(DEVICE)
    y_log = torch.tensor(np.log1p(y_fit), dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        # compute_loss returns total loss; we want the data loss only but
        # since physics head is frozen, physics losses don't help training.
        # Use the same compute_loss for simplicity.
        loss = model.compute_loss(X, y_log, knee_max=MAX_CYCLE_LIFE,
                                  physics_lambda=PHYSICS_LAMBDA)
        loss.backward()
        opt.step()

    # Unfreeze for later use
    for p in model.physics_head.parameters():
        p.requires_grad = True
    return model


def finetune_pure_nn(base_model, X_fit, y_fit, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR):
    """Continue training Pure_NN on (X_fit, y_fit)."""
    model = copy.deepcopy(base_model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    X = torch.tensor(X_fit, dtype=torch.float32).to(DEVICE)
    y_log = torch.tensor(np.log1p(y_fit), dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X).squeeze()
        loss = torch.nn.functional.mse_loss(pred, y_log)
        loss.backward()
        opt.step()
    return model


def main():
    print("=" * 72)
    print("  FEW-SHOT CROSS-CHEMISTRY TRANSFER via FINE-TUNING")
    print("  Severson (LFP) base -> NASA (LiCoO2) fine-tune")
    print("=" * 72)

    print("\n[1/4] Loading Severson cells...")
    severson = load_severson_only()
    print(f"  {len(severson)} cells")

    print("\n[2/4] Loading NASA cells...")
    nasa = load_nasa_cells_with_80pct_knee()
    if len(nasa) < 2:
        print("  Need >=2 NASA cells. Aborting.")
        return
    print(f"  {len(nasa)} cells with valid 80%-SOH knee")

    # ---- Build Severson feature matrix ----
    print(f"\n[3/4] Building features at n_early={N_EARLY}...")
    X_sev, y_sev, _, _ = build_feature_matrix(severson, N_EARLY)
    X_nasa, y_nasa, _, _ = build_feature_matrix(nasa, N_EARLY)
    print(f"  Severson: {X_sev.shape}, NASA: {X_nasa.shape}")

    if len(X_nasa) < 2:
        print("  Too few NASA cells at this n_early. Aborting.")
        return

    # Internal 80/20 train/cal split of Severson
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_sev))
    n_cal = max(1, int(0.2 * len(X_sev)))
    cal_idx = perm[:n_cal]
    tr_idx = perm[n_cal:]
    X_sev_cal = X_sev[cal_idx]; y_sev_cal = y_sev[cal_idx]
    X_sev_tr = X_sev[tr_idx]; y_sev_tr = y_sev[tr_idx]

    # Normalize based on Severson train stats
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_sev_tr)
    X_sev_tr_n = scaler.transform(X_sev_tr)
    X_sev_cal_n = scaler.transform(X_sev_cal)
    X_nasa_n = scaler.transform(X_nasa)
    n_features = X_sev_tr_n.shape[1]

    # ---- Train base models once on Severson ----
    print("\n[4/4] Training base models on Severson...")
    t0 = time.time()

    # Base PINN_Knee
    print("  Training PINN_Knee on Severson...")
    np.random.seed(42); torch.manual_seed(42)
    base_pinn = create_model("PINN_Knee", n_features=n_features, device=DEVICE)
    base_pinn, _ = train_pinn_knee(
        base_pinn, X_sev_tr_n, y_sev_tr, n_early=N_EARLY,
        X_val=X_sev_cal_n, y_val=y_sev_cal, verbose=False,
    )
    base_pinn.eval()
    print(f"    done ({(time.time()-t0)/60:.1f} min)")

    # Base Pure_NN
    print("  Training Pure_NN on Severson...")
    np.random.seed(42); torch.manual_seed(42)
    base_purenn = create_model("Pure_NN", n_features=n_features, device=DEVICE)
    base_purenn, _ = train_nn_model(
        base_purenn, X_sev_tr_n, np.log1p(y_sev_tr),
        X_val=X_sev_cal_n, y_val=np.log1p(y_sev_cal),
        verbose=False,
    )
    base_purenn.eval()

    # ---- Leave-one-out across NASA cells ----
    print("\n[5/5] Leave-one-out fine-tune across NASA cells...")
    rows = []

    nasa_names = [c["name"] for c in nasa]
    n_nasa = len(X_nasa_n)

    for fit_idx in range(n_nasa):
        te_idx = [i for i in range(n_nasa) if i != fit_idx]
        X_fit = X_nasa_n[fit_idx:fit_idx + 1]
        y_fit = y_nasa[fit_idx:fit_idx + 1]
        X_te = X_nasa_n[te_idx]
        y_te = y_nasa[te_idx]

        print(f"\n  Fold {fit_idx}: fine-tune on {nasa_names[fit_idx]}, "
              f"test on {[nasa_names[i] for i in te_idx]}")

        for seed in SEEDS:
            # --- PINN_Knee: freeze physics, fine-tune NN head ---
            np.random.seed(seed); torch.manual_seed(seed)
            pinn_ft = finetune_pinn_head(base_pinn, X_fit, y_fit)
            pinn_ft.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                y_pred_pinn = pinn_ft.predict_raw(X_t).cpu().numpy()
            mae_pinn = float(np.mean(np.abs(y_pred_pinn - y_te)))

            # --- Pure_NN: full fine-tune ---
            np.random.seed(seed); torch.manual_seed(seed)
            nn_ft = finetune_pure_nn(base_purenn, X_fit, y_fit)
            nn_ft.eval()
            with torch.no_grad():
                y_pred_nn = nn_ft(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy().squeeze()
                if y_pred_nn.ndim == 0:
                    y_pred_nn = np.array([y_pred_nn])
                y_pred_nn = np.expm1(y_pred_nn)
            mae_nn = float(np.mean(np.abs(y_pred_nn - y_te)))

            # --- PINN NO fine-tune (zero-shot) for comparison ---
            with torch.no_grad():
                y_pred_zs = base_pinn.predict_raw(
                    torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                ).cpu().numpy()
            mae_pinn_zs = float(np.mean(np.abs(y_pred_zs - y_te)))

            # --- Pure_NN NO fine-tune (zero-shot) ---
            with torch.no_grad():
                y_pred_nn_zs = base_purenn(
                    torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
                ).cpu().numpy().squeeze()
                if y_pred_nn_zs.ndim == 0:
                    y_pred_nn_zs = np.array([y_pred_nn_zs])
                y_pred_nn_zs = np.expm1(y_pred_nn_zs)
            mae_nn_zs = float(np.mean(np.abs(y_pred_nn_zs - y_te)))

            # --- XGBoost: retrain on Severson + 1 NASA cell ---
            xgb = create_model("XGBoost")
            X_combo = np.vstack([X_sev_tr_n, X_fit])
            y_combo = np.concatenate([y_sev_tr, y_fit])
            xgb.fit(X_combo, np.log1p(y_combo))
            y_pred_xgb = np.expm1(xgb.predict(X_te))
            mae_xgb = float(np.mean(np.abs(y_pred_xgb - y_te)))

            print(f"    seed={seed}: PINN-FT={mae_pinn:.1f} "
                  f"(zs={mae_pinn_zs:.1f}), Pure_NN-FT={mae_nn:.1f} "
                  f"(zs={mae_nn_zs:.1f}), XGB-retrain={mae_xgb:.1f}")

            rows.append({
                "fold": fit_idx, "fit_cell": nasa_names[fit_idx], "seed": seed,
                "PINN_Knee_finetune": mae_pinn,
                "PINN_Knee_zeroshot": mae_pinn_zs,
                "Pure_NN_finetune": mae_nn,
                "Pure_NN_zeroshot": mae_nn_zs,
                "XGBoost_retrain": mae_xgb,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RES, "nasa_finetune.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY (MAE across all folds x seeds)")
    print("=" * 72)
    summary = {}
    for col in ["PINN_Knee_zeroshot", "PINN_Knee_finetune",
                "Pure_NN_zeroshot", "Pure_NN_finetune",
                "XGBoost_retrain"]:
        vals = df[col].values
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        summary[col] = {"mean": mean, "std": std, "n": int(len(vals))}
        print(f"  {col:<24s} {mean:>7.1f} +/- {std:>5.1f}  (n={len(vals)})")

    with open(os.path.join(RES, "nasa_finetune_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Key finding
    pinn_improv = summary["PINN_Knee_zeroshot"]["mean"] - summary["PINN_Knee_finetune"]["mean"]
    nn_improv = summary["Pure_NN_zeroshot"]["mean"] - summary["Pure_NN_finetune"]["mean"]
    print(f"\n  Fine-tune improvement:")
    print(f"    PINN_Knee:  -{pinn_improv:.1f} MAE  ({pinn_improv/summary['PINN_Knee_zeroshot']['mean']*100:.1f}%)")
    print(f"    Pure_NN:    -{nn_improv:.1f} MAE  ({nn_improv/summary['Pure_NN_zeroshot']['mean']*100:.1f}%)")

    print(f"\nRuntime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
