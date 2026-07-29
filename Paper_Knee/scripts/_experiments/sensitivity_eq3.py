"""
Sensitivity analysis for Eq. (3) log-knee formula coefficients.

Eq. (3): log n_knee = alpha * log(b) + beta * log(d) + gamma * c + delta
with defaults (alpha, beta, gamma, delta) = (-0.8, -0.3, 1.0, -0.4).

We perturb each coefficient by +/-20% while holding the others fixed,
yielding 9 configs (baseline + 4 coeffs x 2 directions). Each config is
evaluated on 5-fold CV x 3 seeds x n_early=100 = 15 runs.

Outputs:
    results/sensitivity_eq3.csv
    results/sensitivity_eq3_summary.json
    tables/sensitivity_eq3.tex
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from severson_only import load_severson_only
from run_experiments import _kfold_split
from features import build_feature_matrix, normalize_features
from models import create_model
from train import train_pinn_knee
from metrics import evaluate_knee_predictions
from config import DEVICE, PHYSICS_LAMBDA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, "results")
TABLE_DIR = os.path.join(BASE, "tables")

N_EARLY = 100
SEEDS = [0, 1, 2]
N_SPLITS = 5

DEFAULTS = {
    "alpha": -0.8,
    "beta": -0.3,
    "gamma": 1.0,
    "delta": -0.4,
}
PERTURBATION = 0.20  # +/-20%


def build_configs():
    """Return list of (config_name, dict of 4 coeffs)."""
    configs = [("baseline", dict(DEFAULTS))]
    for key, val in DEFAULTS.items():
        for sign, lbl in [(+1, "+20pct"), (-1, "-20pct")]:
            d = dict(DEFAULTS)
            d[key] = val * (1.0 + sign * PERTURBATION)
            configs.append((f"{key}_{lbl}", d))
    return configs


def run_one(coeffs, n_early, seed, train, cal, test):
    """One training run with custom Eq. (3) coefficients."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train, y_train, _, _ = build_feature_matrix(train, n_early)
    X_cal, y_cal, _, _ = build_feature_matrix(cal, n_early)
    X_test, y_test, _, _ = build_feature_matrix(test, n_early)

    if X_train.size == 0 or X_test.size == 0:
        return {"status": "skipped_no_data"}

    X_train_n, X_test_n, X_cal_n, _ = normalize_features(
        X_train, X_test, X_cal if X_cal.size > 0 else None,
    )
    n_features = X_train_n.shape[1]

    model = create_model("PINN_Knee", n_features=n_features, device=DEVICE)
    # Inject custom Eq. (3) coefficients
    model.eq3_alpha = coeffs["alpha"]
    model.eq3_beta = coeffs["beta"]
    model.eq3_gamma = coeffs["gamma"]
    model.eq3_delta = coeffs["delta"]

    model, _ = train_pinn_knee(
        model, X_train_n, y_train, train, n_early,
        physics_lambda=dict(PHYSICS_LAMBDA),
        X_val=X_cal_n, y_val=y_cal if y_cal.size > 0 else None,
        use_log_target=True,
    )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_n, dtype=torch.float32).to(DEVICE)
        # FIX(H7): ravel to (N,); otherwise evaluate_knee_predictions
        # broadcasts (N,1) against y_test (N,) and averages MAE over all pairs.
        y_pred_log = model(X_test_t).cpu().numpy().ravel()
        y_pred = np.expm1(y_pred_log)

    metrics = evaluate_knee_predictions(y_test, y_pred)
    out = {"status": "ok", "n_train": len(y_train), "n_test": len(y_test)}
    out.update({k: round(v, 4) for k, v in metrics.items()})
    return out


def main():
    print("[1/3] Loading cells...")
    cells = load_severson_only()
    print(f"  {len(cells)} Severson cells")

    configs = build_configs()
    print(f"\n[2/3] {len(configs)} sensitivity configs:")
    for name, c in configs:
        print(f"    {name:<18s} alpha={c['alpha']:+.3f} beta={c['beta']:+.3f} "
              f"gamma={c['gamma']:+.3f} delta={c['delta']:+.3f}")

    fold_splits = _kfold_split(cells, n_folds=N_SPLITS, seed=42)
    total = len(configs) * N_SPLITS * len(SEEDS)
    print(f"\n[3/3] Running {total} runs "
          f"(9 configs x 5 folds x 3 seeds)...")

    rows = []
    t0 = time.time()
    idx = 0

    for fold, (train, cal, test) in enumerate(fold_splits):
        for name, coeffs in configs:
            for seed in SEEDS:
                idx += 1
                elapsed = time.time() - t0
                eta = elapsed / idx * (total - idx) if idx else 0

                try:
                    r = run_one(coeffs, N_EARLY, seed, train, cal, test)
                    r["config"] = name
                    r["fold"] = fold
                    r["seed"] = seed
                    r.update(coeffs)
                    rows.append(r)
                    mae = r.get("MAE")
                    mae_str = f"{mae:.1f}" if isinstance(mae, (int, float)) else str(mae)
                    print(f"  [{idx:>4d}/{total}] fold={fold} "
                          f"{name:<18s} seed={seed}: MAE={mae_str}  "
                          f"ETA {eta/60:.1f}min")
                except Exception as e:
                    print(f"  [{idx}/{total}] fold={fold} {name} seed={seed}: "
                          f"FAIL ({e})")
                    rows.append({"config": name, "fold": fold, "seed": seed,
                                 "status": "error", **coeffs})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULT_DIR, "sensitivity_eq3.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    summary = {}
    ok = df[df["status"] == "ok"].copy()
    for name, _ in configs:
        sub = ok[ok["config"] == name]
        if len(sub) == 0:
            continue
        summary[name] = {
            "MAE_mean": float(round(sub["MAE"].mean(), 1)),
            "MAE_std": float(round(sub["MAE"].std(), 1)),
            "n_runs": int(len(sub)),
        }

    with open(os.path.join(RESULT_DIR, "sensitivity_eq3_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    baseline_mae = summary.get("baseline", {}).get("MAE_mean")
    print("\n" + "=" * 78)
    print("  SENSITIVITY ANALYSIS: Eq. (3) coefficients (+/-20%)")
    print("=" * 78)
    print(f"\n  {'Config':<20s} {'MAE':>12s}  {'Delta vs base':>14s}")
    print("  " + "-" * 54)
    for name, _ in configs:
        s = summary.get(name)
        if s is None:
            continue
        diff = s["MAE_mean"] - baseline_mae if baseline_mae else 0
        diff_str = f"{diff:+.1f}" if name != "baseline" else "---"
        print(f"  {name:<20s} {s['MAE_mean']:>6.1f}+/-{s['MAE_std']:>4.1f}  "
              f"{diff_str:>14s}")

    write_table(summary, configs, baseline_mae)
    print(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")


def write_table(summary, configs, baseline_mae):
    if baseline_mae is None:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity analysis of the log-knee coefficients in "
        r"Eq.~\eqref{eq:logknee}. Each coefficient is perturbed by "
        r"$\pm 20\%$ while the others are held at their default values. "
        r"5-fold CV with 3 seeds per fold (15 runs per configuration) at "
        r"$n_{\mathrm{early}}=100$. $\Delta\mathrm{MAE}$ is the shift in "
        r"mean MAE relative to the default coefficients.}",
        r"\label{tab:sensitivity_eq3}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Config & Coefficient & Value & MAE & $\Delta$MAE \\",
        r"\midrule",
    ]
    name_map = {
        "baseline": (r"\textbf{baseline}", "---", "---"),
        "alpha_+20pct": (r"$\alpha$ $+20\%$", r"$\alpha$", "-0.96"),
        "alpha_-20pct": (r"$\alpha$ $-20\%$", r"$\alpha$", "-0.64"),
        "beta_+20pct":  (r"$\beta$ $+20\%$",  r"$\beta$",  "-0.36"),
        "beta_-20pct":  (r"$\beta$ $-20\%$",  r"$\beta$",  "-0.24"),
        "gamma_+20pct": (r"$\gamma$ $+20\%$", r"$\gamma$", "+1.20"),
        "gamma_-20pct": (r"$\gamma$ $-20\%$", r"$\gamma$", "+0.80"),
        "delta_+20pct": (r"$\delta$ $+20\%$", r"$\delta$", "-0.48"),
        "delta_-20pct": (r"$\delta$ $-20\%$", r"$\delta$", "-0.32"),
    }
    order = ["baseline",
             "alpha_+20pct", "alpha_-20pct",
             "beta_+20pct", "beta_-20pct",
             "gamma_+20pct", "gamma_-20pct",
             "delta_+20pct", "delta_-20pct"]
    for c in order:
        if c not in summary:
            continue
        s = summary[c]
        label, coef, val = name_map.get(c, (c, "", ""))
        mae_str = f"{s['MAE_mean']}$\\pm${s['MAE_std']}"
        if c == "baseline":
            delta_str = "---"
        else:
            d = s["MAE_mean"] - baseline_mae
            delta_str = f"{d:+.1f}"
            if abs(d) > 5:
                delta_str = r"\textbf{" + delta_str + "}"
        lines.append(f"{label} & {coef} & {val} & {mae_str} & {delta_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(TABLE_DIR, "sensitivity_eq3.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
