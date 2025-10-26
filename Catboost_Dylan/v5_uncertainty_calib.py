
"""
v5_uncertainty_calib.py
Small ensemble for mean/std + tail calibration via isotonic regression.

Outputs: ./outputs_v5/uncertainty/summary.json, scoreboard.csv
Also saves per-seed OOF to inspect dispersion.

bestTest = 0.1164455444
bestIteration = 1829

Shrink model to first 1830 iterations.
[V5 Uncertainty] Mean ensemble CV: 0.121016 | Calibrated: 0.117083 | Avg single-seed: 0.122682

Benefits:
 - Public/private split stability.
 - Quantifies uncertainty; better tail behavior with calibration.
Drawbacks:
 - Multiple fits increase compute.
 - Calibration must avoid leakage (done on OOF only).
 
"""

import os, json
import numpy as np
import pandas as pd

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle, top_quantile_rmsle, calibrate_tail_isotonic
)

SEED = 19981124

def vary_params(seed, jitter):
    p = catboost_default_params()
    p["random_seed"] = seed
    p["rsm"] = max(0.6, min(0.95, p["rsm"] + jitter))
    return p

def main():
    outdir = "./outputs_v5/uncertainty"
    ensure_dir(outdir)

    df = load_train("train.csv")
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)

    seeds = [SEED + i*101 for i in range(7)]
    oof_mat = []
    cv_list = []

    for i, s in enumerate(seeds):
        jitter = (i - 3) * 0.03
        params = vary_params(s, jitter)
        oof_i, _, meta = run_catboost_cv(X, y, cat_cols=cat_cols, use_group_kfold=True, groups=None, params=params, label=f"seed_{s}")
        oof_mat.append(oof_i)
        cv_list.append(meta["summary"]["cv_rmsle"])

    OOF = np.vstack(oof_mat)  # (n_seeds, n_samples)
    mean_pred = OOF.mean(axis=0)
    std_pred = OOF.std(axis=0)

    base_cv = float(np.mean(cv_list))
    cv_rmsle = rmsle(y.values, mean_pred)

    # Tail calibration (top 20%)
    cal_pred = calibrate_tail_isotonic(y.values, mean_pred, q=0.8)
    cv_rmsle_cal = rmsle(y.values, cal_pred)

    # Save
    summary = {
        "ensemble_mean_cv": float(cv_rmsle),
        "ensemble_mean_calibrated_cv": float(cv_rmsle_cal),
        "avg_single_seed_cv": float(base_cv),
        "delta_vs_avg_single": float(base_cv - cv_rmsle),
        "delta_with_calibration": float(base_cv - cv_rmsle_cal),
        "top20_rmsle_before": float(top_quantile_rmsle(y.values, mean_pred, q=0.8)),
        "top20_rmsle_after": float(top_quantile_rmsle(y.values, cal_pred, q=0.8)),
        "notes": "7-seed bagging + isotonic tail calibration."
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame([scoreboard_row("v5_uncertainty_calib", cv_rmsle_cal, "Bagging improves stability; tail calibration fixes luxury bias.")]).to_csv(
        os.path.join(outdir, "scoreboard.csv"), index=False
    )

    # Diagnostics
    pd.DataFrame({"oof_mean": mean_pred, "oof_std": std_pred, "y": y.values}).to_csv(os.path.join(outdir, "oof_mean_std.csv"), index=False)

    print(f"[V5 Uncertainty] Mean ensemble CV: {cv_rmsle:.6f} | Calibrated: {cv_rmsle_cal:.6f} | Avg single-seed: {base_cv:.6f}")
    print("\nBenefits:\n - Public/private split stability.\n - Quantifies uncertainty; better tail behavior with calibration.")
    print("Drawbacks:\n - Multiple fits increase compute.\n - Calibration must avoid leakage (done on OOF only).")

if __name__ == "__main__":
    main()
