
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
from typing import Tuple

from catboost import Pool
from sklearn.isotonic import IsotonicRegression

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle, top_quantile_rmsle
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_data_path(filename: str) -> str:
    """Search for filename near the script or in the shared data directory."""
    candidates = [
        os.path.join(SCRIPT_DIR, filename),
        os.path.join(SCRIPT_DIR, "..", filename),
        os.path.join(SCRIPT_DIR, "..", "data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find '{filename}' in {candidates}")

def build_neighborhood_te_map(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, float]:
    """Return a log-target mean per neighborhood plus global fallback."""
    if "Neighborhood" not in X.columns:
        y_log = np.log1p(y)
        return pd.Series(dtype=float), float(y_log.mean())
    y_log = np.log1p(y)
    mapping = y_log.groupby(X["Neighborhood"]).mean()
    return mapping, float(y_log.mean())

def apply_neighborhood_te(test_X: pd.DataFrame, mapping: pd.Series, default: float) -> pd.DataFrame:
    if "Neighborhood" in test_X.columns:
        test_X["Neighborhood_TE"] = test_X["Neighborhood"].map(mapping).fillna(default)
    return test_X

def fit_tail_calibration(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.8):
    thresh = np.quantile(y_pred, q)
    mask = y_pred >= thresh
    iso = IsotonicRegression(out_of_bounds="clip")
    y_adjusted = y_pred.copy()
    if mask.any():
        iso.fit(y_pred[mask], y_true[mask])
        y_adjusted[mask] = iso.transform(y_pred[mask])
    return y_adjusted, iso, thresh

def calibrate_test_predictions(mean_pred: np.ndarray, iso: IsotonicRegression, thresh: float) -> np.ndarray:
    mask = mean_pred >= thresh
    adjusted = mean_pred.copy()
    if mask.any():
        adjusted[mask] = iso.transform(mean_pred[mask])
    return adjusted

SEED = 19981124

def vary_params(seed, jitter):
    p = catboost_default_params()
    p["random_seed"] = seed
    p["rsm"] = max(0.6, min(0.95, p["rsm"] + jitter))
    return p

def main():
    outdir = "./outputs_v5/uncertainty"
    ensure_dir(outdir)

    train_path = find_data_path("train.csv")
    df = load_train(train_path)
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)
    te_map, te_default = build_neighborhood_te_map(X, y)

    seeds = [SEED + i * 101 for i in range(7)]
    oof_mat = []
    cv_list = []
    models_by_seed = []

    for i, s in enumerate(seeds):
        jitter = (i - 3) * 0.03
        params = vary_params(s, jitter)
        oof_i, _, meta = run_catboost_cv(
            X, y, cat_cols=cat_cols, use_group_kfold=True, groups=None, params=params, label=f"seed_{s}"
        )
        oof_mat.append(oof_i)
        cv_list.append(meta["summary"]["cv_rmsle"])
        models_by_seed.append(meta["models"])

    OOF = np.vstack(oof_mat)
    mean_pred = OOF.mean(axis=0)
    std_pred = OOF.std(axis=0)

    base_cv = float(np.mean(cv_list))
    cv_rmsle = rmsle(y.values, mean_pred)

    cal_pred, tail_iso, tail_thresh = fit_tail_calibration(y.values, mean_pred, q=0.8)
    cv_rmsle_cal = rmsle(y.values, cal_pred)

    summary = {
        "ensemble_mean_cv": float(cv_rmsle),
        "ensemble_mean_calibrated_cv": float(cv_rmsle_cal),
        "avg_single_seed_cv": float(base_cv),
        "delta_vs_avg_single": float(base_cv - cv_rmsle),
        "delta_with_calibration": float(base_cv - cv_rmsle_cal),
        "top20_rmsle_before": float(top_quantile_rmsle(y.values, mean_pred, q=0.8)),
        "top20_rmsle_after": float(top_quantile_rmsle(y.values, cal_pred, q=0.8)),
        "notes": "7-seed bagging + isotonic tail calibration.",
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame(
        [scoreboard_row("v5_uncertainty_calib", cv_rmsle_cal, "Bagging + tail isotonic for robust luxury predictions.")]
    ).to_csv(os.path.join(outdir, "scoreboard.csv"), index=False)

    pd.DataFrame({"oof_mean": mean_pred, "oof_std": std_pred, "y": y.values}).to_csv(
        os.path.join(outdir, "oof_mean_std.csv"), index=False
    )

    test_path = find_data_path("test.csv")
    test_df = pd.read_csv(test_path)
    test_ids = test_df["Id"]
    test_X, _, _, _ = feature_engineering(test_df, None)
    test_X = apply_neighborhood_te(test_X, te_map, te_default)
    test_X = test_X.reindex(columns=X.columns, fill_value=0.0)
    test_cat_idx = [test_X.columns.get_loc(c) for c in cat_cols if c in test_X.columns]
    test_pool = Pool(test_X, cat_features=test_cat_idx)

    seed_test_preds = []
    for seed_models in models_by_seed:
        fold_preds = []
        for model in seed_models:
            pred_log = model.predict(test_pool)
            fold_preds.append(np.expm1(pred_log))
        seed_test_preds.append(np.mean(fold_preds, axis=0))
    seed_test_preds = np.vstack(seed_test_preds)

    test_mean_pred = seed_test_preds.mean(axis=0)
    test_std_pred = seed_test_preds.std(axis=0)
    test_cal_pred = calibrate_test_predictions(test_mean_pred, tail_iso, tail_thresh)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_cal_pred})
    submission_path = os.path.join(outdir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            "Id": test_ids,
            "mean_pred": test_mean_pred,
            "std_pred": test_std_pred,
            "calibrated_pred": test_cal_pred,
        }
    ).to_csv(os.path.join(outdir, "test_mean_std.csv"), index=False)

    print(f"[V5 Uncertainty] Mean ensemble CV: {cv_rmsle:.6f} | Calibrated: {cv_rmsle_cal:.6f} | Avg single-seed: {base_cv:.6f}")
    print(f"Saved calibrated submission to {submission_path}")
    print("\nBenefits:\n - Public/private split stability.\n - Quantifies uncertainty; better tail behavior with calibration.")
    print("Drawbacks:\n - Multiple fits increase compute.\n - Calibration must avoid leakage (done on OOF only).")

if __name__ == "__main__":
    main()
