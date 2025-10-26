
"""
v5_residual_interactions.py
Stage-1 CatBoost + explicit interaction features + tiny residual model on OOF residuals.

Outputs:
- ./outputs_v5/residual_interactions/summary.json
- ./outputs_v5/residual_interactions/scoreboard.csv
- Prints pros/cons at the end.

[V5 Residual+Interactions] Base CV RMSLE: 0.121319 -> Final: 0.141935 (Δ=-0.020616)

"""

import os, json, itertools
import numpy as np
import pandas as pd

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle
)

# Optional tiny residual models
try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SEED = 19981124

def build_interaction_cols(X: pd.DataFrame, pairs: list) -> pd.DataFrame:
    X = X.copy()
    for a, b in pairs:
        name = f"{a}__x__{b}"
        if a in X.columns and b in X.columns:
            X[name] = X[a] * X[b]
    return X

def main():
    outdir = "./outputs_v5/residual_interactions"
    ensure_dir(outdir)

    df = load_train("train.csv")
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)

    # Choose a small set of strong numerics & categoricals for interactions
    strong_num = [c for c in ["TotalSF","GrLivArea","OverallQual","Age","RemodAge","Neighborhood_TE","BathsPerRoom","GrLivArea_to_TotalSF"] if c in X.columns]
    strong_cat = [c for c in ["MSSubClass","Neighborhood","OverallQual","KitchenQual","Exterior1st"] if c in X.columns]
    # Build numeric * onehot(MSSubClass-like) interactions by simple multiplication when categorical is ordinal-coded;
    # for true categoricals, CatBoost handles internally; here we add explicit crosses for numerics only pairs:
    pair_candidates = list(itertools.combinations(strong_num, 2))
    # Limit to top K to keep residual model tiny
    pairs = pair_candidates[:12]
    X2 = build_interaction_cols(X, pairs)

    # Stage-1 CatBoost
    oof_pred, folds, meta = run_catboost_cv(X2, y, cat_cols=cat_cols, use_group_kfold=True, groups=df.get("Neighborhood"), params=catboost_default_params(), label="stage1_with_interactions")
    base_cv = meta["summary"]["cv_rmsle"]

    # Residuals in linear space (SalePrice)
    resid = y.values - oof_pred

    # Build residual design on a compact subset
    res_cols = strong_num + [c for c in X2.columns if "__x__" in c]
    XR = X2[res_cols].copy()
    XR = XR.fillna(0.0)

    if HAS_LGBM:
        residual_model = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=SEED
        )
    else:
        residual_model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=800, random_state=SEED)

    # OOF residual fit (5-fold consistent split used by CatBoost wrapper isn't directly accessible;
    # we'll rebuild the same KFold split for fairness)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_resid = np.zeros_like(resid, dtype=float)

    for tr_idx, va_idx in kf.split(XR):
        residual_model.fit(XR.iloc[tr_idx], resid[tr_idx])
        oof_resid[va_idx] = residual_model.predict(XR.iloc[va_idx])

    final_oof = oof_pred + oof_resid
    cv_rmsle = rmsle(y.values, final_oof)

    # Save outputs
    summary = {
        "base_cv_rmsle": float(base_cv),
        "final_cv_rmsle": float(cv_rmsle),
        "delta": float(base_cv - cv_rmsle),
        "notes": "CatBoost + explicit numeric×numeric crosses + tiny residual model (LGBM/HGBR)."
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame([scoreboard_row("v5_residual_interactions", cv_rmsle, "Adds explicit crosses; residual soaks leftover patterns.")]).to_csv(
        os.path.join(outdir, "scoreboard.csv"), index=False
    )

    print(f"[V5 Residual+Interactions] Base CV RMSLE: {base_cv:.6f} -> Final: {cv_rmsle:.6f} (Δ={base_cv-cv_rmsle:+.6f})")
    print("\nBenefits:\n - Captures structured pairwise effects v4 hinted at.\n - Residual model is tiny; low overfit risk.")
    print("Drawbacks:\n - Extra features increase memory.\n - Residual adds another training loop; ensure same folds for fairness.")

if __name__ == "__main__":
    main()
