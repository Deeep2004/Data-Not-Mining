
"""
v5_stack.py
Stacking 3 experts with a linear meta-learner:

A) CatBoost (v3-lite params)
B) Linear/Lasso on numeric-only
C) Leaf-index regressor: use CatBoost's leaf indices as features -> Ridge

Outputs: ./outputs_v5/stack/summary.json, scoreboard.csv

bestTest = 0.1287482933
bestIteration = 1019

Benefits:
 - Adaptive weighting by regime.
 - Uses leaf structure + trend model.
Drawbacks:
 - Careful OOF bookkeeping needed.
 - Slightly slower; more moving parts.

"""

import os, json
import numpy as np
import pandas as pd

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle
)

from sklearn.linear_model import LassoCV, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

SEED = 19981124

def get_leaf_features(models, X, cat_cols):
    """Return stacked leaf indices (one model per fold) concatenated.
    Uses a CatBoost Pool with the same categorical feature indices for safety.
    """
    from catboost import Pool as CBPool
    import numpy as np
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    pool = CBPool(X, cat_features=cat_idx if len(cat_idx) > 0 else None)
    all_leaf = []
    for m in models:
        leaf = m.calc_leaf_indexes(pool)
        all_leaf.append(leaf)
    LF = np.concatenate(all_leaf, axis=1)
    return LF

def main():
    outdir = "./outputs_v5/stack"
    ensure_dir(outdir)

    df = load_train("train.csv")
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)

    # Expert A: CatBoost
    oofA, foldsA, metaA = run_catboost_cv(X, y, cat_cols=cat_cols, use_group_kfold=True, groups=df.get("Neighborhood"), params=catboost_default_params(), label="catboost_A")
    modelsA = metaA["models"]
    base_cv = metaA["summary"]["cv_rmsle"]

    # Expert B: Linear/Lasso on numeric-only (simple trend)
    X_num = X[num_cols].copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oofB = np.zeros(len(y), dtype=float)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    for tr_idx, va_idx in kf.split(X_num):
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lasso", LassoCV(n_alphas=100, cv=5, random_state=SEED, max_iter=10000))
        ])
        pipe.fit(X_num.iloc[tr_idx], np.log1p(y.iloc[tr_idx]))
        pred_log = pipe.predict(X_num.iloc[va_idx])
        oofB[va_idx] = np.expm1(pred_log)

    # Expert C: Leaf-index features -> Ridge
    LF = get_leaf_features(modelsA, X, cat_cols)
    # One-hot encode leaves via hashing by converting to strings (to limit dimensionality we skip OHE and use Ridge on raw indices)
    oofC = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in kf.split(LF):
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(LF[tr_idx], np.log1p(y.iloc[tr_idx]))
        pred_log = ridge.predict(LF[va_idx])
        oofC[va_idx] = np.expm1(pred_log)

    # Meta learner on OOF predictions (+ a couple meta features)
    meta_df = pd.DataFrame({
        "A_cat": oofA,
        "B_lin": oofB,
        "C_leaf": oofC,
        "log_TotalSF": np.log1p(X["TotalSF"].values),
        "OverallQual": X["OverallQual"].values if "OverallQual" in X.columns else 0,
    })
    oof_meta = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in kf.split(meta_df):
        meta = Ridge(alpha=0.1, random_state=SEED)
        meta.fit(meta_df.iloc[tr_idx], np.log1p(y.iloc[tr_idx]))
        pred_log = meta.predict(meta_df.iloc[va_idx])
        oof_meta[va_idx] = np.expm1(pred_log)

    cv_rmsle = rmsle(y.values, oof_meta)

    summary = {
        "base_cv_rmsle": float(base_cv),
        "final_cv_rmsle": float(cv_rmsle),
        "delta": float(base_cv - cv_rmsle),
        "notes": "Stacked A/B/C experts with Ridge meta on OOF."
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame([scoreboard_row("v5_stack", cv_rmsle, "Learns when to trust each expert.")]).to_csv(
        os.path.join(outdir, "scoreboard.csv"), index=False
    )

    print(f"[V5 Stack] Base CV RMSLE: {base_cv:.6f} -> Final: {cv_rmsle:.6f} (Δ={base_cv-cv_rmsle:+.6f})")
    print("\nBenefits:\n - Adaptive weighting by regime.\n - Uses leaf structure + trend model.")
    print("Drawbacks:\n - Careful OOF bookkeeping needed.\n - Slightly slower; more moving parts.")

if __name__ == "__main__":
    main()
