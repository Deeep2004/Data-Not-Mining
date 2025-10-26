
"""
v5_tabular_embed_fusion.py
A lightweight MLP (sklearn MLPRegressor) on one-hot major categoricals + scaled numerics.
Used as residual on top of CatBoost, then optionally as an extra expert to blend.

Outputs: ./outputs_v5/emb_fusion/summary.json, scoreboard.csv

0.122828

[V5 Embedding Fusion] Base CV RMSLE: 0.140096 -> Final: 0.139499 (Δ=+0.000598)

"""

import os, json
import numpy as np
import pandas as pd

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

SEED = 19981124

def main():
    outdir = "./outputs_v5/emb_fusion"
    ensure_dir(outdir)

    df = load_train("train.csv")
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)

    # Stage-1 CatBoost
    oof_cat, folds, meta = run_catboost_cv(X, y, cat_cols=cat_cols, use_group_kfold=True, groups=df.get("Neighborhood"), params=catboost_default_params(), label="stage1_cat")
    base_cv = meta["summary"]["cv_rmsle"]
    resid = y.values - oof_cat

    # Minimal MLP on one-hot of a subset of high-impact categoricals
    major_cats = [c for c in ["Neighborhood","MSZoning","Exterior1st","Exterior2nd","KitchenQual","HouseStyle"] if c in cat_cols]
    num_keep = [c for c in num_cols if c in ["TotalSF","GrLivArea","OverallQual","Age","RemodAge","Neighborhood_TE","BathsPerRoom","GrLivArea_to_TotalSF"]]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_keep),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), major_cats)
        ],
        remainder="drop",
    )

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, learning_rate_init=1e-3, random_state=SEED, max_iter=500, early_stopping=True, n_iter_no_change=20)

    pipe = Pipeline([("pre", pre), ("mlp", mlp)])

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_resid = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in kf.split(X):
        pipe.fit(X.iloc[tr_idx][num_keep + major_cats], resid[tr_idx])
        oof_resid[va_idx] = pipe.predict(X.iloc[va_idx][num_keep + major_cats])

    final_oof = oof_cat + oof_resid
    cv_rmsle = rmsle(y.values, final_oof)

    summary = {
        "base_cv_rmsle": float(base_cv),
        "final_cv_rmsle": float(cv_rmsle),
        "delta": float(base_cv - cv_rmsle),
        "notes": "CatBoost + MLP residual with one-hot 'embeddings' for major categoricals."
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame([scoreboard_row("v5_tabular_embed_fusion", cv_rmsle, "Neural residual complements tree splits for smooth category similarity.")]).to_csv(
        os.path.join(outdir, "scoreboard.csv"), index=False
    )

    print(f"[V5 Embedding Fusion] Base CV RMSLE: {base_cv:.6f} -> Final: {cv_rmsle:.6f} (Δ={base_cv-cv_rmsle:+.6f})")
    print("\nBenefits:\n - Captures smooth similarities across categories.\n - Keeps residual small to avoid overfit.")
    print("Drawbacks:\n - Needs careful regularization; may be sensitive to seeds.\n - Without true learned embeddings, gains are modest.")

if __name__ == "__main__":
    main()
