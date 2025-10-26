# =============== CATBOOST v3_FINAL — Combined Data + Model Refinement =================
# Combines v2.2_plus_fix data pipeline with v3 tuned hyperparameters.
# Purpose: Unified fast, tuned, interpretable CatBoost model.

# ✅ Final RMSLE=0.14199
# =====================================================================

import math, json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_log_error

# Import data-centric functions from v2
from v2_2_plus_fix import robust_na_imputation, add_ordinal_views, add_features

# ---------------- Config ----------------
SEED = 42
N_SPLITS = 5
USE_GROUP_KFOLD = True

# Best tuned parameters from v3_fast
CB_PARAMS = dict(
    loss_function="RMSE",
    depth=6,
    l2_leaf_reg=3.0,
    subsample=1.0,
    bagging_temperature=1.0,
    learning_rate=0.04,
    rsm=0.85,
    iterations=8000,
    random_seed=SEED,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=300,
    use_best_model=True,
    verbose=200,
    thread_count=-1,
)

DATA_PATH = Path("train.csv")
ALT_DATA_PATH = Path("../data/train.csv")
OUT_DIR = Path("./catboost_v3_final_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utils ----------------
def rmsle(y_true, y_pred):
    y_true, y_pred = np.maximum(y_true,0), np.maximum(y_pred,0)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

def load_train():
    p = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    return pd.read_csv(p)

def get_cat_idx(df,label_col):
    feat_cols = [c for c in df.columns if c != label_col]
    cat_cols = [c for c in feat_cols if (df[c].dtype=="object") or str(df[c].dtype).startswith("category")]
    return [feat_cols.index(c) for c in cat_cols]

# ---------------- Main ----------------
def run_cv():
    df = load_train()
    label_col = "SalePrice"

    # === Apply v2-style data pipeline ===
    df = robust_na_imputation(df)
    df = add_ordinal_views(df)
    df = add_features(df)

    y = np.log1p(df[label_col].values)
    X = df.drop(columns=[label_col])

    # Clean categoricals
    cat_cols = [c for c in X.columns if X[c].dtype=="object" or str(X[c].dtype).startswith("category")]
    for c in cat_cols:
        X[c] = X[c].astype(str).replace({"nan":"__NA__","None":"__NA__"})

    cat_idx = get_cat_idx(df, label_col)
    groups = df["Neighborhood"].values if "Neighborhood" in df.columns else np.arange(len(df))
    splitter = GroupKFold(N_SPLITS) if USE_GROUP_KFOLD else KFold(N_SPLITS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(X))
    fold_scores = []

    print(f"Starting CatBoost v3_final with {N_SPLITS}-fold CV (GroupKFold={USE_GROUP_KFOLD})")
    for fold,(tr,va) in enumerate(splitter.split(X, groups=groups),1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")
        train_pool = Pool(X.iloc[tr], label=y[tr], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[va], label=y[va], cat_features=cat_idx)

        model = CatBoostRegressor(**CB_PARAMS)
        model.fit(train_pool, eval_set=valid_pool)

        pred = model.predict(valid_pool)
        oof[va] = pred
        rmse_log = np.sqrt(np.mean((y[va]-pred)**2))
        fold_scores.append(rmse_log)
        print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}")

    rmsle_cv = rmsle(np.expm1(y), np.expm1(oof))
    print(f"\nOverall CV: RMSE(log)={np.mean(fold_scores):.5f}, RMSLE={rmsle_cv:.5f}")

    df_out = pd.DataFrame({
        "Id": df.get("Id", pd.Series(np.arange(1,len(df)+1))),
        "SalePrice_true": np.expm1(y),
        "SalePrice_oof": np.expm1(oof),
    })
    df_out.to_csv(OUT_DIR/"oof_v3_final.csv", index=False)

    feat_imp = model.get_feature_importance(type="FeatureImportance")
    pd.DataFrame({"feature":X.columns,"importance":feat_imp}).sort_values("importance",ascending=False)\
        .to_csv(OUT_DIR/"feature_importances_v3_final.csv", index=False)

    print("\nSaved outputs in:", OUT_DIR)
    print(json.dumps(CB_PARAMS, indent=2))
    print(f"\n✅ Final RMSLE={rmsle_cv:.5f}")

if __name__ == "__main__":
    run_cv()
