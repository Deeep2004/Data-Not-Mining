
# =============== CATBOOST V2 — DATA-CENTRIC REFINEMENT ==============
# Feature engineering, Neighborhood target-encoding, LotFrontage impute, and GroupKFold by Neighborhood. Keeps log-target training for RMSLE fairness.
# Overall CV: RMSE(log)=0.13855, RMSLE=0.13855
# --------------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool

# ----------------------------- Config --------------------------------
SEED = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 300
ITERATIONS = 20000
LEARNING_RATE = 0.05
DEPTH = 8
L2_LEAF_REG = 3.0
SUBSAMPLE = 0.8
BAGGING_TEMPERATURE = 1.0

# Outlier handling (optional; set thresholds to None to disable)
REMOVE_OUTLIERS = True
Z_MAX = 3.5           # z-score threshold (on log SalePrice)
MAX_GRLIVAREA = 4000  # drop extreme very large houses

# Paths
DATA_DIR = Path("../data")
TRAIN_PATH = DATA_DIR / "train.csv"   # Kaggle train.csv

# ----------------------------- Utils ---------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE on linear space with safety clipping at 1."""
    y_true = np.clip(y_true, 1, None)
    y_pred = np.clip(y_pred, 1, None)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

def set_seed(seed: int = 42):
    np.random.seed(seed)

def minimal_catboost_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categoricals with 'None'. Leave numeric NaNs (CatBoost handles them)."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].fillna("None")
    return df

# ----------------------- Feature Engineering -------------------------
def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Total square footage including basement
    df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    # Age at sale and years since remodel
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    # Bathroom ratio per room
    df["BathsPerRoom"] = (
        df.get("FullBath", 0)
        + 0.5 * df.get("HalfBath", 0)
        + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("BsmtHalfBath", 0)
    ) / df["TotRmsAbvGrd"].replace(0, np.nan)
    # Ratio of above-ground living area to total
    df["GrLivAreaRatio"] = df["GrLivArea"] / df["TotalSF"].replace(0, np.nan)
    return df

def impute_lot_frontage_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        med = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(med)
    return df

def optional_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not REMOVE_OUTLIERS:
        return df
    df = df.copy()
    # z-score on log SalePrice (train-only column)
    if "SalePrice" in df.columns:
        logp = np.log1p(df["SalePrice"])
        z = (logp - logp.mean()) / (logp.std(ddof=0) + 1e-12)
        df = df.loc[z.abs() < Z_MAX]
    if MAX_GRLIVAREA is not None and "GrLivArea" in df.columns:
        df = df.loc[df["GrLivArea"] <= MAX_GRLIVAREA]
    return df

# ------------------------- Target Encoding ---------------------------
def target_encode_neighborhood(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, col: str = "Neighborhood"
) -> Tuple[pd.Series, pd.Series]:
    """
    Fold-safe target encoding of 'Neighborhood' using MEDIAN of log1p(SalePrice) on train fold only.
    """
    assert "SalePrice" in train_df.columns, "train_df must include SalePrice for target encoding"
    tr = train_df.copy()
    tr["_log_price_"] = np.log1p(tr["SalePrice"])
    mapping = tr.groupby(col)["_log_price_"].median()
    tr_enc = tr[col].map(mapping)
    va_enc = valid_df[col].map(mapping)
    # Fallback for unseen categories: global median
    global_med = tr["_log_price_"].median()
    tr_enc = tr_enc.fillna(global_med)
    va_enc = va_enc.fillna(global_med)
    return tr_enc, va_enc

# ----------------------- Categorical Handling ------------------------
def get_cat_cols(df: pd.DataFrame) -> List[str]:
    # Object/category columns are categorical for CatBoost
    return list(df.select_dtypes(include=["object", "category"]).columns)

# ------------------------------ Main ---------------------------------
def main():
    set_seed(SEED)
    print("Loading data from:", TRAIN_PATH)
    train = pd.read_csv(TRAIN_PATH)

    # Basic cleaning prior to feature engineering
    train = minimal_catboost_clean(train)

    # Data quality: LotFrontage imputation by Neighborhood
    train = impute_lot_frontage_by_neighborhood(train)

    # Feature engineering
    train = add_structural_features(train)

    # Optional: outlier filtering (train only)
    train = optional_outlier_filter(train)

    # Prepare features/target (log target)
    y_log = np.log1p(train["SalePrice"])
    feature_cols = [c for c in train.columns if c not in ["SalePrice", "Id"]]

    # Create a working copy for encoding
    X_all = train[feature_cols].copy()

    # CV setup: GroupKFold by Neighborhood to test location robustness
    if "Neighborhood" not in X_all.columns:
        raise ValueError("Neighborhood column is required for GroupKFold in v2.")
    groups = X_all["Neighborhood"].astype(str).values

    # Collect categorical columns (indices for CatBoost Pool)
    cat_cols = get_cat_cols(X_all)
    # We'll add one more engineered categorical column below: NeighborhoodEnc (numeric), so keep list separate
    print(f"Found {len(cat_cols)} categorical columns.")

    gkf = GroupKFold(n_splits=N_SPLITS)

    oof_log_pred = np.zeros(len(train), dtype=float)
    fold_importances = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, groups=groups), 1):
        X_tr = X_all.iloc[tr_idx].copy()
        X_va = X_all.iloc[va_idx].copy()
        y_tr = y_log.iloc[tr_idx].copy()
        y_va = y_log.iloc[va_idx].copy()

        # Fold-safe target encoding for Neighborhood (on log SalePrice medians)
        tr_enc, va_enc = target_encode_neighborhood(train.iloc[tr_idx], train.iloc[va_idx], col="Neighborhood")
        X_tr["NeighborhoodEnc"] = tr_enc.values
        X_va["NeighborhoodEnc"] = va_enc.values

        # Build categorical index list (by position) for CatBoost
        # CatBoost treats numeric 'NeighborhoodEnc' as numeric; leave original 'Neighborhood' as categorical.
        cat_features_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_features_idx)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_features_idx)

        model = CatBoostRegressor(
            iterations=ITERATIONS,
            learning_rate=LEARNING_RATE,
            depth=DEPTH,
            l2_leaf_reg=L2_LEAF_REG,
            loss_function="RMSE",     # RMSE on log target -> corresponds to RMSLE after expm1
            eval_metric="RMSE",
            random_seed=SEED,
            od_type="Iter",
            od_wait=EARLY_STOPPING_ROUNDS,
            subsample=SUBSAMPLE,
            bagging_temperature=BAGGING_TEMPERATURE,
            verbose=False,
        )

        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)

        va_pred_log = model.predict(valid_pool)
        oof_log_pred[va_idx] = va_pred_log

        # Metrics (log-RMSE and RMSLE on linear)
        rmse_log = np.sqrt(np.mean((y_va - va_pred_log) ** 2))
        rmsle_lin = rmsle(np.expm1(y_va.values), np.expm1(va_pred_log))
        print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}, RMSLE={rmsle_lin:.5f}, best_iter={model.get_best_iteration()}")

        # Store fold importances
        imp = pd.DataFrame({
            "feature": X_tr.columns,
            "importance": model.get_feature_importance(train_pool, type="PredictionValuesChange"),
            "fold": fold,
        })
        fold_importances.append(imp)

    # Overall CV
    cv_rmse_log = np.sqrt(np.mean((y_log - oof_log_pred) ** 2))
    cv_rmsle = rmsle(np.expm1(y_log.values), np.expm1(oof_log_pred))
    print(f"\nOverall CV: RMSE(log)={cv_rmse_log:.5f}, RMSLE={cv_rmsle:.5f}")

    # Save artifacts
    out_dir = DATA_DIR / "catboost_v2_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOF predictions (both log and linear)
    pd.DataFrame({
        "Id": train["Id"].values if "Id" in train.columns else np.arange(len(train)) + 1,
        "SalePrice_log_oof": oof_log_pred,
        "SalePrice_oof": np.expm1(oof_log_pred),
        "SalePrice_true": train["SalePrice"].values,
    }).to_csv(out_dir / "oof_predictions.csv", index=False)

    # Feature importances
    imp_df = pd.concat(fold_importances, ignore_index=True)
    imp_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)\
          .to_csv(out_dir / "feature_importance_mean.csv", index=False)
    imp_df.to_csv(out_dir / "feature_importance_by_fold.csv", index=False)

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print(" - oof_predictions.csv")
    print(" - feature_importance_mean.csv")
    print(" - feature_importance_by_fold.csv")

if __name__ == "__main__":
    main()
