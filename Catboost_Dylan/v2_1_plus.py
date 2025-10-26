
# =============== CATBOOST v2+ — Deeper Data-Centric Improvements ==================
# Focus ONLY on data (no model fine-tuning): richer features, robust NA policies, ordinal encodings (kept alongside raw categoricals), rare bucketing, and safer CV.
# Overall CV: RMSE(log)=0.15101, RMSLE=0.15101
# ==================================================================================

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GroupKFold

# ----------------------------- Config --------------------------------
SEED = 42
N_SPLITS = 5

# Keep model settings similar to v1 for fair comparison
CB_PARAMS = dict(
    iterations=20000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=SEED,
    od_type="Iter",
    od_wait=300,
    verbose=False,
)

DATA_DIR = Path("../data")
TRAIN_PATH = DATA_DIR / "train.csv"

# ----------------------------- Utils ---------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.clip(y_true, 1, None)
    y_pred = np.clip(y_pred, 1, None)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

def minimal_catboost_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].fillna("None")
    return df

def get_cat_cols(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=["object", "category"]).columns)

# ----------------------- Ordinal Encodings (added) -------------------
def add_ordinal_views(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric ordinal versions of known quality/ordered cols (keep originals)."""
    df = df.copy()
    qual_map = {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5, "None":0}
    exp_map  = {"No":0, "Mn":1, "Av":2, "Gd":3, "None":-1}
    fin_map  = {"Unf":0, "LwQ":1, "Rec":2, "BLQ":3, "ALQ":4, "GLQ":5, "None":-1}
    lotshape_map = {"IR3":0, "IR2":1, "IR1":2, "Reg":3, "None":-1}

    for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
        if c in df.columns:
            df[c+"_Ord"] = df[c].map(qual_map).fillna(0).astype(float)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure_Ord"] = df["BsmtExposure"].map(exp_map).fillna(-1).astype(float)
    for c in ["BsmtFinType1","BsmtFinType2"]:
        if c in df.columns:
            df[c+"_Ord"] = df[c].map(fin_map).fillna(-1).astype(float)
    if "LotShape" in df.columns:
        df["LotShape_Ord"] = df["LotShape"].map(lotshape_map).fillna(-1).astype(float)
    return df

# ----------------------- Robust NA Policies --------------------------
def robust_na_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Intentional imputations using dataset semantics (no leakage)."""
    df = df.copy()
    # LotFrontage by neighborhood median
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        med = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(med)

    # Garage-related: if no garage, fill numeric with 0 and cat with "None"
    garage_num = ["GarageYrBlt","GarageArea","GarageCars"]
    garage_cat = ["GarageType","GarageFinish","GarageQual","GarageCond"]
    if "GarageCars" in df.columns:
        no_garage = (df["GarageCars"].fillna(0) == 0) & (df.get("GarageArea",0).fillna(0) == 0)
    else:
        no_garage = df["GarageType"].isna() if "GarageType" in df.columns else pd.Series(False, index=df.index)
    for c in garage_num:
        if c in df.columns:
            df.loc[no_garage, c] = df.loc[no_garage, c].fillna(0)
    for c in garage_cat:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    # Basement-related
    bsmt_num = ["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath"]
    bsmt_cat = ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
    has_bsmt = df[bsmt_num].sum(axis=1).fillna(0) > 0 if set(bsmt_num).issubset(df.columns) else pd.Series(False, index=df.index)
    for c in bsmt_num:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in bsmt_cat:
        if c in df.columns:
            df[c] = df[c].fillna("None")

    # Fireplace
    if "FireplaceQu" in df.columns:
        df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

    # Masonry veneer
    for c in ["MasVnrType"]:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    for c in ["MasVnrArea"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Alley/Fence/MiscFeature/PoolQC already filled by minimal_catboost_clean
    return df

# ----------------------- Feature Engineering -------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Structural
    df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)
    df["SinceGarage"] = df["YrSold"] - df["GarageYrBlt"] if "GarageYrBlt" in df.columns else 0

    # Bathrooms
    df["TotalBaths"] = (
        df.get("FullBath", 0)
        + 0.5 * df.get("HalfBath", 0)
        + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("BsmtHalfBath", 0)
    )
    df["BathsPerRoom"] = df["TotalBaths"] / df["TotRmsAbvGrd"].replace(0, np.nan)

    # Porches / Decks
    df["TotalPorchSF"] = df.get("OpenPorchSF", 0) + df.get("EnclosedPorch", 0) + df.get("3SsnPorch", 0) + df.get("ScreenPorch", 0)

    # Ratios & interactions
    df["GrLivAreaRatio"] = df["GrLivArea"] / df["TotalSF"].replace(0, np.nan)
    df["Qual_x_GrLiv"] = df["OverallQual"] * np.log1p(df["GrLivArea"])
    df["Qual_x_TotalSF"] = df["OverallQual"] * np.log1p(df["TotalSF"])

    # Flags
    df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
    df["Has2ndFlr"] = (df.get("2ndFlrSF", 0) > 0).astype(int)
    df["HasGarage"] = ((df.get("GarageCars", 0) > 0) | (df.get("GarageArea", 0) > 0)).astype(int)
    df["HasBsmt"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)

    # Cyclical month (seasonality) — MoSold
    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * (df["MoSold"] / 12))
        df["MoSold_cos"] = np.cos(2 * np.pi * (df["MoSold"] / 12))

    # Log1p transforms for heavy-skew numerics (added as extra columns)
    skew_cols = ["LotArea","GrLivArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","GarageArea","MasVnrArea","TotalPorchSF","WoodDeckSF","OpenPorchSF","EnclosedPorch","ScreenPorch"]
    for c in skew_cols:
        if c in df.columns:
            df[c+"_Log1p"] = np.log1p(df[c].clip(lower=0))

    return df

# ----------------------- Target Encoding -----------------------------
def fold_safe_target_encode(train_df: pd.DataFrame, valid_df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    tr = train_df.copy()
    tr["_log_price_"] = np.log1p(tr["SalePrice"])
    # smoothing to avoid noisy small groups
    stats = tr.groupby(col)["_log_price_"].agg(["median", "count"]).rename(columns={"median":"med","count":"cnt"})
    global_med = tr["_log_price_"].median()
    m = 50.0
    stats["enc"] = (stats["cnt"] * stats["med"] + m * global_med) / (stats["cnt"] + m)
    tr_enc = tr[col].map(stats["enc"]).fillna(global_med)
    va_enc = valid_df[col].map(stats["enc"]).fillna(global_med)
    return tr_enc, va_enc

# ----------------------- Rare Bucket for High-Card Cats --------------
def rare_bucket(df: pd.DataFrame, col: str, min_count: int = 10) -> pd.Series:
    vc = df[col].value_counts()
    rare = vc[vc < min_count].index
    return df[col].astype(str).where(~df[col].isin(rare), other="__RARE__")

# ------------------------------ Main ---------------------------------
def main():
    rng = np.random.RandomState(SEED)
    train = pd.read_csv(TRAIN_PATH)

    # Base cleaning and robust imputation
    train = minimal_catboost_clean(train)
    train = robust_na_imputation(train)

    # Add ordinal numeric views (keep original string columns too)
    train = add_ordinal_views(train)

    # Add engineered features
    train = add_features(train)

    # Drop near-constant columns (reduce noise)
    near_const_cols = []
    for c in train.select_dtypes(include=["object","category"]).columns:
        top_ratio = train[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_ratio > 0.98:
            near_const_cols.append(c)
    # keep their ordinal versions if exist
    keep = set([c+"_Ord" for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC","BsmtExposure","BsmtFinType1","BsmtFinType2","LotShape"]])
    near_const_cols = [c for c in near_const_cols if c not in keep]

    # Rare bucket on some medium/high-card columns
    for c in ["Exterior1st","Exterior2nd","SaleType","Condition1","Condition2","RoofMatl"]:
        if c in train.columns:
            train[c+"_RB"] = rare_bucket(train, c, min_count=10)

    # Prepare target and features
    y_log = np.log1p(train["SalePrice"])
    drop_cols = ["SalePrice","Id"] + near_const_cols
    features = [c for c in train.columns if c not in drop_cols]
    X_all = train[features].copy()

    # GroupKFold by Neighborhood (robust to location shift)
    if "Neighborhood" not in train.columns:
        raise ValueError("Neighborhood is required.")
    groups = train["Neighborhood"].astype(str).values

    # Detect categorical columns for CatBoost (string-based; numeric ordinals remain numeric)
    cat_cols = X_all.select_dtypes(include=["object","category"]).columns.tolist()

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_log = np.zeros(len(train), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, groups=groups), 1):
        X_tr = X_all.iloc[tr_idx].copy()
        X_va = X_all.iloc[va_idx].copy()
        tr_df = train.iloc[tr_idx]
        va_df = train.iloc[va_idx]

        # fold-safe target encoding for Neighborhood
        tr_enc, va_enc = fold_safe_target_encode(tr_df, va_df, "Neighborhood")
        X_tr["NeighborhoodEnc"] = tr_enc.values
        X_va["NeighborhoodEnc"] = va_enc.values

        # Cat indices
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

        model = CatBoostRegressor(**CB_PARAMS)
        model.fit(Pool(X_tr, y_log.iloc[tr_idx], cat_features=cat_idx),
                  eval_set=Pool(X_va, y_log.iloc[va_idx], cat_features=cat_idx),
                  use_best_model=True,
                  verbose=False)

        pred = model.predict(X_va)
        oof_log[va_idx] = pred

        rmse_log = np.sqrt(np.mean((y_log.iloc[va_idx] - pred) ** 2))
        print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}, best_iter={model.get_best_iteration()}")

    cv_rmse_log = np.sqrt(np.mean((y_log - oof_log) ** 2))
    cv_rmsle = rmsle(np.expm1(y_log.values), np.expm1(oof_log))
    print(f"\nOverall CV: RMSE(log)={cv_rmse_log:.5f}, RMSLE={cv_rmsle:.5f}")

if __name__ == "__main__":
    main()
