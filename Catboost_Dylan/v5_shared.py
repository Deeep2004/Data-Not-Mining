
"""
v5_shared.py
Utilities shared across V5 experiments for the Kaggle House Prices dataset.

- Feature engineering (reusing v2 pack): TotalSF, Age, RemodAge, ratios, target encoding for Neighborhood,
  LotFrontage imputation by Neighborhood median, optional outlier filtering.
- CV: KFold (default) and GroupKFold by Neighborhood (flag), consistent seeding.
- CatBoost baseline model with v3-lite defaults.
- RMSLE metric on log1p target.
- Logging helpers and result collectors.

Place this file next to your train.csv. Requires:
  pip install catboost scikit-learn pandas numpy category_encoders
LightGBM is optional for some experiments; scripts fall back to sklearn HGBR if not installed.
"""

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    raise RuntimeError("CatBoost is required. Install via `pip install catboost`.") from e

# Optional encoders
try:
    import category_encoders as ce
except:
    ce = None

SEED = 19981124
N_SPLITS = 5

# === V5 default: use GroupKFold when possible ===
USE_GROUP_KFOLD_DEFAULT = True
DEFAULT_GROUP_COL = "Neighborhood"

# ------------------ Feature engineering (V2 pack) ------------------

def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalSF (commonly used sum of basement + 1st + 2nd)
    df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    # Age and RemodAge relative to YrSold if available
    if "YrSold" in df and "YearBuilt" in df:
        df["Age"] = df["YrSold"] - df["YearBuilt"]
    else:
        # Fallback: use YearBuilt relative to its min
        df["Age"] = df["YearBuilt"] - df["YearBuilt"].min()
    if "YrSold" in df and "YearRemodAdd" in df:
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    else:
        df["RemodAge"] = df["YearRemodAdd"] - df["YearRemodAdd"].min()
    return df

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Baths count (weighted)
    baths = (
        df.get("FullBath", 0) +
        0.5*df.get("HalfBath", 0) +
        df.get("BsmtFullBath", 0) +
        0.5*df.get("BsmtHalfBath", 0)
    )
    rooms = df.get("TotRmsAbvGrd", 1).replace(0, 1)
    df["BathsPerRoom"] = baths / rooms
    # GrLivArea / TotalSF
    tsf = df["TotalSF"].replace(0, np.nan)
    df["GrLivArea_to_TotalSF"] = df["GrLivArea"] / tsf
    df["GrLivArea_to_TotalSF"] = df["GrLivArea_to_TotalSF"].fillna(0.0)
    return df

def impute_lot_frontage_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        med = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(med)
    return df

def make_target_encoding(train_df: pd.DataFrame, col: str, y_log: pd.Series, n_splits: int = N_SPLITS, seed: int = SEED) -> Tuple[pd.Series, Dict[str, float]]:
    """KFold target encoding to avoid leakage."""
    if ce is None:
        # Simple fallback: global mean mapping (with smoothing)
        mapping = train_df.groupby(col)[y_log.name].mean().to_dict()
        return train_df[col].map(mapping), mapping
    te = ce.TargetEncoder(cols=[col], smoothing=0.3)
    # KFold scheme to avoid leakage
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.Series(index=train_df.index, dtype=float)
    for tr_idx, va_idx in kf.split(train_df):
        te.fit(train_df.iloc[tr_idx][[col]], y_log.iloc[tr_idx])
        oof.iloc[va_idx] = te.transform(train_df.iloc[va_idx][[col]])[col].astype(float).values
    # Fit on full for test-time mapping
    te.fit(train_df[[col]], y_log)
    mapping = None  # category_encoders handles transform on new data at inference in scripts
    return oof.astype(float), {"encoder": "category_encoders.TargetEncoder", "smoothing": 0.3}

def add_target_encoding(df: pd.DataFrame, y_log: Optional[pd.Series]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    info = {}
    if y_log is not None and "Neighborhood" in df.columns:
        oof_nei, meta = make_target_encoding(df, "Neighborhood", y_log)
        df["Neighborhood_TE"] = oof_nei.astype(float)
        info["Neighborhood_TE"] = meta
    return df, info

def feature_engineering(train_df: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    """Return (X, num_cols, cat_cols, info)."""
    df = train_df.copy()
    # Log target
    y_log = None if y is None else np.log1p(y)
    # Impute LF
    df = impute_lot_frontage_by_neighborhood(df)
    # Structural + ratios
    df = add_structural_features(df)
    df = add_ratio_features(df)
    # Target encoding (Neighborhood)
    df, te_info = add_target_encoding(df, y_log)

    # Define categoricals (CatBoost handles directly)
    # Coerce object columns to strings and remove NaNs (CatBoost requires str/int, not NaN/float)
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        df[c] = df[c].fillna("NA").astype(str)
    cat_cols = obj_cols
    # Keep some typical categoricals explicitly if dtype inference failed (stay numeric)
    for c in ["MSSubClass", "OverallQual", "OverallCond"]:
        if c in df.columns and df[c].dtype != "object":
            pass

    # Drop obvious leakage or target if present
    drop_cols = ["SalePrice", "Id"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    num_cols = [c for c in df.columns if c not in cat_cols]
    return df, num_cols, cat_cols, {"te": te_info}

# ------------------ CV, metrics, base CatBoost ------------------

def get_cv_splitter(X: pd.DataFrame, use_group_kfold: Optional[bool], groups: Optional[pd.Series] = None, n_splits: int = N_SPLITS, seed: int = SEED):
    """Return a splitter and the group array.
    If use_group_kfold is True or (None and USE_GROUP_KFOLD_DEFAULT), and a group column is available,
    use GroupKFold. Otherwise fall back to shuffled KFold.
    """
    # Decide whether to use groups
    if use_group_kfold is None:
        use_group_kfold = USE_GROUP_KFOLD_DEFAULT
    # If caller didn't pass groups, try to infer from X
    if groups is None and use_group_kfold and DEFAULT_GROUP_COL in X.columns:
        groups = X[DEFAULT_GROUP_COL]
    if use_group_kfold and groups is not None:
        return GroupKFold(n_splits=n_splits), groups.values
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

@dataclass
class FoldResult:
    fold: int
    best_iteration: int
    val_rmsle: float

def catboost_default_params() -> Dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 1.0,
        "bagging_temperature": 1.0,
        "learning_rate": 0.04,
        "rsm": 0.85,
        "iterations": 2000,
        "random_seed": SEED,
        "early_stopping_rounds": 200,
        "eval_metric": "RMSE",
        "verbose": 200,
    }

def run_catboost_cv(X: pd.DataFrame, y: pd.Series, cat_cols: List[str], use_group_kfold: Optional[bool] = None, groups: Optional[pd.Series] = None, params: Optional[Dict[str, Any]] = None, label: str = "base") -> Tuple[np.ndarray, List[FoldResult], Dict[str, Any]]:
    params = params or catboost_default_params()
    kf, group_values = get_cv_splitter(X, use_group_kfold, groups)
    oof_pred = np.zeros(len(y), dtype=float)
    fold_results: List[FoldResult] = []
    models = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y, group_values), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # CatBoost with categorical features (by indices)
        cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

        train_pool = Pool(X_tr, label=np.log1p(y_tr), cat_features=cat_idx)
        valid_pool = Pool(X_va, label=np.log1p(y_va), cat_features=cat_idx)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=params.get("verbose", 200))

        pred_log = model.predict(valid_pool)
        pred = np.expm1(pred_log)
        oof_pred[va_idx] = pred
        val_score = rmsle(y_va, pred)
        fold_results.append(FoldResult(fold, model.get_best_iteration(), val_score))
        models.append(model)

    cv_score = rmsle(y, oof_pred)
    summary = {
        "label": label,
        "cv_rmsle": float(cv_score),
        "folds": [asdict(fr) for fr in fold_results],
        "params": params,
    }
    return oof_pred, fold_results, {"models": models, "summary": summary}

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_train(path: str = "train.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Put Kaggle train.csv next to this script.")
    df = pd.read_csv(path)
    return df

def scoreboard_row(method: str, cv_rmsle: float, notes: str) -> Dict[str, Any]:
    return {"method": method, "cv_rmsle": round(float(cv_rmsle), 6), "notes": notes}

def top_quantile_rmsle(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.8) -> float:
    t = np.quantile(y_pred, q)
    mask = y_pred >= t
    return rmsle(y_true[mask], y_pred[mask])

def calibrate_tail_isotonic(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.8) -> np.ndarray:
    """Fit isotonic mapping on top-quantile and apply to those predictions."""
    thresh = np.quantile(y_pred, q)
    mask = y_pred >= thresh
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_pred[mask], y_true[mask])
    y_adj = y_pred.copy()
    y_adj[mask] = iso.transform(y_pred[mask])
    return y_adj
