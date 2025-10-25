# =============== CATBOOST v3 — Model-Centric Exploration =================
# Theme: Make the model itself smarter (regularization, loss/metric, interactions).
# Usage:
#   python catboost_v3.py
#
# What it does:
# - 5-fold CV on log1p target (RMSLE reported)
# - Optional GroupKFold by Neighborhood (to test location shift)
# - Parameter exploration over depth, l2_leaf_reg, subsample, bagging_temperature, learning_rate
# - Compares RMSE vs MAE loss
# - Early stopping (od_wait) + large iterations to let CatBoost find best_iter
# - Saves: OOF preds, per-fold scores, best setting, feature importances, interaction strengths
#
# Notes:
# - CatBoost handles categoricals directly; we pass indices of categorical columns.
# - If CatBoost in your environment supports 'Huber' loss, you can add it to LOSS_FUNCS below.
# ========================================================================

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_log_error

# ----------------- Toggles / Config -----------------
SEED = 42
N_SPLITS = 5
USE_GROUP_KFOLD = False          # <- change to False to compare with random KFold
USE_BEST_PARAMS_RETRAIN = True  # after CV, retrain on full data with best params
ITERATIONS = 20000              # big number with early stopping
OD_WAIT = 300                   # patience for early stopping
EARLY_STOPPING = True

# Loss candidates — safe set (RMSE, MAE). If your CatBoost supports 'Huber', you may add it.
LOSS_FUNCS = ["RMSE", "MAE"]

# Core regularization/search grid. Keep it small & reasonable to avoid overfitting search.
GRID = {
    "depth":              [6, 8, 10],
    "l2_leaf_reg":        [1.0, 3.0, 5.0, 10.0],
    "subsample":          [1.0, 0.8, 0.65],       # stochastic gradient boosting
    "bagging_temperature":[0.0, 1.0, 5.0],       # Bayesian bootstrap amount
    "learning_rate":      [0.06, 0.04, 0.03],
}

# Files
DATA_PATH = Path("train.csv")              # fallback if running in Kaggle working dir
ALT_DATA_PATH = Path("../data/train.csv")# path provided in this project
OUT_DIR = Path("./catboost_v3_outputs")    # where results are saved
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Utilities -----------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE with safety for non-positive values (clip)."""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

def load_train() -> pd.DataFrame:
    p = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    df = pd.read_csv(p)
    return df

def get_categorical_indices(df: pd.DataFrame, label_col: str) -> List[int]:
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    # CatBoost can also handle "category" dtype:
    cat_cols += [c for c in df.columns if str(df[c].dtype).startswith("category")]
    cat_cols = sorted(list(set(cat_cols)))
    feat_cols = [c for c in df.columns if c != label_col]
    cat_idx = [feat_cols.index(c) for c in feat_cols if c in cat_cols]
    return cat_idx

def make_splitter(groups: np.ndarray):
    if USE_GROUP_KFOLD:
        return GroupKFold(n_splits=N_SPLITS), groups
    else:
        return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED), None

def param_product(grid: Dict[str, List]) -> List[Dict]:
    """Cartesian product of a small param grid (deterministic order)."""
    keys = sorted(grid.keys())
    combos = [[]]
    for k in keys:
        new = []
        for base in combos:
            for v in grid[k]:
                new.append(base + [(k, v)])
        combos = new
    return [dict(pairs) for pairs in combos]

# ----------------- Main CV routine -----------------
def run_cv():
    df = load_train()
    label_col = "SalePrice"
    assert label_col in df.columns, "SalePrice not found in training data"

    # Basic cleaning: CatBoost can handle NaNs; we avoid heavy data changes in v3.
    # Prepare X/y
    y = df[label_col].values
    y_log = np.log1p(y)  # work in log space to align with RMSLE
    X = df.drop(columns=[label_col])

    # ---- Ensure categoricals are true strings (CatBoost requires no float/NaN in cats) ----
    cat_names = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category")]
    for c in cat_names:
        # cast to Python strings; NaN becomes 'nan' then we normalize to a sentinel
        X[c] = X[c].astype(str)
        X[c] = X[c].replace({"nan": "__NA__", "None": "__NA__"})

    # Categorical indices
    cat_idx = get_categorical_indices(df, label_col)

    # Optional grouping by Neighborhood to test robustness to location shift
    groups = df["Neighborhood"].values if "Neighborhood" in df.columns else np.arange(len(df))
    splitter, grp = make_splitter(groups)

    # Search over param combos & loss functions
    grid = param_product(GRID)
    trials_summary = []
    best_score = float("inf")
    best_setup = None
    best_oof = None

    for loss in LOSS_FUNCS:
        for params in grid:
            setting_name = {"loss": loss, **params}
            print("\n==========================")
            print("Try:", setting_name)
            print("==========================")

            oof = np.zeros(len(X), dtype=float)
            fold_scores = []

            for fold, split in enumerate(splitter.split(X, groups=grp) if grp is not None and USE_GROUP_KFOLD
                                         else splitter.split(X), 1):
                tr_idx, va_idx = split
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y_log[tr_idx], y_log[va_idx]

                train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
                valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)

                model = CatBoostRegressor(
                    loss_function=loss,
                    depth=params["depth"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    subsample=params["subsample"],
                    bagging_temperature=params["bagging_temperature"],
                    learning_rate=params["learning_rate"],
                    iterations=ITERATIONS,
                    random_seed=SEED,
                    eval_metric="RMSE",      # in log space; we report RMSE on log target
                    od_type="Iter" if EARLY_STOPPING else "None",
                    od_wait=OD_WAIT if EARLY_STOPPING else None,
                    verbose=False,
                )

                model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
                pred_log = model.predict(valid_pool)
                oof[va_idx] = pred_log

                fold_rmse_log = float(np.sqrt(np.mean((y_va - pred_log) ** 2)))
                fold_scores.append(fold_rmse_log)
                print(f"Fold {fold}: RMSE(log)={fold_rmse_log:.5f}, best_iter={model.get_best_iteration()}")

            cv_rmse_log = float(np.sqrt(np.mean((y_log - oof) ** 2)))
            cv_rmsle = float(rmsle(np.expm1(y_log), np.expm1(oof)))
            print(f"CV Summary -> RMSE(log)={cv_rmse_log:.5f}, RMSLE={cv_rmsle:.5f} for {setting_name}")

            trials_summary.append({
                "loss": loss,
                **params,
                "rmse_log": cv_rmse_log,
                "rmsle": cv_rmsle,
                "fold_rmse_log": fold_scores,
            })

            if cv_rmsle < best_score:
                best_score = cv_rmsle
                best_setup = {"loss": loss, **params}
                best_oof = oof.copy()

    # Save CV trials
    trials_df = pd.DataFrame(trials_summary)
    trials_df.sort_values("rmsle", inplace=True)
    trials_path = OUT_DIR / "cv_trials_v3.csv"
    trials_df.to_csv(trials_path, index=False)

    print("\n================ BEST SETTING ================")
    print(json.dumps(best_setup, indent=2))
    print(f"Best CV RMSLE: {best_score:.5f}")
    print("Saved all trial results to:", trials_path)

    # Save OOF predictions for best setting (in original scale)
    oof_df = pd.DataFrame({
        "Id": df["Id"] if "Id" in df.columns else np.arange(1, len(df) + 1),
        "SalePrice_true": y,
        "SalePrice_oof": np.expm1(best_oof),
    })
    oof_path = OUT_DIR / "oof_best_v3.csv"
    oof_df.to_csv(oof_path, index=False)
    print("Saved OOF predictions to:", oof_path)

    # Retrain on full data with best params (optional), export feature importances & interactions
    feat_imp_path = OUT_DIR / "feature_importances_v3.csv"
    interactions_path = OUT_DIR / "interaction_strength_v3.csv"
    if USE_BEST_PARAMS_RETRAIN and best_setup is not None:
        full_pool = Pool(X, label=y_log, cat_features=cat_idx)

        final_model = CatBoostRegressor(
            loss_function=best_setup["loss"],
            depth=best_setup["depth"],
            l2_leaf_reg=best_setup["l2_leaf_reg"],
            subsample=best_setup["subsample"],
            bagging_temperature=best_setup["bagging_temperature"],
            learning_rate=best_setup["learning_rate"],
            iterations=ITERATIONS,
            random_seed=SEED,
            eval_metric="RMSE",
            od_type="Iter" if EARLY_STOPPING else "None",
            od_wait=OD_WAIT if EARLY_STOPPING else None,
            verbose=False,
        )
        final_model.fit(full_pool, verbose=False)
        # Feature importance
        importances = final_model.get_feature_importance(type="FeatureImportance")
        feat_df = pd.DataFrame({"feature": X.columns, "importance": importances})
        feat_df.sort_values("importance", ascending=False, inplace=True)
        feat_df.to_csv(feat_imp_path, index=False)
        print("Saved feature importances to:", feat_imp_path)

        # Interaction strength (pairwise)
        try:
            inter = final_model.get_feature_importance(type="Interaction")
            inter_df = pd.DataFrame(inter, columns=["feature_i", "feature_j", "strength"])
            # Map indices to names
            inter_df["feature_i"] = inter_df["feature_i"].astype(int).map(dict(enumerate(X.columns)))
            inter_df["feature_j"] = inter_df["feature_j"].astype(int).map(dict(enumerate(X.columns)))
            inter_df.sort_values("strength", ascending=False, inplace=True)
            inter_df.to_csv(interactions_path, index=False)
            print("Saved interaction strengths to:", interactions_path)
        except Exception as e:
            print("Interaction strength not available in this CatBoost build:", e)

        # Save final model as binary
        model_path = OUT_DIR / "catboost_v3_full.cbm"
        final_model.save_model(model_path)
        print("Saved final model to:", model_path)

    return {
        "trials_path": str(trials_path),
        "oof_path": str(oof_path),
        "feat_imp_path": str(feat_imp_path),
        "interactions_path": str(interactions_path),
        "best_setup": best_setup,
        "best_score": best_score,
    }

if __name__ == "__main__":
    summary = run_cv()
    # Print a compact one-liner for logging convenience
    print(f"Overall BEST -> RMSLE={summary['best_score']:.5f}, setup={summary['best_setup']}")
