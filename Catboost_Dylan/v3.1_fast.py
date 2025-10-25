# =============== CATBOOST v3 (FAST) — Model-Centric, Faster =================
# Two-stage search with cached Pools and a trimmed grid.
# Stage 1: quick scan (low iterations) over a randomized subset of params.
# Stage 2: re-evaluate Top-K with larger iterations.
#
# Outputs are saved under ./catboost_v3_outputs_fast/
# ============================================================================

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_log_error

# ----------------- Toggles / Config -----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# CV
N_SPLITS = 5
USE_GROUP_KFOLD = True          # set False to compare with KFold

# Speed strategy
FAST_MODE = True
N_TRIALS_STAGE1 = 12            # randomized samples from param grid (stage 1)
TOP_K_STAGE2 = 5                # re-run best K (stage 2)

# Iterations
ITER_STAGE1 = 2000
OD_WAIT_STAGE1 = 100
ITER_STAGE2 = 8000
OD_WAIT_STAGE2 = 200

# Loss candidates
LOSS_FUNCS = ["RMSE", "MAE"]

# Trimmed grid (keeps strong, diverse regularization knobs)
GRID = {
    "depth":              [6, 8],            # shallower trees are faster and often robust
    "l2_leaf_reg":        [3.0, 6.0, 10.0],
    "subsample":          [1.0, 0.8],        # instance subsampling
    "bagging_temperature":[0.0, 1.0],        # Bayesian bootstrap amount
    "learning_rate":      [0.08, 0.06, 0.04],
    "rsm":                [1.0, 0.85],       # feature sampling per split (speeds up & regularizes)
}

# Files
DATA_PATH = Path("train.csv")
ALT_DATA_PATH = Path("../data/train.csv")
OUT_DIR = Path("./catboost_v3_outputs_fast")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Utilities -----------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

def load_train() -> pd.DataFrame:
    p = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    df = pd.read_csv(p)
    return df

def get_categorical_indices(df: pd.DataFrame, label_col: str) -> List[int]:
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
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
    keys = sorted(grid.keys())
    combos = [[]]
    for k in keys:
        new = []
        for base in combos:
            for v in grid[k]:
                new.append(base + [(k, v)])
        combos = new
    return [dict(pairs) for pairs in combos]

def sample_params(grid: Dict[str, List], n: int) -> List[Dict]:
    all_params = param_product(grid)
    if n >= len(all_params):
        return all_params
    idx = np.random.choice(len(all_params), size=n, replace=False)
    return [all_params[i] for i in idx]

# ----------------- Main CV routine -----------------
def run_cv_fast():
    df = load_train()
    label_col = "SalePrice"
    assert label_col in df.columns

    y = df[label_col].values
    y_log = np.log1p(y)
    X = df.drop(columns=[label_col])

    # Ensure categoricals are strings (handle NaNs) — prevents CatBoostError on NaN cats
    cat_names = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category")]
    for c in cat_names:
        X[c] = X[c].astype(str).replace({"nan": "__NA__", "None": "__NA__"})

    cat_idx = get_categorical_indices(df, label_col)

    groups = df["Neighborhood"].values if "Neighborhood" in df.columns else np.arange(len(df))
    splitter, grp = make_splitter(groups)

    # Precompute splits once
    folds = list(splitter.split(X, groups=grp) if grp is not None and USE_GROUP_KFOLD
                 else splitter.split(X))

    # Cache Pools once per fold (huge speedup vs rebuilding every trial)
    fold_pools = []
    for tr_idx, va_idx in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]
        train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)
        fold_pools.append((train_pool, valid_pool, va_idx))

    trials_stage1 = []
    best_candidates = []

    # ----------------- Stage 1: quick scan -----------------
    for loss in LOSS_FUNCS:
        for params in sample_params(GRID, N_TRIALS_STAGE1):
            cv_scores = []
            oof = np.zeros(len(X), dtype=float)

            for (train_pool, valid_pool, va_idx) in fold_pools:
                model = CatBoostRegressor(
                    loss_function=loss,
                    depth=params["depth"],
                    l2_leaf_reg=params["l2_leaf_reg"],
                    subsample=params["subsample"],
                    bagging_temperature=params["bagging_temperature"],
                    learning_rate=params["learning_rate"],
                    rsm=params["rsm"],
                    iterations=ITER_STAGE1,
                    random_seed=SEED,
                    eval_metric="RMSE",
                    od_type="Iter",
                    od_wait=OD_WAIT_STAGE1,
                    use_best_model=True,
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=-1,
                )
                model.fit(train_pool, eval_set=valid_pool, verbose=False)
                pred_log = model.predict(valid_pool)
                oof[va_idx] = pred_log
                rmse_log = float(np.sqrt(np.mean((valid_pool.get_label() - pred_log) ** 2)))
                cv_scores.append(rmse_log)

            cv_rmse_log = float(np.sqrt(np.mean((y_log - oof) ** 2)))
            cv_rmsle = float(rmsle(np.expm1(y_log), np.expm1(oof)))
            rec = {"stage":"stage1","loss":loss, **params, "rmse_log":cv_rmse_log, "rmsle":cv_rmsle}
            trials_stage1.append(rec)

    # Pick Top-K across all losses
    stage1_df = pd.DataFrame(trials_stage1).sort_values("rmsle")
    topK = stage1_df.head(TOP_K_STAGE2).to_dict(orient="records")

    # ----------------- Stage 2: refine the best -----------------
    trials_stage2 = []
    best_score = float("inf")
    best_setup = None
    best_oof = None

    for cand in topK:
        params = {k: cand[k] for k in GRID.keys()}
        loss = cand["loss"]
        oof = np.zeros(len(X), dtype=float)

        for (train_pool, valid_pool, va_idx) in fold_pools:
            model = CatBoostRegressor(
                loss_function=loss,
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                subsample=params["subsample"],
                bagging_temperature=params["bagging_temperature"],
                learning_rate=params["learning_rate"],
                rsm=params["rsm"],
                iterations=ITER_STAGE2,
                random_seed=SEED,
                eval_metric="RMSE",
                od_type="Iter",
                od_wait=OD_WAIT_STAGE2,
                use_best_model=True,
                allow_writing_files=False,
                verbose=False,
                thread_count=-1,
            )
            model.fit(train_pool, eval_set=valid_pool, verbose=False)
            pred_log = model.predict(valid_pool)
            oof[va_idx] = pred_log

        cv_rmse_log = float(np.sqrt(np.mean((y_log - oof) ** 2)))
        cv_rmsle = float(rmsle(np.expm1(y_log), np.expm1(oof)))
        rec = {"stage":"stage2","loss":loss, **params, "rmse_log":cv_rmse_log, "rmsle":cv_rmsle}
        trials_stage2.append(rec)

        if cv_rmsle < best_score:
            best_score = cv_rmsle
            best_setup = {"loss":loss, **params}
            best_oof = oof.copy()

    # Save trials
    trials_all = pd.concat([stage1_df, pd.DataFrame(trials_stage2)], ignore_index=True, sort=False)
    trials_path = OUT_DIR / "cv_trials_v3_fast.csv"
    trials_all.to_csv(trials_path, index=False)

    print("\n================ BEST (FAST) ================")
    print(json.dumps(best_setup, indent=2))
    print(f"Best CV RMSLE: {best_score:.5f}")
    print("Saved trial results to:", trials_path)

    # Save OOF of best (original scale)
    oof_df = pd.DataFrame({
        "Id": df["Id"] if "Id" in df.columns else np.arange(1, len(df) + 1),
        "SalePrice_true": y,
        "SalePrice_oof": np.expm1(best_oof),
    })
    oof_path = OUT_DIR / "oof_best_v3_fast.csv"
    oof_df.to_csv(oof_path, index=False)
    print("Saved OOF predictions:", oof_path)

    # Retrain full model with best params (moderate iterations)
    feat_imp_path = OUT_DIR / "feature_importances_v3_fast.csv"
    interactions_path = OUT_DIR / "interaction_strength_v3_fast.csv"

    full_pool = Pool(X, label=y_log, cat_features=cat_idx)
    final_model = CatBoostRegressor(
        loss_function=best_setup["loss"],
        depth=best_setup["depth"],
        l2_leaf_reg=best_setup["l2_leaf_reg"],
        subsample=best_setup["subsample"],
        bagging_temperature=best_setup["bagging_temperature"],
        learning_rate=best_setup["learning_rate"],
        rsm=best_setup["rsm"],
        iterations=ITER_STAGE2,
        random_seed=SEED,
        eval_metric="RMSE",
        od_type="Iter",
        od_wait=OD_WAIT_STAGE2,
        use_best_model=True,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    final_model.fit(full_pool, verbose=False)

    # Export feature importance
    importances = final_model.get_feature_importance(type="FeatureImportance")
    feat_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
    feat_df.to_csv(feat_imp_path, index=False)
    print("Saved feature importances:", feat_imp_path)

    # Interaction strength (if supported)
    try:
        inter = final_model.get_feature_importance(type="Interaction")
        inter_df = pd.DataFrame(inter, columns=["feature_i", "feature_j", "strength"])
        inter_df["feature_i"] = inter_df["feature_i"].astype(int).map(dict(enumerate(X.columns)))
        inter_df["feature_j"] = inter_df["feature_j"].astype(int).map(dict(enumerate(X.columns)))
        inter_df.sort_values("strength", ascending=False, inplace=True)
        inter_df.to_csv(interactions_path, index=False)
        print("Saved interaction strengths:", interactions_path)
    except Exception as e:
        print("Interaction strength not available in this CatBoost build:", e)

    # Save model
    model_path = OUT_DIR / "catboost_v3_fast_full.cbm"
    final_model.save_model(model_path)
    print("Saved final model:", model_path)

    return {
        "trials_path": str(trials_path),
        "oof_path": str(oof_path),
        "feat_imp_path": str(feat_imp_path),
        "interactions_path": str(interactions_path),
        "best_setup": best_setup,
        "best_score": best_score,
    }

if __name__ == "__main__":
    summary = run_cv_fast()
    print(f"Overall BEST (FAST) -> RMSLE={summary['best_score']:.5f}, setup={summary['best_setup']}")
