# =============== CATBOOST v3 (FAST + PROGRESS) — Model-Centric, Faster =================
# Two-stage search with cached Pools, trimmed grid, and live progress printing.
# Stage 1: quick randomized scan (low iterations)
# Stage 2: refine top-K with more iterations
# ================ BEST (FAST+PROGRESS) ================
# {
#   "loss": "RMSE",
#   "depth": 6,
#   "l2_leaf_reg": 3.0,
#   "subsample": 1.0,
#   "bagging_temperature": 1.0,
#   "learning_rate": 0.04,
#   "rsm": 0.85
# }
# Best CV RMSLE: 0.14649
# ============================================================================

import json, math, random, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_log_error

# ----------------- Config -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
N_SPLITS = 5
USE_GROUP_KFOLD = True
N_TRIALS_STAGE1 = 12
TOP_K_STAGE2 = 5
ITER_STAGE1, OD_WAIT_STAGE1 = 2000, 100
ITER_STAGE2, OD_WAIT_STAGE2 = 8000, 200
LOSS_FUNCS = ["RMSE", "MAE"]
GRID = {
    "depth":[6,8],
    "l2_leaf_reg":[3.0,6.0,10.0],
    "subsample":[1.0,0.8],
    "bagging_temperature":[0.0,1.0],
    "learning_rate":[0.08,0.06,0.04],
    "rsm":[1.0,0.85],
}
DATA_PATH, ALT_DATA_PATH = Path("train.csv"), Path("../data/train.csv")
OUT_DIR = Path("./catboost_v3_outputs_fast"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Helpers -----------------
def rmsle(y_true, y_pred):
    y_true, y_pred = np.maximum(y_true,0), np.maximum(y_pred,0)
    return math.sqrt(mean_squared_log_error(y_true,y_pred))

def load_train():
    p = DATA_PATH if DATA_PATH.exists() else ALT_DATA_PATH
    return pd.read_csv(p)

def get_categorical_indices(df,label_col):
    cat_cols=[c for c in df.columns if df[c].dtype=="object" or str(df[c].dtype).startswith("category")]
    feat_cols=[c for c in df.columns if c!=label_col]
    return [feat_cols.index(c) for c in feat_cols if c in cat_cols]

def make_splitter(groups):
    return (GroupKFold(n_splits=N_SPLITS),groups) if USE_GROUP_KFOLD else (KFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED),None)

def param_product(grid):
    keys=sorted(grid.keys()); combos=[[]]
    for k in keys:
        combos=[base+[(k,v)] for base in combos for v in grid[k]]
    return [dict(pairs) for pairs in combos]

def sample_params(grid,n):
    allp=param_product(grid)
    return allp if n>=len(allp) else [allp[i] for i in np.random.choice(len(allp),n,replace=False)]

# Progress logger
start_time=time.time()
def log_progress(stage,idx,total,msg=""):
    elapsed=time.time()-start_time; pct=100*(idx+1)/total
    print(f"[{stage}] {pct:5.1f}% done | {elapsed/60:6.1f} min elapsed {msg}")

# ----------------- Main -----------------
def run_cv_fast():
    df=load_train(); y=df["SalePrice"].values; y_log=np.log1p(y); X=df.drop(columns=["SalePrice"])
    # handle categoricals
    cat_names=[c for c in X.columns if X[c].dtype=="object" or str(X[c].dtype).startswith("category")]
    for c in cat_names:
        X[c]=X[c].astype(str).replace({"nan":"__NA__","None":"__NA__"})
    cat_idx=get_categorical_indices(df,"SalePrice")
    groups=df["Neighborhood"].values if "Neighborhood" in df.columns else np.arange(len(df))
    splitter,grp=make_splitter(groups)
    folds=list(splitter.split(X,groups=grp) if grp is not None and USE_GROUP_KFOLD else splitter.split(X))

    # Cache Pools
    fold_pools=[]
    for tr,va in folds:
        train_pool=Pool(X.iloc[tr],label=y_log[tr],cat_features=cat_idx)
        valid_pool=Pool(X.iloc[va],label=y_log[va],cat_features=cat_idx)
        fold_pools.append((train_pool,valid_pool,va))

    # -------- Stage 1 --------
    trials1=[]; best_score_stage1=float("inf")
    all_stage1=[(loss,p) for loss in LOSS_FUNCS for p in sample_params(GRID,N_TRIALS_STAGE1)]
    total1=len(all_stage1)
    print(f"Stage 1: {total1} configs × {N_SPLITS}-fold CV")

    for i,(loss,params) in enumerate(all_stage1):
        oof=np.zeros(len(X))
        for fold,(train_pool,valid_pool,va_idx) in enumerate(fold_pools):
            model=CatBoostRegressor(
                loss_function=loss,**params,iterations=ITER_STAGE1,random_seed=SEED,
                eval_metric="RMSE",od_type="Iter",od_wait=OD_WAIT_STAGE1,
                use_best_model=True,allow_writing_files=False,verbose=False,thread_count=-1)
            model.fit(train_pool,eval_set=valid_pool,verbose=False)
            pred_log=model.predict(valid_pool)
            oof[va_idx]=pred_log
            rmse_log=float(np.sqrt(np.mean((valid_pool.get_label()-pred_log)**2)))
            print(f"   Fold {fold+1}/{N_SPLITS}: RMSE(log)={rmse_log:.5f}")
        rmsle_cv=float(rmsle(np.expm1(y_log),np.expm1(oof)))
        trials1.append({"stage":"stage1","loss":loss,**params,"rmsle":rmsle_cv})
        if rmsle_cv<best_score_stage1: best_score_stage1=rmsle_cv
        log_progress("Stage1",i,total1,f"Best RMSLE {best_score_stage1:.4f}")

    stage1_df=pd.DataFrame(trials1).sort_values("rmsle")
    topK=stage1_df.head(TOP_K_STAGE2).to_dict(orient="records")
    print(f"\nStage 1 done. Top {TOP_K_STAGE2} → Stage 2 refinement.\n")

    # -------- Stage 2 --------
    trials2=[]; best_score=float("inf"); best_setup=None; best_oof=None
    total2=len(topK)
    for j,cand in enumerate(topK):
        params={k:cand[k] for k in GRID.keys()}; loss=cand["loss"]
        oof=np.zeros(len(X))
        for fold,(train_pool,valid_pool,va_idx) in enumerate(fold_pools):
            model=CatBoostRegressor(
                loss_function=loss,**params,iterations=ITER_STAGE2,random_seed=SEED,
                eval_metric="RMSE",od_type="Iter",od_wait=OD_WAIT_STAGE2,
                use_best_model=True,allow_writing_files=False,verbose=False,thread_count=-1)
            model.fit(train_pool,eval_set=valid_pool,verbose=False)
            pred_log=model.predict(valid_pool)
            oof[va_idx]=pred_log
            rmse_log=float(np.sqrt(np.mean((valid_pool.get_label()-pred_log)**2)))
            print(f"   Fold {fold+1}/{N_SPLITS}: RMSE(log)={rmse_log:.5f}")
        rmsle_cv=float(rmsle(np.expm1(y_log),np.expm1(oof)))
        trials2.append({"stage":"stage2","loss":loss,**params,"rmsle":rmsle_cv})
        if rmsle_cv<best_score:
            best_score=rmsle_cv; best_setup={"loss":loss,**params}; best_oof=oof.copy()
        log_progress("Stage2",j,total2,f"Best RMSLE {best_score:.4f}")

    trials_all=pd.concat([stage1_df,pd.DataFrame(trials2)],ignore_index=True)
    trials_path=OUT_DIR/"cv_trials_v3_fast_progress.csv"; trials_all.to_csv(trials_path,index=False)

    print("\n================ BEST (FAST+PROGRESS) ================")
    print(json.dumps(best_setup,indent=2))
    print(f"Best CV RMSLE: {best_score:.5f}")
    print("Saved trials:",trials_path)

    # Save OOF
    oof_df=pd.DataFrame({"Id":df.get("Id",pd.Series(np.arange(1,len(df)+1))),
                         "SalePrice_true":y,"SalePrice_oof":np.expm1(best_oof)})
    oof_path=OUT_DIR/"oof_best_v3_fast_progress.csv"; oof_df.to_csv(oof_path,index=False)
    print("Saved OOF:",oof_path)

    # Retrain full model
    full_pool=Pool(X,label=y_log,cat_features=cat_idx)
    final_model=CatBoostRegressor(
        loss_function=best_setup["loss"],**{k:best_setup[k] for k in GRID.keys()},
        iterations=ITER_STAGE2,random_seed=SEED,eval_metric="RMSE",
        od_type="Iter",od_wait=OD_WAIT_STAGE2,use_best_model=True,
        allow_writing_files=False,verbose=False,thread_count=-1)
    final_model.fit(full_pool,verbose=False)

    imp=final_model.get_feature_importance(type="FeatureImportance")
    pd.DataFrame({"feature":X.columns,"importance":imp}).sort_values("importance",ascending=False)\
        .to_csv(OUT_DIR/"feature_importances_v3_fast_progress.csv",index=False)
    print("Saved feature importances.")

    try:
        inter=final_model.get_feature_importance(type="Interaction")
        df_i=pd.DataFrame(inter,columns=["i","j","strength"])
        df_i["i"]=df_i["i"].astype(int).map(dict(enumerate(X.columns)))
        df_i["j"]=df_i["j"].astype(int).map(dict(enumerate(X.columns)))
        df_i.sort_values("strength",ascending=False).to_csv(
            OUT_DIR/"interaction_strength_v3_fast_progress.csv",index=False)
        print("Saved interactions.")
    except Exception as e: print("Interaction strength not available:",e)

    model_path=OUT_DIR/"catboost_v3_fast_progress_full.cbm"
    final_model.save_model(model_path)
    print("Saved final model:",model_path)

    print(f"\nOverall BEST (FAST+PROGRESS) → RMSLE={best_score:.5f}\n")
    return {"best_score":best_score,"best_setup":best_setup}

if __name__=="__main__":
    run_cv_fast()
