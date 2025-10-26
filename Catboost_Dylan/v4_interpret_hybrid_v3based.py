#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost v4 — Interpretability & Hybridization (v3-lite based)
Single-file, self-contained.

What’s inside
-------------
• v3-lite defaults (params + key feature engineering from v2.2_plus_fix):
  - TotalSF, Age, RemodAge, BathsPerRoom, GrLivArea/TotalSF
  - Robust LotFrontage imputation by Neighborhood median
  - Ordinal encodings for quality/condition
• 5-fold CV on log-target (RMSLE equivalence)
• SHAP (global bar) using CatBoost ShapValues
• Pairwise interaction strength (CatBoost)
• PDPs for top SHAP features
• Simple hybrid: CatBoost + Linear on leaf one-hots (fixed alpha)
• Calibration & residual plots
• Human-readable report with V5 directions

Usage
-----
python v4_interpret_hybrid_v3based.py --data train.csv --outdir v4_outputs --hybrid
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.inspection import PartialDependenceDisplay

from catboost import CatBoostRegressor, Pool

# ---------------- v3-lite default hyperparameters ----------------
CB_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=5000,
    learning_rate=0.04,
    depth=6,
    l2_leaf_reg=3.0,
    subsample=1.0,
    rsm=0.85,
    bagging_temperature=1.0,
    random_seed=42,
    od_type="Iter",
)

N_SPLITS = 5
SEED = 42

# ---------------- Feature Engineering (inlined from v2 ideas) ----------------

ORDINAL_MAPS = {
    "ExterQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "ExterCond": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "BsmtQual": {"NA":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "BsmtCond": {"NA":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "HeatingQC": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "KitchenQual": {"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "FireplaceQu": {"NA":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "GarageQual": {"NA":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
    "GarageCond": {"NA":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},
}

def robust_na_imputation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # LotFrontage by neighborhood median
    if "LotFrontage" in out.columns and "Neighborhood" in out.columns:
        med_by_nb = out.groupby("Neighborhood")["LotFrontage"].transform("median")
        out["LotFrontage"] = out["LotFrontage"].fillna(med_by_nb)
    # Common NA-as-zero numeric basements/garages
    for col in ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                "GarageCars","GarageArea","BsmtFullBath","BsmtHalfBath"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    # Mode-fill some categoricals
    for col in ["MSZoning","Electrical","Exterior1st","Exterior2nd","KitchenQual","SaleType"]:
        if col in out.columns:
            out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out

def add_ordinal_views(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c, mp in ORDINAL_MAPS.items():
        if c in out.columns:
            out[c+"_ord"] = out[c].fillna("NA").map(mp).fillna(0).astype(int)
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Total square footage
    for cols, name in [ (["TotalBsmtSF","1stFlrSF","2ndFlrSF"], "TotalSF") ]:
        if all(c in out.columns for c in cols):
            out[name] = out[cols].sum(axis=1)
    # House age and remodel age
    if "YrSold" in out.columns and "YearBuilt" in out.columns:
        out["Age"] = out["YrSold"] - out["YearBuilt"]
    if "YrSold" in out.columns and "YearRemodAdd" in out.columns:
        out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    # Ratios
    if "GrLivArea" in out.columns and "TotalSF" in out.columns:
        out["GrLivArea_to_TotalSF"] = out["GrLivArea"] / out["TotalSF"].replace(0,np.nan)
        out["GrLivArea_to_TotalSF"] = out["GrLivArea_to_TotalSF"].fillna(0)
    # Bathrooms per room
    num = 0
    for c in ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"]:
        if c in out.columns: num += out[c].fillna(0)
    if "TotRmsAbvGrd" in out.columns:
        out["BathsPerRoom"] = num / out["TotRmsAbvGrd"].replace(0,np.nan)
        out["BathsPerRoom"] = out["BathsPerRoom"].fillna(0)
    return out

# ---------------- Helpers ----------------

def rmsle_from_logs(y_log_true, y_log_pred):
    return float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))

def safe_mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ---------------- Training & CV ----------------

def prepare_data(df: pd.DataFrame, target_col="SalePrice"):
    df1 = robust_na_imputation(df)
    df2 = add_ordinal_views(df1)
    df3 = add_features(df2)

    y_log = np.log1p(df3[target_col].values)
    X = df3.drop(columns=[target_col, "Id"], errors="ignore").copy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    for c in cat_cols:
        X[c] = X[c].astype("str").fillna("__NA__")
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y_log, cat_cols, num_cols

def run_cv(
    df: pd.DataFrame,
    use_group_kfold=False,
    hybrid=True,
    lasso=False,
    alpha=0.70,
    outdir="v4_outputs"
):
    safe_mkdir(outdir); safe_mkdir(os.path.join(outdir,"plots")); safe_mkdir(os.path.join(outdir,"artifacts"))
    X, y_log, cat_cols, num_cols = prepare_data(df)

    if use_group_kfold and "Neighborhood" in df.columns:
        groups = df["Neighborhood"]
        splitter = GroupKFold(n_splits=N_SPLITS)
        split_iter = splitter.split(X, y_log, groups=groups)
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(X)

    oof_cat = np.zeros(len(X)); oof_hyb = np.zeros(len(X)) if hybrid else None
    models = []; hyb_parts = []; lasso_models = []
    for fold, (tr, va) in enumerate(split_iter, 1):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y_log[tr], y_log[va]
        pool_tr = Pool(Xtr, ytr, cat_features=cat_cols)
        pool_va = Pool(Xva, yva, cat_features=cat_cols)

        model = CatBoostRegressor(**CB_PARAMS, early_stopping_rounds=200, verbose=200)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        models.append(model)

        pred_va = model.predict(pool_va)
        oof_cat[va] = pred_va

        if hybrid:
            leaf_tr = model.calc_leaf_indexes(pool_tr)
            leaf_va = model.calc_leaf_indexes(pool_va)
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            except TypeError:
                # for older scikit-learn versions
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
            Xtr_leaf = ohe.fit_transform(leaf_tr)
            Xva_leaf = ohe.transform(leaf_va)
            lin = LinearRegression().fit(Xtr_leaf, ytr)
            lin_va = lin.predict(Xva_leaf)

            if lasso:
                lasso_model = LassoCV(cv=5, random_state=SEED, n_alphas=50).fit(Xtr[num_cols], ytr)
                lasso_va = lasso_model.predict(Xva[num_cols])
                lasso_models.append(lasso_model)
            else:
                lasso_va = 0.0

            hyb = alpha*pred_va + (1-alpha)*lin_va
            if lasso:
                hyb = 0.9*hyb + 0.1*lasso_va
            oof_hyb[va] = hyb
            hyb_parts.append((ohe, lin))

        fold_msg = f"Fold {fold}: CatBoost RMSLE={rmsle_from_logs(yva, pred_va):.5f}"
        if hybrid:
            fold_msg += f" | Hybrid RMSLE={rmsle_from_logs(yva, hyb):.5f}"
        print(fold_msg)

    res = {"cv_cat_rmsle": rmsle_from_logs(y_log, oof_cat)}
    print(f"Overall CV: CatBoost RMSLE={res['cv_cat_rmsle']:.5f}")
    if hybrid:
        res["cv_hybrid_rmsle"] = rmsle_from_logs(y_log, oof_hyb)
        print(f"Overall CV: Hybrid   RMSLE={res['cv_hybrid_rmsle']:.5f}")

    with open(os.path.join(outdir,"cv_results_v4.json"),"w") as f: json.dump(res, f, indent=2)

    pack = dict(models=models, X=X, y_log=y_log, cat_cols=cat_cols, results=res,
                oof_hyb=oof_hyb, oof_cat=oof_cat, hyb_parts=hyb_parts,
                num_cols=num_cols)
    return pack

# ---------------- SHAP / Interactions / PDP / Calibration ----------------

def compute_global_shap(model, X_df, cat_cols, shap_sample=800):
    Xs = X_df.sample(min(shap_sample, len(X_df)), random_state=SEED) if shap_sample else X_df
    pool = Pool(Xs, cat_features=cat_cols)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_contrib = shap_vals[:, :-1]
    mean_abs = np.abs(shap_contrib).mean(axis=0)
    df = pd.DataFrame({"feature": X_df.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    return df.reset_index(drop=True), Xs, shap_contrib

def plot_top_shap_bar(df_shap, out_png, topn=20):
    top = df_shap.head(topn)[::-1]
    plt.figure(figsize=(8, max(4, 0.4*len(top))))
    plt.barh(top["feature"], top["mean_abs_shap"]); plt.title("Top features by |SHAP|"); plt.xlabel("Mean |SHAP| (log)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def compute_interactions(model, X_df, y_log, cat_cols, topk=20):
    pool = Pool(X_df, y_log, cat_features=cat_cols)
    inter = model.get_feature_importance(type="Interaction", data=pool)
    rows = [(X_df.columns[int(i)], X_df.columns[int(j)], float(s)) for s,i,j in inter]
    df = pd.DataFrame(rows, columns=["feature_1","feature_2","interaction_score"]).sort_values("interaction_score", ascending=False).reset_index(drop=True)
    return df.head(topk), df

def plot_top_interactions(df_top, out_png):
    plt.figure(figsize=(8, max(4, 0.4*len(df_top))))
    y = np.arange(len(df_top))[::-1]
    plt.barh(y, df_top["interaction_score"])
    labels = [f"{a} × {b}" for a,b in zip(df_top["feature_1"], df_top["feature_2"])]
    plt.yticks(y, labels); plt.xlabel("Interaction strength"); plt.title("Top pairwise interactions")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_pdp(model, X_df, features, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in features:
        try:
            fig, ax = plt.subplots(figsize=(6,4))
            PartialDependenceDisplay.from_estimator(model, X_df, [f], ax=ax)
            ax.set_title(f"PDP — {f}")
            plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"pdp_{f}.png"), dpi=150); plt.close(fig)
        except Exception as e:
            print(f"[PDP] Skip {f}: {e}")

def plot_calibration_and_residuals(y_log_true, y_log_pred, out_prefix):
    plt.figure(figsize=(5,5))
    plt.scatter(y_log_true, y_log_pred, s=6, alpha=0.5)
    lo, hi = min(y_log_true.min(), y_log_pred.min()), max(y_log_true.max(), y_log_pred.max())
    plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel("True log price"); plt.ylabel("Pred log price"); plt.title("Calibration")
    plt.tight_layout(); plt.savefig(out_prefix+"_calibration.png", dpi=150); plt.close()
    resid = y_log_true - y_log_pred
    plt.figure(figsize=(6,4)); plt.hist(resid, bins=40); plt.title("Residuals (log)"); plt.tight_layout()
    plt.savefig(out_prefix+"_residual_hist.png", dpi=150); plt.close()

# ---------------- Report ----------------

def write_report(path, meta):
    L = []
    L.append("CatBoost v4 — Interpretability & Hybridization (v3-lite based)\n")
    L.append("== CV Results ==\n")
    L.append(f"CatBoost RMSLE: {meta['cv_cat']:.5f}\n")
    if meta.get("cv_hyb") is not None: L.append(f"Hybrid   RMSLE: {meta['cv_hyb']:.5f}\n")
    L.append("\n== What drives price (top features) ==\n")
    for f,v in meta["top_features"]: L.append(f"- {f}: mean |SHAP| = {v:.4f}\n")
    L.append("\nActionable: Concentrate V5 on the top 5–10 features. Try explicit crosses and regime-aware models where PDPs show thresholds.\n\n")
    L.append("== Important interactions (pairs) ==\n")
    for a,b,s in meta["top_pairs"]: L.append(f"- {a} × {b}: interaction = {s:.4f}\n")
    L.append("\nActionable: In V5, add engineered crosses for these pairs, or train a small residual model using just these interactions.\n\n")
    L.append("== PDPs inspected ==\n")
    L.append(", ".join(meta["pdp_features"]) + "\n\n")
    L.append("== Suggested V5 directions ==\n")
    L.append("1) Learn blending weights via stacking (replace fixed alpha with meta-learner on CatBoost+Linear(+Lasso)).\n")
    L.append("2) Residual modeling on top features/interactions (LightGBM/MLP) to capture leftover nonlinearity.\n")
    L.append("3) Regime splits if PDP shows saturation (e.g., GrLivArea > 3000) and fit per-regime models.\n")
    L.append("4) Uncertainty: multi-seed CatBoost ensemble to estimate prediction intervals.\n")
    L.append("5) Target-aware calibration for luxury tail where residuals skew.\n")
    Path(path).write_text("".join(L), encoding="utf-8")

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train.csv")
    ap.add_argument("--outdir", type=str, default="v4_outputs")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--lasso", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--pdp_topn", type=int, default=6)
    ap.add_argument("--shap_sample", type=int, default=800)
    ap.add_argument("--groupkfold", action="store_true", help="Use GroupKFold by Neighborhood")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    pack = run_cv(df, use_group_kfold=args.groupkfold, hybrid=args.hybrid, lasso=args.lasso, alpha=args.alpha, outdir=args.outdir)
    models = pack["models"]
    # pick fold model with best validation RMSE
    best = None; best_score = 1e9
    for m in models:
        sc = m.get_best_score()["validation"]["RMSE"]
        if sc < best_score: best, best_score = m, sc

    X, y_log, cat_cols = pack["X"], pack["y_log"], pack["cat_cols"]
    res = pack["results"]

    # SHAP
    df_shap, Xs, shap_contrib = compute_global_shap(best, X, cat_cols, shap_sample=args.shap_sample)
    df_shap.to_csv(os.path.join(args.outdir,"artifacts","feature_importance_shap_v4.csv"), index=False)
    plot_top_shap_bar(df_shap, os.path.join(args.outdir,"plots","shap_top_bar.png"), topn=20)

    # Interactions
    topk_inter, full_inter = compute_interactions(best, X, y_log, cat_cols, topk=20)
    topk_inter.to_csv(os.path.join(args.outdir,"artifacts","interactions_top20_v4.csv"), index=False)
    full_inter.to_csv(os.path.join(args.outdir,"artifacts","interactions_full_v4.csv"), index=False)
    plot_top_interactions(topk_inter, os.path.join(args.outdir,"plots","interactions_top20.png"))

    # PDP
    pdp_feats = df_shap["feature"].head(args.pdp_topn).tolist()
    plot_pdp(best, X, pdp_feats, os.path.join(args.outdir,"plots","pdp"))

    # Calibration (use best oof)
    if args.hybrid and res.get("cv_hybrid_rmsle") is not None and res["cv_hybrid_rmsle"] <= res["cv_cat_rmsle"]:
        oof = pack["oof_hyb"]; pref = os.path.join(args.outdir,"plots","hybrid")
    else:
        oof = pack["oof_cat"]; pref = os.path.join(args.outdir,"plots","catboost")
    plot_calibration_and_residuals(y_log, oof, pref)

    # Report
    meta = {
        "cv_cat": res["cv_cat_rmsle"],
        "cv_hyb": res.get("cv_hybrid_rmsle"),
        "top_features": list(zip(df_shap["feature"].head(10), df_shap["mean_abs_shap"].head(10))),
        "top_pairs": list(zip(topk_inter["feature_1"].head(10), topk_inter["feature_2"].head(10), topk_inter["interaction_score"].head(10))),
        "pdp_features": pdp_feats
    }
    write_report(os.path.join(args.outdir,"report_v4.txt"), meta)

    print("\n=== DONE ===")
    print(f"Outputs saved under: {args.outdir}")

if __name__ == "__main__":
    main()
