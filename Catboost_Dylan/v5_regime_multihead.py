
"""
v5_regime_multihead.py
Router + specialized CatBoost heads trained per regime (size/type).

Outputs: ./outputs_v5/multihead/summary.json, scoreboard.csv
"""

import os, json
import numpy as np
import pandas as pd
from collections import defaultdict

from v5_shared import (
    load_train, feature_engineering, run_catboost_cv, catboost_default_params,
    ensure_dir, save_json, scoreboard_row, rmsle
)

SEED = 19981124

def router_rules(df: pd.DataFrame) -> pd.Series:
    """Deterministic routing based on TotalSF and MSSubClass family.
    Robust to duplicate 'MSSubClass' columns (picks the first) and missing values.
    """
    # Size buckets from TotalSF (fallback to S if missing)
    if "TotalSF" in df.columns:
        size = pd.cut(df["TotalSF"], bins=[-1, 1500, 2500, 4000, 99999], labels=["S","M","L","XL"])
    else:
        size = pd.Series(["S"] * len(df), index=df.index, dtype="object")

    # Safely get exactly one MSSubClass column
    ms_cols = [c for c in df.columns if c == "MSSubClass"]
    if len(ms_cols) == 0:
        ms = pd.Series(["UNK"] * len(df), index=df.index, dtype="object")
    else:
        ms = df[ms_cols[0]]

    # Coarse family mapping
    ms_str = ms.astype(str)
    ms_digits = ms_str.str.extract(r'(\d+)', expand=False).fillna(ms_str)
    family = ms_digits.replace({
        "20":"SF","60":"SF",
        "70":"Old","75":"Old",
        "80":"Split","85":"Split",
        "90":"Others","120":"Duplex",
        "150":"Others","160":"Others",
        "180":"PUD","190":"Others"
    })
    regime = size.astype(str) + "_" + family.astype(str)
    return regime

def main():
    outdir = "./outputs_v5/multihead"
    ensure_dir(outdir)

    df = load_train("train.csv")
    y = df["SalePrice"]
    X, num_cols, cat_cols, _ = feature_engineering(df, y)

    # Ensure we have a single MSSubClass available without duplicating columns
    X_tmp = X.copy()
    if "MSSubClass" not in X_tmp.columns:
        X_tmp["MSSubClass"] = df["MSSubClass"]
    reg = router_rules(X_tmp)

    Xr = X.copy()
    Xr["REGIME"] = reg.astype(str)
    cat_cols2 = list(dict.fromkeys(list(cat_cols) + ["REGIME"]))

    # Single model with regime as a categorical (strong baseline)
    oof_single, folds_single, meta_single = run_catboost_cv(
        Xr, y, cat_cols=cat_cols2, use_group_kfold=None, groups=None,
        params=catboost_default_params(), label="single_with_regime"
    )
    base_cv = meta_single["summary"]["cv_rmsle"]

    # Multi-head: train separate models per top regimes, fallback to single-with-regime OOF
    regime_counts = Xr["REGIME"].value_counts()
    top_regimes = regime_counts.index[:6].tolist()
    mask_map = {r: (Xr["REGIME"] == r).values for r in top_regimes}

    oof_multi = np.zeros(len(y), dtype=float)
    for r in top_regimes:
        idx = np.where(mask_map[r])[0]
        if len(idx) < 60:
            continue
        X_sub = X.iloc[idx].copy()
        y_sub = y.iloc[idx].copy()
        # heads specialize on native cats; do NOT include REGIME inside heads
        oof_r, _, _ = run_catboost_cv(
            X_sub, y_sub, cat_cols=cat_cols, use_group_kfold=None, groups=None,
            params=catboost_default_params(), label=f"head_{r}"
        )
        oof_multi[idx] = oof_r

    # Fallback: for indices not covered (oof=0), use single-with-regime oof
    fallback_mask = (oof_multi == 0)
    oof_multi[fallback_mask] = oof_single[fallback_mask]

    cv_rmsle = rmsle(y.values, oof_multi)

    summary = {
        "single_with_regime_cv": float(base_cv),
        "multihead_cv": float(cv_rmsle),
        "delta": float(base_cv - cv_rmsle),
        "notes": "Router by size/type regimes; specialized heads for top buckets."
    }
    save_json(summary, os.path.join(outdir, "summary.json"))
    pd.DataFrame([scoreboard_row("v5_multihead", cv_rmsle, "Specialists per regime + fallback.")]).to_csv(
        os.path.join(outdir, "scoreboard.csv"), index=False
    )

    print(f"[V5 Multi-head] Single-with-regime CV: {base_cv:.6f} -> Multi-head: {cv_rmsle:.6f} (Δ={base_cv-cv_rmsle:+.6f})")
    print("\nBenefits:\n - Lets models specialize for structurally different homes.\n - Keeps a strong single-model fallback.")
    print("Drawbacks:\n - Smaller buckets risk overfit; keep heads small.\n - Slightly more orchestration at inference.")

if __name__ == "__main__":
    main()
