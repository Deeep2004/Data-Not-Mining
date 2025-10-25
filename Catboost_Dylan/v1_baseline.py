# =============== CATBOOST V1 BASELINE ===============
# A clean first version of CatBoost model with CV
# Overall CV: RMSE(log)=0.12663, RMSLE=0.12663
# ----------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool
from pathlib import Path
import math

# ---------- Helper ----------
def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 1, None)
    y_pred = np.clip(y_pred, 1, None)
    return math.sqrt(mean_squared_log_error(y_true, y_pred))

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# ---------- Minimal cleaning ----------
def minimal_catboost_clean(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df[obj_cols] = df[obj_cols].fillna("None")
    return df

train_clean = minimal_catboost_clean(train)
test_clean  = minimal_catboost_clean(test)

# ---------- Prepare features ----------
y = np.log1p(train_clean["SalePrice"])
X = train_clean.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test_clean[X.columns]

cat_idx = [i for i, c in enumerate(X.columns)
           if str(X[c].dtype) in ("object", "category")]

# ---------- Model setup ----------
params = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    learning_rate=0.05,
    depth=6,
    iterations=2000,
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    verbose=False
)

# ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== Fold {fold} ===")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, label=y_va, cat_features=cat_idx)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool)

    # store predictions
    oof_pred[va_idx] = model.predict(valid_pool)
    test_pred += np.expm1(model.predict(X_test)) / kf.n_splits

    rmse_log = np.sqrt(np.mean((y_va - oof_pred[va_idx])**2))
    rmsle_lin = rmsle(np.expm1(y_va), np.expm1(oof_pred[va_idx]))
    print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}, RMSLE={rmsle_lin:.5f}")

# ---------- Final CV score ----------
cv_rmse_log = np.sqrt(np.mean((y - oof_pred)**2))
cv_rmsle_lin = rmsle(np.expm1(y), np.expm1(oof_pred))
print(f"\nOverall CV: RMSE(log)={cv_rmse_log:.5f}, RMSLE={cv_rmsle_lin:.5f}")

# # ---------- Submission ----------
# submission = pd.DataFrame({
#     "Id": test["Id"],
#     "SalePrice": test_pred
# })
# submission.to_csv("submission_catv1.csv", index=False)
# print("\nSaved submission_catv1.csv")

# =============== END ===============
