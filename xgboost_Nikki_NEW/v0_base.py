# =============== XGBoost v0_base ===============
# Simple baseline with ordinal encoding only
# Establishes trustworthy control group
# Overall CV RMSE(log) = 0.14242, R2 = 0.8728
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn import metrics
from pathlib import Path

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

#%% -------- Simple Preprocessing (Ordinal Only) -----------
def simple_ordinal_preprocess(df):
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Missing')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Ordinal encode all categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    return df

# Apply preprocessing
train_clean = simple_ordinal_preprocess(train)
test_clean = simple_ordinal_preprocess(test)

#%% -------- Prepare Features -----------
y = np.log1p(train_clean["SalePrice"])
X = train_clean.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test_clean.drop(columns=["SalePrice", "Id"], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")

#%% ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # SIMPLE baseline model - near default parameters
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    oof_pred[va_idx] = model.predict(X_va)
    test_pred += model.predict(X_test) / kf.n_splits
    
    rmse_log = np.sqrt(metrics.mean_squared_error(y_va, oof_pred[va_idx]))
    fold_scores.append(rmse_log)
    print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}")

# ---------- Final CV Score ----------
cv_rmse_log = np.sqrt(metrics.mean_squared_error(y, oof_pred))
r2 = metrics.r2_score(y, oof_pred)
print(f"\n=== v0_base Results ===")
print(f"Overall CV RMSE(log) = {cv_rmse_log:.5f}, R2 = {r2:.4f}")
print(f"Fold std: {np.std(fold_scores):.5f}")

# Save predictions

submission = pd.DataFrame({
    "Id": test_clean["Id"],
    "SalePrice": np.expm1(test_pred)
})
submission.to_csv("xgb_v0_base.csv", index=False)
print("Saved xgb_v0_base.csv")
