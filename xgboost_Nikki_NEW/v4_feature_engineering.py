# =============== XGBoost v4_feature_engineering ===============
# Feature engineering using existing robust functions
# Uses: robust_na_imputation, add_ordinal_views, add_features
# Overall CV RMSE(log) = 0.12970, R2 = 0.8945
# v3 Score: 0.12670
# v4 Improvement: -0.00300
# Feature engineering did not improve performance
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as psd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import metrics
from pathlib import Path
import json
import sys

# Add the directory to Python path
CATBOOST_DIR = Path("../Catboost_Dylan")
sys.path.append(str(CATBOOST_DIR))

# Now import
from v2_2_plus_fix import robust_na_imputation, add_ordinal_views, add_features

#%% ---------- Load best parameters from v3 ----------
try:
    with open('v3_best_params.json', 'r') as f:
        v3_best_params = json.load(f)
    print("Loaded best parameters from v3:")
    for k, v in v3_best_params.items():
        print(f"  {k}: {v}")
except FileNotFoundError:
    print("ERROR: v3_best_params.json not found. Run v3 first!")
    exit()

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"Original data - Train: {train.shape}, Test: {test.shape}")

#%% ---------- Feature Engineering Preprocessing -----------
def feature_engineering_preprocess(train_df, test_df, target_col='SalePrice'):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Combine for consistent engineering
    combined = pd.concat([train_df, test_df], ignore_index=True)
    
    # Apply your robust preprocessing pipeline
    print("Applying robust_na_imputation...")
    combined = robust_na_imputation(combined)
    
    print("Adding ordinal views...")
    combined = add_ordinal_views(combined)
    
    print("Adding engineered features...")
    combined = add_features(combined)
    
    # Use label encoding for ALL categoricals
    categorical_cols = combined.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Split back
    train_final = combined.iloc[:len(train_df)].copy()
    test_final = combined.iloc[len(train_df):].copy()
    
    print(f"Feature engineering complete.")
    print(f"Original features: {len(train_df.columns)}, New features: {len(train_final.columns)}")
    
    return train_final, test_final

# Apply preprocessing
train_clean, test_clean = feature_engineering_preprocess(train, test)

#%% -------- Prepare Features -----------
y = np.log1p(train_clean["SalePrice"])
X = train_clean.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test_clean.drop(columns=["SalePrice", "Id"], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")

#%% ---------- Cross Validation with V3 TUNED PARAMETERS ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # USE V3 TUNED PARAMETERS
    model = xgb.XGBRegressor(**v3_best_params)
    
    model.fit(X_tr, y_tr)
    
    oof_pred[va_idx] = model.predict(X_va)
    test_pred += model.predict(X_test) / kf.n_splits
    
    rmse_log = np.sqrt(metrics.mean_squared_error(y_va, oof_pred[va_idx]))
    fold_scores.append(rmse_log)
    print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}")

# ---------- Final CV Score ----------
cv_rmse_log = np.sqrt(metrics.mean_squared_error(y, oof_pred))
r2 = metrics.r2_score(y, oof_pred)
print(f"\n=== v4_feature_engineering Results ===")
print(f"Overall CV RMSE(log) = {cv_rmse_log:.5f}, R2 = {r2:.4f}")
print(f"Fold std: {np.std(fold_scores):.5f}")

# Compare with v3
v3_score = 0.12670
improvement = v3_score - cv_rmse_log
print(f"v3 Score: {v3_score:.5f}")
print(f"v4 Improvement: {improvement:.5f}")

if improvement > 0:
    print("SUCCESS: Feature engineering improved performance!")
else:
    print("Feature engineering did not improve performance")

#%% ---------- Save Results ----------
submission = pd.DataFrame({
    "Id": test_clean["Id"],
    "SalePrice": np.expm1(test_pred)
})
submission.to_csv("xgb_v4_feature_eng.csv", index=False)
print("Saved xgb_v4_feateng.csv")