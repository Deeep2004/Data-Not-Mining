# =============== XGBoost V1 ===============
# First version of XGBoost model
# Base Model
# Overall CV: RMSE(log): 0.14126, R2 = 0.8748
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from pathlib import Path

# ---------- Load data ----------
train = pd.read_csv("xgb_train_clean.csv")
test = pd.read_csv("xgb_test_clean.csv")

#%% -------- Prepare Features -----------
y = np.log1p(train["SalePrice"])
X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test.drop(columns=["SalePrice", "Id"], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")

#%% ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))  # Out-of-fold predictions for training data
test_pred = np.zeros(len(X_test))  # Ensemble predictions for test data

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Split data
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # Create and train model
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.08, 
        gamma=0, 
        subsample=0.75,
        colsample_bytree=1,
        max_depth=7,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    # Store out-of-fold predictions
    oof_pred[va_idx] = model.predict(X_va)
    
    # Add to ensemble test predictions
    test_pred += model.predict(X_test) / kf.n_splits
    
    # Calculate fold performance
    rmse_log = np.sqrt(metrics.mean_squared_error(y_va, oof_pred[va_idx]))
    print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}")

# ---------- Final CV Score ----------
cv_rmse_log = np.sqrt(metrics.mean_squared_error(y, oof_pred))
r2 = metrics.r2_score(y, oof_pred)
print(f"Overall CV RMSE(log) = {cv_rmse_log:.5f}, R2 = {r2:.4f}")

#%% ---------- Submission to CSV ----------
# Convert back from log to actual prices
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": np.expm1(test_pred)
})

#out_path = Path(__file__).resolve().parent / "xgb_v1_output.csv"
#submission.to_csv(out_path, index=False)
#print("Saved xgb_v1_output.csv to", out_path)
