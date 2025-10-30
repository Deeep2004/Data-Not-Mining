# =============== XGBoost V2 ===============
# Hyperparameter tuning with Optuna
#
# ------------Hypertuning Results-----------------
# Best RMSE(log): 0.12833828190286342
# Best params: {'n_estimators': 490, 
#               'learning_rate': 0.03804813682808823, 
#               'max_depth': 5, 
#               'subsample': 0.6898021217101118, 
#               'colsample_bytree': 0.8472495306853262, 
#               'gamma': 1.197220994321415e-08}
#
# ------------Cross-Validation Results----------------
# Overall CV RMSE(log) = 0.13239, R2 = 0.8901
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
import optuna

# ---------- Load data ----------
train = pd.read_csv("xgb_train_clean.csv")
test = pd.read_csv("xgb_test_clean.csv")

#%% -------- Prepare Features -----------
y = np.log1p(train["SalePrice"])
X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test.drop(columns=["SalePrice", "Id"], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")

#%% ---------- Bayesian hyperparameter tuning ----------

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 10.0),
        "random_state": 42,
        "verbosity": 0,
    }
    model = xgb.XGBRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # cross_val_score returns negative MSE for 'neg_mean_squared_error'
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
    rmse = np.sqrt(-scores).mean()
    return rmse

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=25)   # change n_trials for more/less search

print("Best RMSE(log):", study.best_value)
print("Best params:", study.best_params)

#%% Prepare parameters for training + Cross Validation
best_params = study.best_params.copy()
best_params.update({"random_state": 42, "verbosity": 0})

#%% ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))  # Out-of-fold predictions for training data
test_pred = np.zeros(len(X_test))  # Ensemble predictions for test data

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Split data
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # Create and train model (with updated params)
    model = xgb.XGBRegressor(**best_params)
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

#out_path = Path(__file__).resolve().parent / "xgb_v2_output.csv"
#submission.to_csv(out_path, index=False)
#print("Saved xgb_v2_output.csv to", out_path)

