# =============== XGBoost v3_hyperparameter_tuning ===============
# Advanced hyperparameter tuning with Optuna + best preprocessing
# Overall CV RMSE(log) = 0.12670, R2 = 0.8993
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from pathlib import Path
import optuna
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Use the best preprocessing from previous experiments
from v2_mixed_encoding import mixed_encoding_preprocess

#%% Load Data

# ---------- Load and preprocess data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

train_clean, test_clean = mixed_encoding_preprocess(train, test)

#%% -------- Prepare Features -----------
y = np.log1p(train_clean["SalePrice"])
X = train_clean.drop(columns=["SalePrice", "Id"], errors="ignore")
X_test = test_clean.drop(columns=["SalePrice", "Id"], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")

#%% ---------- Bayesian Hyperparameter Tuning with Optuna ----------
def objective(trial):
    """Optimization objective for hyperparameter tuning"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Use cross-validation for robust evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X, y, 
        scoring="neg_mean_squared_error", 
        cv=cv, 
        n_jobs=-1,
        error_score='raise'
    )
    
    rmse = np.sqrt(-scores.mean())
    return rmse

print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(
    direction="minimize", 
    sampler=optuna.samplers.TPESampler(seed=42)
)

# Optimize with timeout to prevent runaway trials
study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1 hour

print("\n=== Hyperparameter Tuning Results ===")
print("Best RMSE(log):", study.best_value)
print("Best params:", study.best_params)

#%% ---------- Cross Validation with Optimized Parameters ----------
best_params = study.best_params.copy()
# Ensure required parameters are set
best_params.update({
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_scores = []
models = []  # Store models for later analysis

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    # Create model with optimized parameters
    model = xgb.XGBRegressor(**best_params)
    
    model.fit(X_tr, y_tr)
    
    # Store model and predictions
    models.append(model)
    oof_pred[va_idx] = model.predict(X_va)
    test_pred += model.predict(X_test) / kf.n_splits
    
    rmse_log = np.sqrt(metrics.mean_squared_error(y_va, oof_pred[va_idx]))
    fold_scores.append(rmse_log)
    print(f"Fold {fold}: RMSE(log)={rmse_log:.5f}")

# ---------- Final CV Score ----------
cv_rmse_log = np.sqrt(metrics.mean_squared_error(y, oof_pred))
r2 = metrics.r2_score(y, oof_pred)
print(f"\n=== v3_hyperparameter_tuning Final Results ===")
print(f"Overall CV RMSE(log) = {cv_rmse_log:.5f}, R2 = {r2:.4f}")
print(f"Fold std: {np.std(fold_scores):.5f}")
print(f"Improvement over baseline: Compare with v0_base results")

#%% ---------- Save Results ----------
# Save predictions
submission = pd.DataFrame({
    "Id": test_clean["Id"],
    "SalePrice": np.expm1(test_pred)
})
submission.to_csv("xgb_v3_tuned.csv", index=False)
print("\nSaved xgb_v3_tuned.csv")

# Save best parameters for future use
import json
best_params_serializable = {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                           for k, v in best_params.items()}
with open('v3_best_params.json', 'w') as f:
    json.dump(best_params_serializable, f, indent=2)
print("Saved v3_best_params.json")
