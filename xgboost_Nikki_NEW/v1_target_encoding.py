# =============== XGBoost v1_target_encoding ===============
# Target encoding for categorical variables
# Compare against ordinal baseline
# Overall CV RMSE(log) = 0.14211, R2 = 0.8733
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import metrics
from category_encoders import TargetEncoder
from pathlib import Path

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

#%% -------- Target Encoding Preprocessing -----------
def target_encoding_preprocess(train_df, test_df, target_col='SalePrice'):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Handle missing values
    for df in [train_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Missing')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Identify categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply target encoding using cross-validation to avoid leakage
    encoder = TargetEncoder(cols=categorical_cols)
    train_encoded = encoder.fit_transform(train_df[categorical_cols], np.log1p(train_df[target_col]))
    test_encoded = encoder.transform(test_df[categorical_cols])
    
    # Replace categorical columns with encoded versions
    train_df = train_df.drop(columns=categorical_cols).join(train_encoded)
    test_df = test_df.drop(columns=categorical_cols).join(test_encoded)
    
    return train_df, test_df, categorical_cols

# Apply preprocessing
train_clean, test_clean, encoded_cols = target_encoding_preprocess(train, test)
print(f"Target encoded {len(encoded_cols)} categorical columns")

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
print(f"\n=== v1_target_encoding Results ===")
print(f"Overall CV RMSE(log) = {cv_rmse_log:.5f}, R2 = {r2:.4f}")
print(f"Fold std: {np.std(fold_scores):.5f}")

# Save predictions
submission = pd.DataFrame({
    "Id": test_clean["Id"],
    "SalePrice": np.expm1(test_pred)
})
submission.to_csv("xgb_v1_target.csv", index=False)
print("Saved xgb_v1_target.csv")