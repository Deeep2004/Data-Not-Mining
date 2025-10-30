# =============== Extra Data Cleaning ===============
# Data preprocessing for XGBoost model
# Handling categorical variables with one-hot encoding and label encoding
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# %% -------- Preprocessing -----------

# Preprocessing Function
def xgboost_preprocess(df):
    df = df.copy()
    
    # Handle object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    
    for col in obj_cols:
        df[col] = df[col].fillna("None")
        
        # One-hot encoding (<10 categories)
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

        # Label encoding (>10 categories)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Combine datasets
combined = pd.concat([train, test], ignore_index=True)

# Apply to combined data
combined_clean = xgboost_preprocess(combined)

# Split processed datasets (train and test)
train_clean = combined_clean[:len(train)]  # First part = training data
test_clean = combined_clean[len(train):]   # Second part = test data

#%% ---------- Process to CSV -------------
out_path = Path(__file__).resolve().parent / "xgb_train_clean.csv"
train_clean.to_csv(out_path, index=False)
print("Saved xgb_train_clean.csv to", out_path)

out_path = Path(__file__).resolve().parent / "xgb_test_clean.csv"
test_clean.to_csv(out_path, index=False)
print("Saved xgb_test_clean.csv to", out_path)