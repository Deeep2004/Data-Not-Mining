from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

RANDOM_STATE = 42

def main():
    # ---- 1) Load data (relative to repo root) ----
    DATA_DIR = Path("data")
    train_path = DATA_DIR / "train.csv"
    test_path  = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")

    # 2) Split features/target & log-transform y 
    y_log = np.log(train_df["SalePrice"].values)
    X = train_df.drop(columns=["SalePrice"])

    # detect basic dtypes: numerics vs categoricals
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # keep an eye on Id: remove from features if present
    if "Id" in numeric_cols:
        numeric_cols.remove("Id")
    elif "Id" in categorical_cols:
        categorical_cols.remove("Id")

    print(f"Numeric cols: {len(numeric_cols)} | Categorical cols: {len(categorical_cols)}")

    X_tr, X_val, y_tr_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    #3) Minimal preprocessing 
    numeric_preproc = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median"))
    ])

    categorical_preproc = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preproc, numeric_cols),
            ("cat", categorical_preproc, categorical_cols),
        ],
        remainder="drop"
    )

    #  4) Very basic RandomForest on log(SalePrice)
    rf = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("rf", rf)
    ])

    model.fit(X_tr, y_tr_log)

    #  5) Evaluate on validation 
    y_val_pred_log = model.predict(X_val)
    y_val_pred = np.exp(y_val_pred_log)
    y_val = np.exp(y_val_log)

    # Use np.sqrt(MSE) for compatibility across sklearn versions
    rmse_dollars = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_dollars = mean_absolute_error(y_val, y_val_pred)
    rmsle = np.sqrt(mean_squared_error(y_val_log, y_val_pred_log))

    print(f"\n[RandomForest RAW] Validation RMSE($): {rmse_dollars:,.2f}")
    print(f"[RandomForest RAW] Validation MAE($):  {mae_dollars:,.2f}")
    print(f"[RandomForest RAW] Validation RMSLE:   {rmsle:.5f}")

    # 6) Train on all data & predict test 
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_ids = test_df["Id"] if "Id" in test_df.columns else pd.Series(np.arange(len(test_df)))
        # drop Id from features if present
        X_test = test_df.drop(columns=["Id"]) if "Id" in test_df.columns else test_df.copy()

        model.fit(X, y_log)  # refit on full training data
        test_pred_log = model.predict(X_test)
        test_pred = np.exp(test_pred_log)

        sub = pd.DataFrame({"Id": test_ids, "SalePrice": test_pred})
        out_path = Path("submission_rf_basic.csv")
        sub.to_csv(out_path, index=False)
        print(f"\nSubmission written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()