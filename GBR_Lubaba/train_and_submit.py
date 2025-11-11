# GBR_Lubaba/train_and_submit.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from .features import engineer_features, apply_ordinal_mapping
from .models import build_preprocessor, tune_and_select_model


def main():
    # 1. Paths
    data_dir = Path("data")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    # 2. Load
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 3. Feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # 4. Target & base features
    y_raw = train_df["SalePrice"].copy()
    y_log = np.log(y_raw)

    X_full = train_df.drop(columns=["SalePrice"]).copy()
    X_test_full = test_df.copy()

    if "Id" in X_full.columns:
        X_full = X_full.drop(columns=["Id"])
    if "Id" in X_test_full.columns:
        X_test_full = X_test_full.drop(columns=["Id"])

    # 5. Build preprocessor & get column groups
    #    This also returns X_full_mapped where ordinal cols are already numeric.
    X_full_mapped, preprocessor, numeric_cols, ordinal_cols, nominal_cols = build_preprocessor(X_full)

    # 6. Train/validation split on the MAPPED train data
    X_tr, X_val, y_tr_log, y_val_log, y_tr_raw, y_val_raw = train_test_split(
        X_full_mapped, y_log, y_raw,
        test_size=0.2,
        random_state=42
    )

    # 7. Tune RF + GBR and select best (these expect preprocessed/mapped inputs)
    best_type, best_cfg, best_rmse, best_mae, best_rmsle, best_pipeline = tune_and_select_model(
        X_tr, y_tr_log, X_val, y_val_log, y_val_raw, preprocessor
    )

    print(f"\nBest model: {best_type} {best_cfg}")
    print(f"Validation RMSE($): {best_rmse:.2f}")
    print(f"Validation MAE($):  {best_mae:.2f}")
    print(f"Validation RMSLE:   {best_rmsle:.5f}")

    # 8. Retrain best model on ALL mapped training data
    final_model = clone(best_pipeline)
    final_model.fit(X_full_mapped, y_log)

    # 9. Apply SAME ordinal mapping to test data
    #    Use the ordinal_cols discovered from training.
    X_test_mapped = apply_ordinal_mapping(X_test_full, ordinal_cols)

    # 10. Predict on mapped test data using the pipeline
    test_pred_log = final_model.predict(X_test_mapped)
    test_pred = np.exp(test_pred_log)

    # 11. Build submission
    test_ids = test_df["Id"]
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": test_pred
    })

    out_path = Path("GBR_Lubaba") / "submission_lubaba_gbr.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission to {out_path.resolve()}")
    
    # diagnostics plots on validation set
    import matplotlib.pyplot as plt

    # Use the selected best_pipeline on the validation fold
    y_val_pred_log = best_pipeline.predict(X_val)
    y_val_pred = np.exp(y_val_pred_log)
    residuals = y_val_raw - y_val_pred

    # (A) Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_raw, y_val_pred, alpha=0.5)
    lims = [min(y_val_raw.min(), y_val_pred.min()),
            max(y_val_raw.max(), y_val_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.xlabel("Actual SalePrice (validation)")
    plt.ylabel("Predicted SalePrice (validation)")
    plt.title(f"Predicted vs Actual ({best_type})")
    plt.tight_layout()
    plt.show()

    # (B) Residuals vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_val_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted SalePrice (validation)")
    plt.ylabel("Residual = Actual - Predicted")
    plt.title(f"Residuals Plot ({best_type})")
    plt.tight_layout()
    plt.show()

    # (C) Histogram of residuals
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution ({best_type})")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()