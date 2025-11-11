# lubaba_gbr/models.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .features import get_column_groups, apply_ordinal_mapping


def build_preprocessor(X_full):
    numeric_cols, ordinal_cols, nominal_cols = get_column_groups(X_full)

    # Ordinal mapping
    X_full_mapped = apply_ordinal_mapping(X_full, ordinal_cols)

    numeric_preproc = Pipeline([
        ("impute", SimpleImputer(strategy="median"))
    ])

    ordinal_preproc = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent"))
    ])

    nominal_preproc = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preproc, numeric_cols),
            ("ord", ordinal_preproc, ordinal_cols),
            ("nom", nominal_preproc, nominal_cols),
        ],
        remainder="drop"
    )

    # Return both: mapped data + preprocessor + col groups (for potential reuse)
    return X_full_mapped, preprocessor, numeric_cols, ordinal_cols, nominal_cols


def tune_and_select_model(X_tr, y_tr_log, X_val, y_val_log, y_val_raw, preprocessor):
    results = []

    # === Random Forest (small grid, 3-fold) ===
    rf_base = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf_pipeline = Pipeline([
        ("prep", preprocessor),
        ("rf", rf_base)
    ])

    rf_param_grid = {
        "rf__max_depth": [15, None],
        "rf__min_samples_leaf": [2, 5],
    }

    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    rf_grid.fit(X_tr, y_tr_log)
    best_rf = rf_grid.best_estimator_

    y_pred_log = best_rf.predict(X_val)
    y_pred = np.exp(y_pred_log)

    rf_rmse = mean_squared_error(y_val_raw, y_pred) ** 0.5
    rf_mae = mean_absolute_error(y_val_raw, y_pred)
    rf_rmsle = mean_squared_error(y_val_log, y_pred_log) ** 0.5

    results.append(("RF", rf_grid.best_params_, rf_rmse, rf_mae, rf_rmsle, best_rf))

    # === Gradient Boosting (small grid, 3-fold) ===
    gbr_base = GradientBoostingRegressor(random_state=42)

    gbr_pipeline = Pipeline([
        ("prep", preprocessor),
        ("gbr", gbr_base)
    ])

    gbr_param_grid = {
        "gbr__n_estimators": [200, 300],
        "gbr__learning_rate": [0.05, 0.1],
        "gbr__max_depth": [2, 3],
    }

    gbr_grid = GridSearchCV(
        estimator=gbr_pipeline,
        param_grid=gbr_param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    gbr_grid.fit(X_tr, y_tr_log)
    best_gbr = gbr_grid.best_estimator_

    y_pred_log = best_gbr.predict(X_val)
    y_pred = np.exp(y_pred_log)

    gbr_rmse = mean_squared_error(y_val_raw, y_pred) ** 0.5
    gbr_mae = mean_absolute_error(y_val_raw, y_pred)
    gbr_rmsle = mean_squared_error(y_val_log, y_pred_log) ** 0.5

    results.append(("GBR", gbr_grid.best_params_, gbr_rmse, gbr_mae, gbr_rmsle, best_gbr))

    # === Select best by RMSE in dollars ===
    best_idx = np.argmin([r[2] for r in results])
    best_type, best_cfg, best_rmse, best_mae, best_rmsle, best_pipeline = results[best_idx]

    return best_type, best_cfg, best_rmse, best_mae, best_rmsle, best_pipeline
