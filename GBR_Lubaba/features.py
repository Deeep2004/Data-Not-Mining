# lubaba_gbr/features.py
import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalSF
    if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
        df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    else:
        df["TotalSF"] = np.nan

    # TotalBaths
    full = df["FullBath"] if "FullBath" in df.columns else 0
    half = df["HalfBath"] if "HalfBath" in df.columns else 0
    df["TotalBaths"] = full + 0.5 * half

    # HouseAge
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    elif "YearBuilt" in df.columns:
        df["HouseAge"] = 2010 - df["YearBuilt"]
    else:
        df["HouseAge"] = np.nan

    # GarageExists
    if "GarageArea" in df.columns:
        df["GarageExists"] = (df["GarageArea"] > 0).astype(int)
    else:
        df["GarageExists"] = 0

    # Remodeled
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
        df["Remodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    else:
        df["Remodeled"] = 0

    # AgeSinceRemodel
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        age = df["YrSold"] - df["YearRemodAdd"]
        df["AgeSinceRemodel"] = age.clip(lower=0)
    elif "YearRemodAdd" in df.columns:
        df["AgeSinceRemodel"] = 2010 - df["YearRemodAdd"]
    else:
        df["AgeSinceRemodel"] = np.nan

    # BasementFinishedRatio
    if "BsmtFinSF1" in df.columns and "TotalBsmtSF" in df.columns:
        df["BasementFinishedRatio"] = df["BsmtFinSF1"] / df["TotalBsmtSF"]
    else:
        df["BasementFinishedRatio"] = np.nan

    # HasFireplace
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    else:
        df["HasFireplace"] = 0

    # QualxSF
    if "OverallQual" in df.columns:
        if "TotalSF" in df.columns:
            df["QualxSF"] = df["OverallQual"] * df["TotalSF"]
        elif "GrLivArea" in df.columns:
            df["QualxSF"] = df["OverallQual"] * df["GrLivArea"]
        else:
            df["QualxSF"] = df["OverallQual"]
    else:
        df["QualxSF"] = np.nan

    return df


def get_column_groups(X: pd.DataFrame):
    """Return numeric_cols, ordinal_cols, nominal_cols."""
    ordinal_cols = [c for c in ["ExterQual", "BsmtQual", "KitchenQual"] if c in X.columns]

    candidate_nominals = [
        "Neighborhood", "HouseStyle", "MSZoning", "RoofMatl",
        "Exterior1st", "Exterior2nd", "Street", "Alley", "LotShape",
        "LandContour", "Utilities", "LotConfig", "LandSlope",
        "Condition1", "Condition2", "BldgType", "Foundation",
        "Heating", "HeatingQC", "CentralAir", "Electrical",
        "Functional", "FireplaceQu", "GarageType", "GarageFinish",
        "GarageQual", "GarageCond", "PavedDrive", "PoolQC",
        "Fence", "MiscFeature", "SaleType", "SaleCondition"
    ]
    nominal_cols = [c for c in candidate_nominals if c in X.columns]

    numeric_cols = []
    for col in X.columns:
        if col not in ordinal_cols and col not in nominal_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                numeric_cols.append(col)

    return numeric_cols, ordinal_cols, nominal_cols


def apply_ordinal_mapping(X: pd.DataFrame, ordinal_cols):
    """Map quality strings -> numeric scores in-place-compatible fashion."""
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    X = X.copy()
    for col in ordinal_cols:
        X[col] = X[col].map(quality_map).fillna(0)
    return X
