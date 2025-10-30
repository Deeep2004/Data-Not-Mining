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
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

#%% checking number of unique values in object columns
print(train.select_dtypes(include=["object"]).nunique())

#%% count NA values in each column
na_counts = train.isna().sum()
print(na_counts[na_counts > 0].sort_values(ascending=False))

train.info()
# %%
