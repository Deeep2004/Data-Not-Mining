# =============== FEATURES ===============

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Load data ----------
DATA_DIR = Path("../data")
train = pd.read_csv(DATA_DIR / "train.csv")


# %% -------- Preprocessing -----------
def xgboost_preprocess(df):
    df = df.copy()
    
    # Handle object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    
    for col in obj_cols:
        df[col] = df[col].fillna("None")

        # Label encoding (>10 categories)
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

cleaned = xgboost_preprocess(train)
print(cleaned.head())


# %% numeric only

train_non_object = train.select_dtypes(exclude=['object'])

cor2=train_non_object.corr()['SalePrice'].sort_values(ascending=False).drop('SalePrice')

plt.figure(figsize=(8,6))
plt.bar(x=list(cor2.index), height=list(cor2.values), color='teal')
plt.xticks(rotation=90)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.title('Correlation of Numeric Features with Price', fontsize=15)
plt.show()

# %% object only

def correlation_ratio(categories, values):
    categories = pd.Categorical(categories)
    cat_values = [values[categories == cat] for cat in categories.categories]
    n_total = len(values)
    grand_mean = np.mean(values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in cat_values)
    ss_total = sum((values - grand_mean)**2)
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0

# compute eta for each object column
obj_cols = train.select_dtypes(include=['object']).columns
etas = {col: correlation_ratio(train[col].fillna('None'), train['SalePrice'].values) for col in obj_cols}
etas_series = pd.Series(etas).sort_values(ascending=False)
print(etas_series)

# plot eta (correlation ratio) for categorical features
plt.figure(figsize=(10,6))
etas_series.plot(kind='bar', color='teal')
plt.xticks(rotation=90)
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Correlation ratio (eta)', fontsize=12)
plt.title('Categorical features association with SalePrice (eta)', fontsize=14)
plt.tight_layout()
plt.show()

# %% heatmap
import seaborn as sns

matrix = train_non_object.corr()

plt.figure(figsize=(18,18))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# %%
