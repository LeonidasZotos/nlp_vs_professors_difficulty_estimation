"""Script to calculate the error and correlation of the average estimates"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
from scipy import stats

DATASET = "aml"  # or "nn"

GOLD_LABEL_COLUMN = "Correct_Answer_Rate"
SYSTEM_LABEL_COLUMN = "estimate"

dfs = []
for teacher in [2, 3]:
    temp_df = pd.read_csv(f"results/{DATASET}_teacher_{teacher}.csv")
    temp_df = temp_df.rename(columns={'estimate': f"estimate_{teacher}"})
    dfs.append(temp_df)

df = pd.concat(dfs, axis=1)
df = df.loc[:, ~df.columns.duplicated()]
df = df.iloc[1:]

df["avg"] = df[[f"estimate_{i}" for i in range(2, 4)]].mean(axis=1)

rmse = root_mean_squared_error(
    df[GOLD_LABEL_COLUMN].values,
    df["avg"].values,
)
spearman_corr = stats.spearmanr(
    df[GOLD_LABEL_COLUMN].values,
    df["avg"].values,
)

mean_error = df["avg"].values - df[GOLD_LABEL_COLUMN].values
mean_error = np.mean(mean_error)

print(f"RMSE: {rmse:.3f}, Error: {mean_error:.3f}, rho: {spearman_corr.statistic:.3f}, n= {len(df)}")
