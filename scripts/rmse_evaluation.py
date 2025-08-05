"""Script to calculate the RMSE and correlation of system estimates against gold labels."""
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import numpy as np
from scipy import stats

dataset = "aml"  # or "nn"
# or e.g., dataset + "_teacher_3"
FILE_NAME = dataset + "_gemini_LLM_single-shot_2-5"
GOLD_LABEL_COLUMN = "Correct_Answer_Rate"
SYSTEM_LABEL_COLUMN = "estimate"  # or "gemini_direct_estimate"

file_path = "results/" + FILE_NAME + ".csv"

df = pd.read_csv(file_path)

df = df.drop(index=0)  # delete the first row, as it's the example given
df = df.dropna(subset=[SYSTEM_LABEL_COLUMN])

rmse = root_mean_squared_error(
    df[GOLD_LABEL_COLUMN].values,
    df[SYSTEM_LABEL_COLUMN].values,
)
spearman_corr = stats.spearmanr(
    df[GOLD_LABEL_COLUMN].values,
    df[SYSTEM_LABEL_COLUMN].values,
)

mean_error = df[SYSTEM_LABEL_COLUMN].values - df[GOLD_LABEL_COLUMN].values
mean_error = np.mean(mean_error)

print(f"RMSE: {rmse:.3f}, Error: {mean_error:.3f}, rho: {spearman_corr.statistic:.3f}(p= {spearman_corr.pvalue:.3f}), n= {len(df)}")
