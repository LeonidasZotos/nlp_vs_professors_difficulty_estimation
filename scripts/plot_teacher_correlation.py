"""Script to plot the correlation between different estimators."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

sns.set_style("white")
cmap_choice = 'viridis'

column = 'estimate'
dataset = 'aml'
combined_teachers = pd.DataFrame()

for teacher in ['1', '2', '3']:
    teacher_df = pd.read_csv(f'results/{dataset}_teacher_{teacher}.csv')
    teacher_df = teacher_df[[column, 'id']]
    teacher_df = teacher_df.rename(
        columns={'estimate': f'Professor {teacher}'})
    combined_teachers = pd.concat([combined_teachers, teacher_df], axis=1)

gemini_df_1 = pd.read_csv(f'results/{dataset}_gemini_LLM_single-shot_2-0.csv')
gemini_df_1 = gemini_df_1[[column, 'id']]
gemini_df_1 = gemini_df_1.rename(
    columns={'estimate': 'gemini_LLM_single-shot_2-0'})
combined_teachers = pd.concat([combined_teachers, gemini_df_1], axis=1)
gemini_df_2 = pd.read_csv(f'results/{dataset}_gemini_LLM_single-shot_2-5.csv')
gemini_df_2 = gemini_df_2[[column, 'id']]
gemini_df_2 = gemini_df_2.rename(
    columns={'estimate': 'gemini_LLM_single-shot_2-5'})
combined_teachers = pd.concat([combined_teachers, gemini_df_2], axis=1)

before_removal = combined_teachers.shape[0]
combined_teachers = combined_teachers.dropna(
    subset=[f'Professor {teacher}' for teacher in ['1', '2', '3']])
after_removal = combined_teachers.shape[0]
print(f"Removed {before_removal - after_removal} rows with NaN values.")
combined_teachers = combined_teachers.drop(columns=['id'])
combined_teachers = combined_teachers.rename(columns={
    'gemini_LLM_single-shot_2-0': 'Gemini 2.0',
    'gemini_LLM_single-shot_2-5': 'Gemini 2.5'
})


corr = combined_teachers.corr(method='spearman')
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
ax = sns.heatmap(corr,
                 annot=True,           # Show values in cells
                 mask=mask,            # Mask the upper triangle
                 fmt=".2f",            # Format values to 2 decimal places
                 cmap=cmap_choice,     # Color map
                 linecolor='lightgray',  # Color of the lines
                 cbar=False,           # Show color bar
                 annot_kws={"size": 20}
                 )


ax.tick_params(axis='x', labelsize=25)  # Adjust the x-axis tick label size
ax.tick_params(axis='y', labelsize=25)  # Adjust the y-axis tick label size

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'plots/{dataset}_teacher_correlation.pdf')
