"""This script plots creates the plot with the estimates for individual questions."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

sns.set_style("white")
cmap_choice = 'vlag'

dataset = 'nn'
files = []
# We choose the best teacher per dataset
if dataset == 'aml':
    files = ['gemini_LLM_single-shot_2-5', "teacher_2"]
elif dataset == 'nn':
    files = ['gemini_LLM_single-shot_2-5', "teacher_3"]

# Copy over the id and correct answer rate from the first file
combined_results = pd.read_csv(f'results/{dataset}_{files[0]}.csv')
combined_results = combined_results[['id', 'Correct_Answer_Rate']]

for file_name in files:
    df = pd.read_csv(f'results/{dataset}_{file_name}.csv')[['id', 'estimate']]
    # Remove the 1st row, as it's the example question
    df = df.iloc[1:]
    if 'teacher' in file_name:
        df = df.rename(columns={'estimate': 'Best_Professor'})
    df = df.rename(columns={'estimate': file_name})
    if combined_results.empty:
        combined_results = df
    else:
        combined_results = pd.merge(combined_results, df, on='id', how='outer')


all_ids = combined_results['id'].tolist()

best_svm_results = pd.read_csv(
    f'results/{dataset}_best_SVM.csv')[['id', 'estimate']]
best_svm_ids_test_split = best_svm_results['id'].tolist()
# We add the results to the combined_results dataframe matching the id column
best_svm_results = best_svm_results.rename(columns={'estimate': 'best_svm'})
combined_results = pd.merge(
    combined_results, best_svm_results, on='id', how='outer')

train_split = combined_results[combined_results['best_svm'].isna()]
test_split = combined_results[~combined_results['best_svm'].isna()]

train_split = train_split.sort_values(
    by='Correct_Answer_Rate', ascending=False)
test_split = test_split.sort_values(by='Correct_Answer_Rate', ascending=False)

############################################# PLOTTING #############################################
value_columns = [
    'Correct_Answer_Rate',
    'gemini_LLM_single-shot_2-5',
    'Best_Professor',
    'best_svm'
]

labels = [
    'Actual Rate',
    'Gemini 2.5 (Per question)',
    'Best Professor',
    'Best Supervised Learning Model'
]

colors = plt.cm.tab10(np.linspace(0, 1, len(value_columns)))
markers = ['*', 'o', '^', 'D']  # Unique markers

actual_rate_size = 100
other_marker_size = 80

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(20, 5),  # Wider figure
    sharey=True,
    gridspec_kw={'width_ratios': [len(train_split), len(test_split)]}
)

fig.subplots_adjust(wspace=0.05)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

x_train = np.arange(len(train_split))
for i, col in enumerate(value_columns):
    current_label = labels[i]
    if current_label == 'Actual Rate':
        plot_color = 'goldenrod'  # Golden color for Actual Rate
        plot_size = actual_rate_size
    else:
        plot_color = colors[i]
        plot_size = other_marker_size

    ax1.scatter(
        x_train,
        train_split[col],
        label=current_label,
        color=plot_color,
        marker=markers[i],
        alpha=0.8,
        s=plot_size
    )

ax1.set_facecolor('lightcyan')
ax2.set_facecolor('peachpuff')

ax1.set_ylabel('(Estimated) $P^+$-value', fontsize=28)

ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax1.tick_params(axis='y', labelsize=23)
ax1.set_xticks([])

x_test = np.arange(len(test_split))
for i, col in enumerate(value_columns):
    current_label = labels[i]
    if current_label == 'Actual Rate':
        plot_color = 'goldenrod'
        plot_size = actual_rate_size
    else:
        plot_color = colors[i]
        plot_size = other_marker_size

    ax2.scatter(
        x_test,
        test_split[col],
        color=plot_color,
        marker=markers[i],
        alpha=0.8,
        s=plot_size
    )

ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.tick_params(axis='y', labelsize=23)
ax2.set_xticks([])


handles, labels_from_plot = ax1.get_legend_handles_labels()
ordered_handles = []
ordered_labels = []
label_to_handle = dict(zip(labels_from_plot, handles))
for label in labels:
    if label in label_to_handle:
        ordered_handles.append(label_to_handle[label])
        ordered_labels.append(label)

for handle in ordered_handles:
    label = handle.get_label()
    if label == 'Actual Rate':
        handle.set_sizes([actual_rate_size])
    else:
        handle.set_sizes([other_marker_size])

xcords, ycords = [], []

# This is done manually to ensure the line is drawn correctly
if dataset == 'aml':
    xcords = [0.801, 0.801]
    ycords = [0.12, 0.925]
elif dataset == 'nn':
    xcords = [0.804, 0.804]
    ycords = [0.121, 0.924]

line = mlines.Line2D(
    xcords,
    ycords,
    transform=fig.transFigure,  # Use figure coordinate system
    figure=fig,
    color='gray',      # Customize line color
    linewidth=2,        # Customize line width
    linestyle='--',     # Customize line style (e.g., '--', ':', '-.')
    clip_on=False,      # Allow drawing outside axes boundaries
    zorder=10           # Make sure it's drawn on top
)

fig.lines.append(line)  # Add the line to the figure

fig.text(0.5, 0.05, 'Individual Questions',
         ha='center', va='center', fontsize=28)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])

plt.show()

if not os.path.exists('plots'):
    os.makedirs('plots')

fig.savefig(f'plots/{dataset}_estimates_short.pdf', bbox_inches='tight')

print(f"Plot saved to plots/{dataset}_estimates_short.pdf")
