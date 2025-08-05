"""This script complements plot_estimates.py and creates its legend."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines

cmap_choice = 'vlag'

colors = plt.cm.tab10(np.linspace(0, 1, 4))

grid_colors = ['lightcyan', 'peachpuff']  # Colors for the grid backgrounds
grid_labels = ['Train Split', 'Test Split']  # Labels for the grid backgrounds

legend_elements = [
    Line2D([0], [0], color='goldenrod', lw=2,
           linestyle='', marker='*', label='Actual Rate'),
    Line2D([0], [0], color=colors[1],  lw=2, linestyle='',
           marker='o', label='Gemini 2.5 (Per question)'),
    Line2D([0], [0], color=colors[2], lw=2, linestyle='',
           marker='^', label='Best Professor'),
    Line2D([0], [0], color=colors[3], lw=2, linestyle='',
           marker='D', label='Best Supervised Learning Model'),
    Line2D([0], [0], color='lightcyan', lw=2, linestyle='',
           marker='s', label='Train Split', markersize=13),
    Line2D([0], [0], color='peachpuff', lw=2, linestyle='',
           marker='s', label='Test Split', markersize=13),
]

fig_legend = plt.figure(figsize=(4, 1))


legend_obj = fig_legend.legend(
    handles=legend_elements,
    loc='center',
    frameon=False,
    ncol=3
)

separation_line = mlines.Line2D(
    [1, 1],
    [0.12, 0.93],
    transform=fig_legend.transFigure,  # Use figure coordinate system
    figure=fig_legend,
    color='gray',      # Customize line color
    linewidth=1,        # Customize line width
    linestyle='--',     # Customize line style (e.g., '--', ':', '-.')
    clip_on=False,      # Allow drawing outside axes boundaries
    zorder=10           # Make sure it's drawn on top
)

fig_legend.lines.append(separation_line)  # Add the line to the figure
fig_legend.canvas.draw()
bbox = legend_obj.get_window_extent().transformed(
    fig_legend.dpi_scale_trans.inverted())
fig_legend.savefig(
    'plots/shared_legend.pdf',
    dpi="figure",
    bbox_inches=bbox,
    pad_inches=0.05
)

plt.close(fig_legend)
