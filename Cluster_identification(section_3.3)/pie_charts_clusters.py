

import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from matplotlib.patches import Patch

# Read the CSV file
df = pd.read_csv("strategy_dynamics.csv")
#
# If you need to combine datasets
# df2 = pd.read_csv("strategy_dynamics_2D_T=2.csv")
# # Merge on a common column (change 'ID' to the actual column name)
# df_combined = pd.merge(df, df2)  # Use 'outer' for all row
# # Save the combined file
# df_combined.to_csv("combined.csv", index=False)
# df_combined = pd.concat([df, df2], ignore_index=True)
# df_combined.to_csv("combined.csv", index=False)
# df = pd.read_csv("combined.csv")


# Filter the DataFrame for a specific time step (e.g., time_step == 29)
df_filtered = df[df['time_step'] == 99]

# Get sorted unique values of T (rounded to 2 decimals)
unique_T_values = sorted(round(T, 2) for T in df_filtered['T'].unique())

# Initialize dictionaries to store the average percentages for stable and unstable clusters for each T
average_stable_percentages = {}
average_unstable_percentages = {}

# Also, collect all unique cluster types (for consistent colors)
all_cluster_types = set()
all_unstable_cluster_types = set()

# First, iterate to collect all unique cluster types from stable and unstable clusters
for T in unique_T_values:
    df_T = df_filtered[round(df_filtered['T'], 2) == T]
    for _, row in df_T.iterrows():
        # Parse stable cluster details
        stable_details = ast.literal_eval(row['cooperator_cluster_details'])
        all_cluster_types.update(stable_details.keys())
        # Parse unstable cluster details
        unstable_details = ast.literal_eval(row['unstable_cooperator_cluster_details'])
        all_unstable_cluster_types.update(unstable_details.keys())

# Create consistent color mappings for stable and unstable clusters
colors_stable = plt.cm.get_cmap("tab10", len(all_cluster_types))
stable_color_map = {cluster_type: colors_stable(i) for i, cluster_type in enumerate(sorted(all_cluster_types))}

colors_unstable = plt.cm.get_cmap("Set3", len(all_unstable_cluster_types))
unstable_color_map = {cluster_type: colors_unstable(i) for i, cluster_type in enumerate(sorted(all_unstable_cluster_types))}

# Custom autopct function: only display percentages >= 2.5%
def autopct_func(pct):
    return ('%1.1f%%' % pct) if pct >= 2.5 else ''

# Compute average percentages for each T
for T in unique_T_values:
    df_T = df_filtered[round(df_filtered['T'], 2) == T]

    # For stable clusters
    stable_total_counts = {}
    stable_total = 0
    for _, row in df_T.iterrows():
        stable_details = ast.literal_eval(row['cooperator_cluster_details'])
        for cluster_type, count in stable_details.items():
            stable_total_counts[cluster_type] = stable_total_counts.get(cluster_type, 0) + count
        stable_total += sum(stable_details.values())

    if stable_total > 0:
        average_stable_percentages[T] = {ct: (count / stable_total) * 100 for ct, count in stable_total_counts.items()}
    else:
        average_stable_percentages[T] = {}

    # For unstable clusters
    unstable_total_counts = {}
    unstable_total = 0
    for _, row in df_T.iterrows():
        unstable_details = ast.literal_eval(row['unstable_cooperator_cluster_details'])
        for cluster_type, count in unstable_details.items():
            unstable_total_counts[cluster_type] = unstable_total_counts.get(cluster_type, 0) + count
        unstable_total += sum(unstable_details.values())

    if unstable_total > 0:
        average_unstable_percentages[T] = {ct: (count / unstable_total) * 100 for ct, count in unstable_total_counts.items()}
    else:
        average_unstable_percentages[T] = {}

# Set up subplot grid: fewer columns for larger plots
num_T = len(unique_T_values)
cols = 3  # Fewer columns to make pies bigger
rows = (num_T // cols) + (num_T % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))  # Bigger figure
axes = axes.flatten()

# Generate donut charts for each T value
for i, T in enumerate(unique_T_values):
    ax = axes[i]

    # Get stable data for T
    stable_data = average_stable_percentages[T]
    stable_labels = list(stable_data.keys())
    stable_sizes = list(stable_data.values())
    stable_colors = [stable_color_map[label] for label in stable_labels]

    # Get unstable data for T
    unstable_data = average_unstable_percentages[T]
    unstable_labels = list(unstable_data.keys())
    unstable_sizes = list(unstable_data.values())
    unstable_colors = [unstable_color_map[label] for label in unstable_labels]

    # Plot inner donut (stable clusters) with custom autopct
    if stable_sizes:
        wedges1, texts1, autotexts1 = ax.pie(
            stable_sizes, radius=0.7, labels=None,  # No direct labels
            autopct=autopct_func, pctdistance=0.35, colors=stable_colors,
            wedgeprops=dict(width=0.3, edgecolor='white')
        )

        # For inner clusters, add the cluster label just above the percentage (if percentage >= 2.5)
        for wedge, autotext, label, size in zip(wedges1, autotexts1, stable_labels, stable_sizes):
            if size >= 2.5 and autotext.get_text() != '':
                pos = autotext.get_position()  # (x,y) position of the autopct text
                # Shift slightly upward (y+0.1)
                ax.text(pos[0], pos[1] + 0.1, label, ha='center', va='bottom', fontsize=9, color='black',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))

    # Plot outer donut (unstable clusters) with custom autopct
    if unstable_sizes:
        wedges2, texts2, autotexts2 = ax.pie(
            unstable_sizes, radius=1.0, labels=None,  # No direct labels
            autopct=autopct_func, pctdistance=0.75, colors=unstable_colors,
            wedgeprops=dict(width=0.3, edgecolor='white')
        )

        # For outer clusters, add the cluster label just below the percentage (if percentage >= 2.5)
        for wedge, label, size in zip(wedges2, unstable_labels, unstable_sizes):
            if size >= 2.5:
                ang = (wedge.theta2 + wedge.theta1) / 2
                x = np.cos(np.radians(ang)) * 1.3
                y = np.sin(np.radians(ang)) * 1.3
                # Shift slightly downward (y-0.1)
                ax.text(x, y - 0.1, label, ha='center', va='top', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))

    # Set title for the subplot
    ax.set_title(f'T = {T}', fontsize=12)

# Remove extra subplots if any
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Create a global legend to explain inner and outer rings
legend_elements = [
    Patch(facecolor='gray', edgecolor='gray', label='Inner ring: Stable clusters'),
    Patch(facecolor='white', edgecolor='white', label='Outer ring: Unstable clusters')
]
fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95), fontsize=12)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the legend
plt.savefig("cooperator_cluster_donut_charts.png", dpi=300, bbox_inches="tight")
plt.show()
