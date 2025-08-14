import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("strategy_dynamics_section_3.2.csv")

# Filter the DataFrame for time step 99
df_filtered = df[df["time_step"] == 99].copy()

# Convert percentages to fractions
df_filtered.loc[:, "always_coop_fraction"] = df_filtered["always_coop_fraction"] / 100.0
df_filtered.loc[:, "always_defect_fraction"] = df_filtered["always_defect_fraction"] / 100.0
df_filtered.loc[:, "oscillators_fraction"] = df_filtered["oscillators_fraction"] / 100.0

# Get unique network types
network_types = df_filtered["network_type"].unique()

# Create a combined figure with 4 plots, one for each network type
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
axs = axs.flatten()

for i, network_type in enumerate(network_types):
    # Filter the DataFrame for the current network type
    df_network = df_filtered[df_filtered["network_type"] == network_type]

    # Group by T and calculate mean values, excluding non-numeric columns
    df_grouped = df_network.groupby("T").mean(numeric_only=True)

    # Plot the results
    ax1 = axs[i]
    ax1.plot(df_grouped.index, df_grouped["coop_fraction"], label="Coop Fraction")
    ax1.plot(df_grouped.index, df_grouped["always_coop_fraction"], label="Consistent Coop Fraction")
    ax1.plot(df_grouped.index, df_grouped["always_defect_fraction"], label="Consistent Defect Fraction")
    ax1.plot(df_grouped.index, df_grouped["oscillators_fraction"], label="Oscillators Fraction")
    ax1.set_xlabel("T (Temptation)")
    ax1.set_ylabel("Fraction")
    ax1.set_title(f"Average Fractions vs. T at Time Step 99 ({network_type})")
    ax1.legend(loc="upper left")
    ax1.grid()
    # Draw vertical line at T = 5/3
    x_flag = 5 / 3
    ax1.axvline(x=x_flag, color='black', linestyle='--', linewidth=2, alpha=0.8)

    # Add custom label below the x-axis in larger font
    ax1.text(x_flag, ax1.get_ylim()[0] - 0.01 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
             r"$\frac{5}{3}$", fontsize=14, ha='center', va='top')

    # Create a second y-axis for the number of cooperator clusters
    ax2 = ax1.twinx()
    ax2.plot(df_grouped.index, df_grouped["num_cooperator_clusters"], label="Num Cooperator Clusters", color='purple', linestyle='--')
    #ax2.set_yscale('log')
    ax2.set_ylabel("Number of Cooperator Clusters (log scale)")
    ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("average_fractions_vs_T_with_clusters_combined.png")
plt.show()