import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pandas as pd
import logging

np.random.seed(46)  # Set a fixed seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
num_nodes = 10000  # Network size
num_repetitions = 1  # Number of simulations per T value
time_steps = 20  # Number of steps per simulation
k_mean = 8  # Mean degree
#T_values = np.linspace(1.0, 2.0, 101)  # Range of T values for additional analysis
T_values = np.linspace(1.4, 1.4, 1)  # Range of T values for additional analysis
#T_values = np.linspace(1.0, 2.0, num=11)  # 11 points from 1.0 to 2.0 inclusive

# Network initialization functions
def create_2d_lattice():
    grid_size = int(np.sqrt(num_nodes))
    network = nx.grid_2d_graph(grid_size, grid_size, periodic=True)
    for node in network.nodes:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    neighbor = ((node[0] + dx) % grid_size, (node[1] + dy) % grid_size)
                    network.add_edge(node, neighbor)
    return nx.convert_node_labels_to_integers(network)

def create_random_regular():
    return nx.random_regular_graph(k_mean, num_nodes)

def create_erdos_renyi():
    p = k_mean / (num_nodes - 1)
    return nx.erdos_renyi_graph(num_nodes, p)

def create_barabasi_albert():
    m = k_mean // 2
    return nx.barabasi_albert_graph(num_nodes, m)

# Select networks
network_types = {
    "2D Lattice": create_2d_lattice,
    #"Random Regular (RR)": create_random_regular,
    #"Erdős–Rényi (ER)": create_erdos_renyi,
    #"Barabási–Albert (BA)": create_barabasi_albert
}

def initialize_strategy(network):
    return {node: np.random.choice([0, 1]) for node in network.nodes}

def calculate_average_clustering(network):
    return nx.average_clustering(network)

def calculate_payoffs(network, T, strategies):
    payoffs = {}
    for node in network.nodes:
        payoff = sum(
            1 if strategies[node] == 1 and strategies[neighbor] == 1 else
            T if strategies[node] == 0 and strategies[neighbor] == 1 else 0
            for neighbor in network.neighbors(node)
        )
        payoffs[node] = payoff
    return payoffs

def unconditional_imitation(network, T, strategies):
    new_strategies = strategies.copy()
    payoffs = calculate_payoffs(network, T, strategies)
    for node in network.nodes:
        best_neighbor = max(network.neighbors(node), key=lambda n: payoffs[n], default=node)
        if payoffs[best_neighbor] > payoffs[node]:
            new_strategies[node] = strategies[best_neighbor]
    return new_strategies

def classify_nodes(strategy_history):
    always_coop = sum(1 for hist in strategy_history.values() if all(s == 1 for s in hist))
    always_defect = sum(1 for hist in strategy_history.values() if all(s == 0 for s in hist))
    oscillators = num_nodes - always_coop - always_defect
    return always_coop, always_defect, oscillators

def paint_network(network, strategy_history, network_type, T, time_step, repetition, num_nodes):
    """Paint the network with nodes colored based on their strategy history."""
    pos = {node: (node % int(np.sqrt(num_nodes)), node // int(np.sqrt(num_nodes))) for node in network.nodes}
    #pos = nx.spring_layout(network)  # Use spring layout

    colors = []
    for node in network.nodes:
        if time_step == 0:
            # Consider the entire history
            history = strategy_history[node]
            if all(s == 1 for s in history):
                colors.append('green')   # Always cooperators in green
            elif all(s == 0 for s in history):
                colors.append('red')     # Always defectors in red
            else:
                colors.append('yellow')  # Oscillators in yellow

        else:
            # Consider only the last two rounds
            history = strategy_history[node][-2:]  # Consider only the last two rounds
            if history == [1, 1]:  # Cooperated in the last two rounds
                colors.append('green')
            elif history == [0, 0]:  # Defected in the last two rounds
                colors.append('red')
            else:  # Changed behavior in the last two rounds
                colors.append('yellow')

    node_size = max(num_nodes / np.sqrt(num_nodes)/10, 5)  # Ensure nodes remain visible

    # Remove labels and edges if more than 100 nodes
    show_labels = num_nodes <= 100
    draw_edges = num_nodes <= 100
    # Add title with network type, T, time step, repetition number, and number of nodes
    plt.title(
        f"Network Type: {network_type}, T: {T}, Time Step: {time_step}, Simulation: {repetition}, "
        f"Nodes: {num_nodes}")
    nx.draw(network,
            pos=pos,
            node_color=colors,
            node_size=node_size,
            with_labels=show_labels,
            edge_color='gray' if draw_edges else None)  # Hide edges if too many nodes

    # Save the plot as a PNG file
    plt.savefig(f"plot_{network_type}_T{T}_step{time_step}_rep{repetition}_nodes{num_nodes}.png")
    plt.close()  # Close the plot to release memory
    # plt.show()


def count_cooperator_clusters(network, strategies):
    """Count the number of clusters of cooperators in the network."""
    # Create a subgraph with only cooperator nodes
    cooperator_nodes = [node for node, strategy in strategies.items() if strategy == 1]
    cooperator_subgraph = network.subgraph(cooperator_nodes)

    # Count the number of connected components (clusters) in the subgraph
    num_clusters = nx.number_connected_components(cooperator_subgraph)

    return num_clusters


def simulate_for_T(network, T_values, num_repetitions, time_steps, num_nodes, network_type):
    """Run the simulation for a range of T values and return results."""
    results = []

    # Fixed T values for different network types
    T_fixed_values = {"2D Lattice": 1.4, "Erdős–Rényi (ER)": 1.7, "Random Regular (RR)": 1.7,
                      "Barabási–Albert (BA)": 1.4}

    for T in T_values:
        for rep in range(num_repetitions):
            # Initialize strategies randomly with 50% cooperators
            strategies = initialize_strategy(network)
            strategy_history = {node: [strategies[node]] for node in network.nodes}  # Initialize with first strategy

            print("Initial strategies:", strategies)

            # Visualize the initial network before any updates
            if network_type in T_fixed_values and T == T_fixed_values[network_type] and rep == 0:
                paint_network(network, strategy_history, network_type, T, 0, rep, num_nodes)

            # Calculate fractions for the initial step
            always_coop, always_defect, oscillators = classify_nodes(strategy_history)
            coop_fraction = sum(strategies.values()) / num_nodes

            avg_clustering_coeff = calculate_average_clustering(network)

            # Store results for the initial step
            results.append({
                "network_type": network_type,
                "T": T,
                "time_step": 0,
                "repetition": rep,
                "coop_fraction": coop_fraction,
                "always_coop_fraction": always_coop / num_nodes * 100,
                "always_defect_fraction": always_defect / num_nodes * 100,
                "oscillators_fraction": oscillators / num_nodes * 100,
                "num_cooperator_clusters": 0,  # No clusters in the initial step
                "avg_clustering_coeff": avg_clustering_coeff
            })

            # Run the simulation
            for step in range(1, time_steps):
                print(f"Before update ({T}, step {step}):", strategies)
                strategies = unconditional_imitation(network, T, strategies)
                print(f"After update ({T}, step {step}):", strategies)

                # Record fraction of cooperators
                coop_fraction = sum(strategies.values()) / num_nodes

                # Store strategies for last 2 steps
                for node in network.nodes:
                    strategy_history[node].append(strategies[node])
                    if len(strategy_history[node]) > 2:
                        strategy_history[node].pop(0)

                # Calculate fractions based on the current and last round
                always_coop, always_defect, oscillators = classify_nodes(
                    {node: strategy_history[node][-2:] for node in network.nodes})

                # Count the number of cooperator clusters in the last step
                num_cooperator_clusters = count_cooperator_clusters(network, strategies)

                # Store results with time step and repetition number
                results.append({
                    "network_type": network_type,
                    "T": T,
                    "time_step": step,
                    "repetition": rep,
                    "coop_fraction": coop_fraction,
                    "always_coop_fraction": always_coop / num_nodes * 100,
                    "always_defect_fraction": always_defect / num_nodes * 100,
                    "oscillators_fraction": oscillators / num_nodes * 100,
                    "num_cooperator_clusters": num_cooperator_clusters,
                    "avg_clustering_coeff": avg_clustering_coeff
                })

                # Paint the network at specific timesteps only for the first repetition
                if network_type in T_fixed_values and T == T_fixed_values[network_type] and rep == 0:
                    paint_network(network, strategy_history, network_type, T, step, rep, num_nodes)

    return results

if __name__ == "__main__":
    num_cores = min(40, cpu_count())

    all_results = []  # Collect all results in one place

    for network_name, create_network_func in network_types.items():
        logging.info(f"Simulating cooperation vs T for: {network_name}")
        network = create_network_func()

        with Pool(processes=num_cores) as pool:
            results = pool.starmap(simulate_for_T, [(network, T_values, num_repetitions, time_steps, num_nodes, network_name)])

        # Flatten the list of results
        results = [item for sublist in results for item in sublist]

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        df["network_type"] = network_name  # Keep track of network type
        all_results.append(df)

        # PLOT 1
        # Plot results for this network type
        plt.errorbar(df["T"], df["coop_fraction"], fmt='o-', label="Average cooperation")
        plt.xlabel("T (Temptation)")
        plt.ylabel("Average fraction of cooperators")
        plt.title(f"Fraction of cooperators vs. T on {network_name}")
        plt.legend()
        plt.savefig(f"cooperation_vs_T_{network_name.replace(' ', '_')}.png")
        plt.show()
        plt.clf()

        logging.info(f"Results processed for {network_name}")

    # Combine all results into a single CSV
    combined_df = pd.concat(all_results)
    combined_df.to_csv("strategy_dynamics.csv", index=False)  # Save only ONE CSV

    # PLOT 2
    # Strategy dynamics plot
    # Define the selected T values for each network type
    selected_T_values = {
        "2D Lattice": 1.4,
        "Random Regular (RR)": 1.7,
        "Erdős–Rényi (ER)": 1.9,
        "Barabási–Albert (BA)": 1.4
    }

    # Filter DataFrame and create a copy to avoid SettingWithCopyWarning
    filtered_df = combined_df[combined_df.apply(lambda row: row["T"] == selected_T_values.get(row["network_type"], None), axis=1)].copy()

    # Convert percentages to fractions
    percentage_columns = ["always_coop_fraction", "always_defect_fraction", "oscillators_fraction"]
    filtered_df[percentage_columns] = filtered_df[percentage_columns] / 100.0

    print(filtered_df[percentage_columns])

    # Group by network type and calculate mean values
    filtered_grouped_df = filtered_df.groupby("network_type")[
        ["coop_fraction", "always_coop_fraction", "always_defect_fraction", "oscillators_fraction"]].mean()

    # Now use this single CSV for the final plot
    plt.figure(figsize=(12, 8))
    filtered_grouped_df.plot(kind='bar', figsize=(12, 8))
    plt.xlabel("Network Type")
    plt.ylabel("Fraction")
    plt.title("Cooperation and Strategy Dynamics Across Networks")
    plt.legend(title="Metrics")
    plt.grid()
    plt.savefig("strategy_dynamics.png")
    plt.show()