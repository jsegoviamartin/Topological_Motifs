import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pandas as pd
import logging
import json

np.random.seed(61)  # Set a fixed seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
num_nodes = 10000  # Network size
num_repetitions = 1  # Number of simulations per T value
time_steps = 100  # Number of steps per simulation
k_mean = 8  # Mean degree
#T_values = np.linspace(1.0, 2.0, 101)  # Range of T values for additional analysis
T_values = np.linspace(1.8, 1.8, 1)  # Range of T values for additional analysis
#T_values = np.linspace(1.0, 2.0, num=11)  # 11 points from 1.0 to 2.0 inclusive
#T_values = np.sort(np.concatenate((np.linspace(1.0, 2.0, num=11), [1.66, 1.67, 1.68])))


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
    #"2D Lattice": create_2d_lattice,
    "Random Regular (RR)": create_random_regular,
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
    """Paint the network with nodes colored based on their strategy history,
    and also paint a version with defector nodes removed.
    """
    # Set up positions for nodes (using grid positions here)
    pos = {node: (node % int(np.sqrt(num_nodes)), node // int(np.sqrt(num_nodes))) for node in network.nodes}
    # Alternatively, you can use: pos = nx.spring_layout(network)

    # Prepare colors for each node based on strategy history
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
            history = strategy_history[node][-2:]
            if history == [1, 1]:  # Cooperated in the last two rounds
                colors.append('green')
            elif history == [0, 0]:  # Defected in the last two rounds
                colors.append('red')
            else:  # Changed behavior in the last two rounds
                colors.append('yellow')

    node_size = max(num_nodes / np.sqrt(num_nodes) / 10, 5)  # Ensure nodes remain visible

    # Determine whether to show labels and edges (only if there are few nodes)
    show_labels = num_nodes <= 100
    draw_edges = num_nodes <= 100

    # Plot the full network with defectors
    plt.figure()
    plt.title(
        f"Network Type: {network_type}, T: {T}, Time Step: {time_step}, Simulation: {repetition}, Nodes: {num_nodes}"
    )
    nx.draw(network,
            pos=pos,
            node_color=colors,
            node_size=node_size,
            with_labels=show_labels,
            edge_color='gray' if draw_edges else None)
    plt.savefig(f"plot_{network_type}_T{T}_step{time_step}_rep{repetition}_nodes{num_nodes}.png")
    plt.close()  # Close the plot to release memory

    # Now create a copy of the network and remove defector nodes (and their links)
    network_no_defectors = network.copy()
    defectors = []
    for node in network.nodes:
        if time_step == 0:
            history = strategy_history[node]
            if all(s == 0 for s in history):
                defectors.append(node)
        else:
            history = strategy_history[node][-2:]
            if history == [0, 0]:
                defectors.append(node)
    network_no_defectors.remove_nodes_from(defectors)

    # Prepare positions for the new network (only for nodes that remain)
    pos_no_defectors = {node: pos[node] for node in network_no_defectors.nodes}

    # For visualization, you might choose to color remaining nodes uniformly or recalc colors
    # Here, we simply color them in green if they are all cooperators (or non-defectors)
    # If needed, you can use the previous logic to assign colors to oscillators as well.
    # For now, we'll assume remaining nodes are cooperators or oscillators and paint them green.
    no_defectors_color = 'green'

    # Plot the network without defectors
    plt.figure()
    plt.title(
        f"Network without Defectors: {network_type}, T: {T}, Time Step: {time_step}, Simulation: {repetition}, "
        f"Nodes: {network_no_defectors.number_of_nodes()}"
    )
    nx.draw(network_no_defectors,
            pos=pos_no_defectors,
            node_color=no_defectors_color,
            node_size=node_size,
            with_labels=show_labels,
            edge_color='gray' if draw_edges else None)
    plt.savefig(f"plot_{network_type}_T{T}_step{time_step}_rep{repetition}_nodes{network_no_defectors.number_of_nodes()}_no_defectors.png")
    plt.close()


def classify_cluster_shape(cluster):
    """
    Classifies a cluster of cooperators based on geometric/topological properties.

    Args:
        cluster: A NetworkX subgraph of cooperators.

    Returns:
        A stable, human-readable label.
    """
    size = len(cluster.nodes)
    if size >= 20:
        return "large_cluster"

    if size == 1:
        return "single_node"

    # Perimeter nodes: Nodes with fewer than the max possible neighbors (assuming 2D grid, max=4)
    perimeter_nodes = sum(1 for n in cluster.nodes if cluster.degree[n] < 4)

    # Compactness: size / perimeter count (higher means denser)
    compactness = size / perimeter_nodes if perimeter_nodes > 0 else float('inf')

    # Approximate the longest shortest-path distance (diameter of cluster)
    try:
        longest_path = nx.approximation.diameter(cluster)
    except nx.NetworkXError:  # If graph is disconnected, approximate longest shortest-path
        ecc = nx.eccentricity(cluster)
        longest_path = max(ecc.values()) if ecc else 1

    aspect_ratio = longest_path / size  # Close to 1 means chain-like

    # Euler characteristic (holes detection)
    num_edges = cluster.number_of_edges()
    euler_characteristic = size - num_edges  # Ignoring components since cluster is connected

    # **Classification Rules**
    if compactness > 1.5:
        return f"compact_{size}"
    elif aspect_ratio > 0.7:  # Chain-like structure
        return f"chain_{size}"
    elif euler_characteristic < 1:  # If Euler characteristic suggests a hole
        return f"loop_{size}"
    else:
        return f"irregular_{size}"

###COUNTING STABLE AND UNSTABLE CLUSTERS
def count_cooperator_clusters(network, current_strategies, prev_strategies=None):
    """
    Count cooperator clusters and classify them by shape.
    If prev_strategies is provided, the function returns both stable clusters
    (clusters that have exactly the same set of nodes as in the previous step)
    and unstable clusters (clusters that do not have a matching cluster in the previous step).

    Args:
        network: The full NetworkX graph.
        current_strategies: Dict of node -> 0 or 1 for the current time step.
        prev_strategies: (Optional) Dict of node -> 0 or 1 for the previous time step.

    Returns:
        If prev_strategies is None:
            num_clusters (int), cluster_type_counts (dict)
        Otherwise, returns four values:
            num_stable_clusters (int), stable_cluster_type_counts (dict),
            num_unstable_clusters (int), unstable_cluster_type_counts (dict)
    """
    # Build current cooperator subgraph
    current_cooperators = [n for n, s in current_strategies.items() if s == 1]
    current_subgraph = network.subgraph(current_cooperators)
    current_clusters = [current_subgraph.subgraph(c).copy() for c in nx.connected_components(current_subgraph)]

    # If no previous strategies provided, count all clusters
    if prev_strategies is None:
        num_clusters = len(current_clusters)
        cluster_type_counts = {}
        for cluster in current_clusters:
            shape_label = classify_cluster_shape(cluster)
            cluster_type_counts[shape_label] = cluster_type_counts.get(shape_label, 0) + 1
        return num_clusters, cluster_type_counts
    else:
        # Build previous cooperator subgraph
        prev_cooperators = [n for n, s in prev_strategies.items() if s == 1]
        prev_subgraph = network.subgraph(prev_cooperators)
        prev_clusters = [prev_subgraph.subgraph(c).copy() for c in nx.connected_components(prev_subgraph)]

        stable_clusters = []
        unstable_clusters = []
        # For each current cluster, check if there exists an identical cluster (i.e. same nodes) in the previous step
        for curr_cluster in current_clusters:
            curr_nodes = set(curr_cluster.nodes())
            if any(curr_nodes == set(prev_cluster.nodes()) for prev_cluster in prev_clusters):
                stable_clusters.append(curr_cluster)
            else:
                unstable_clusters.append(curr_cluster)

        # Count and classify stable clusters
        num_stable_clusters = len(stable_clusters)
        stable_cluster_type_counts = {}
        for cluster in stable_clusters:
            shape_label = classify_cluster_shape(cluster)
            stable_cluster_type_counts[shape_label] = stable_cluster_type_counts.get(shape_label, 0) + 1

        # Count and classify unstable clusters
        num_unstable_clusters = len(unstable_clusters)
        unstable_cluster_type_counts = {}
        for cluster in unstable_clusters:
            shape_label = classify_cluster_shape(cluster)
            unstable_cluster_type_counts[shape_label] = unstable_cluster_type_counts.get(shape_label, 0) + 1

        return num_stable_clusters, stable_cluster_type_counts, num_unstable_clusters, unstable_cluster_type_counts

def simulate_for_T(network, T_values, num_repetitions, time_steps, num_nodes, network_type):
    """Run the simulation for a range of T values and return results."""
    results = []

    # Fixed T values for different network types
    T_fixed_values = {"2D Lattice": 1.4, "Erdős–Rényi (ER)": 1.7, "Random Regular (RR)": 1.8,
                      "Barabási–Albert (BA)": 1.4}

    for T in T_values:
        for rep in range(num_repetitions):
            # Initialize strategies randomly with 50% cooperators
            strategies = initialize_strategy(network)
            # Each node's strategy history is maintained as a list; initially, only one entry exists.
            strategy_history = {node: [strategies[node]] for node in network.nodes}

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
                "cooperator_cluster_details": {},
                "num_unstable_cooperator_clusters": 0,
                "unstable_cooperator_cluster_details": {},
                "avg_clustering_coeff": avg_clustering_coeff
            })

            # Run the simulation over time steps
            for step in range(1, time_steps):
                print(f"Before update ({T}, step {step}):", strategies)
                strategies = unconditional_imitation(network, T, strategies)
                print(f"After update ({T}, step {step}):", strategies)

                # Record fraction of cooperators
                coop_fraction = sum(strategies.values()) / num_nodes

                # Update strategy_history for each node, keeping only the last two time steps.
                for node in network.nodes:
                    strategy_history[node].append(strategies[node])
                    if len(strategy_history[node]) > 2:
                        strategy_history[node].pop(0)

                # Calculate fractions based on the current and last round of strategies
                always_coop, always_defect, oscillators = classify_nodes(
                    {node: strategy_history[node][-2:] for node in network.nodes}
                )

                # Count stable and unstable clusters if we have at least two steps
                if step >= 1:
                    prev_strategies = {node: strategy_history[node][0] for node in network.nodes}
                    current_strategies = {node: strategy_history[node][1] for node in network.nodes}
                    (num_stable_clusters, stable_cluster_type_counts,
                     num_unstable_clusters, unstable_cluster_type_counts) = count_cooperator_clusters(
                         network, current_strategies, prev_strategies
                     )
                else:
                    num_stable_clusters, stable_cluster_type_counts = 0, {}
                    num_unstable_clusters, unstable_cluster_type_counts = 0, {}

                # Optionally, you could store the details as JSON strings:
                stable_details_str = json.dumps(stable_cluster_type_counts)
                unstable_details_str = json.dumps(unstable_cluster_type_counts)

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
                    "num_cooperator_clusters": num_stable_clusters,  # stable clusters
                    "cooperator_cluster_details": stable_details_str,
                    "num_unstable_cooperator_clusters": num_unstable_clusters,
                    "unstable_cooperator_cluster_details": unstable_details_str,
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
    combined_df.to_csv("strategy_dynamics_RR_T1.8_TEST.csv", index=False)  # Save only ONE CSV

    # PLOT 2
    # Strategy dynamics plot
    # Define the selected T values for each network type
    selected_T_values = {
        "2D Lattice": 1.4,
        "Random Regular (RR)": 1.8,
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
    plt.savefig("strategy_dynamics_RR_T1.8_TEST.png")
    plt.show()