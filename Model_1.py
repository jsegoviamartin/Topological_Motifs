import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Parameters
num_nodes = 10  # Network size (10x10 lattice)
T_values = np.linspace(1.0, 2.0, 101)  # T values from 1 to 2 with increments of 0.01
num_repetitions = 20  # Number of repetitions per T value
time_steps = 10  # Number of steps per simulation

# Initialize lattice network with degree k=8 (Moore neighborhood)
grid_size = int(np.sqrt(num_nodes))
network = nx.grid_2d_graph(grid_size, grid_size, periodic=True)
for node in network.nodes:
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                neighbor = ((node[0] + dx) % grid_size, (node[1] + dy) % grid_size)
                network.add_edge(node, neighbor)

# Initialize results storage
results = []

def initialize_strategy():
    """Initialize each node with a random strategy (0=Defect, 1=Cooperate)."""
    return {node: np.random.choice([0, 1]) for node in network.nodes}

def calculate_payoff(T, node, strategies):
    """Calculate payoff for a node based on its neighbors in the lattice."""
    payoff = 0
    for neighbor in network.neighbors(node):
        if strategies[node] == 1 and strategies[neighbor] == 1:
            payoff += 1  # Reward (R)
        elif strategies[node] == 1 and strategies[neighbor] == 0:
            payoff += 0  # Sucker's payoff (S=0)
        elif strategies[node] == 0 and strategies[neighbor] == 1:
            payoff += T  # Temptation (T)
        else:
            payoff += 0  # Punishment (P=0)
    return payoff

def unconditional_imitation(T, strategies):
    """Update strategies based on unconditional imitation."""
    new_strategies = strategies.copy()
    for node in network.nodes:
        # Calculate payoffs for each neighbor
        payoffs = {neighbor: calculate_payoff(T, neighbor, strategies) for neighbor in network.neighbors(node)}
        # Include the node's own payoff
        payoffs[node] = calculate_payoff(T, node, strategies)

        # Find the strategy of the neighbor with the highest payoff
        best_neighbor = max(payoffs, key=payoffs.get)
        if payoffs[best_neighbor] > payoffs[node]:
            new_strategies[node] = strategies[best_neighbor]
    return new_strategies

for T in T_values:
    coop_fractions_reps = []
    for rep in range(num_repetitions):
        # Initialize strategies randomly with 50% cooperators
        strategies = initialize_strategy()

        # Run the simulation
        coop_fractions = []
        for step in range(time_steps):
            strategies = unconditional_imitation(T, strategies)
            # Record fraction of cooperators
            coop_fraction = sum(strategies.values()) / num_nodes
            coop_fractions.append(coop_fraction)

        # Store the average fraction of cooperators for the last n steps
        avg_coop = np.mean(coop_fractions[-100:])
        coop_fractions_reps.append(avg_coop)

    # Calculate mean and standard deviation across all repetitions for this T
    mean_coop = np.mean(coop_fractions_reps)
    std_coop = np.std(coop_fractions_reps)
    results.append([T, mean_coop, std_coop])

# Convert results to numpy array for easier plotting
results = np.array(results)

# Plot results
plt.errorbar(results[:, 0], results[:, 1], yerr=results[:, 2], fmt='o-', label="Average cooperation")
plt.xlabel("T (Temptation)")
plt.ylabel("Average fraction of cooperators")
plt.title("Fraction of cooperators vs. Temptation (T) on a 2D lattice")
plt.legend()
plt.show()

# Save the plot as a PNG file
plt.savefig("cooperation_vs_T_plot.png")