import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import config
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
import utility
from ILP_Generators import ILPWithExplicitProcessors, ILPWithImplicitProcessors
from Greedy import get_greedy_schedule
from pprint import pprint as pp

def layer(subset_sizes, target):
    for layer_idx, subset_size in enumerate(np.cumsum(subset_sizes)):
        if target <= subset_size:
            return layer_idx
    raise Exception("Could not find target")

def layer_by_layer(n, p, plot_graphs=False, favor_longer_graphs=True):
    subset_sizes = np.array([])
    while np.sum(subset_sizes) < n-1:
        y = (n - np.sum(subset_sizes))
        if favor_longer_graphs:
            y /= np.random.randint(1, y)
        x = np.random.randint(1, y + 1)
        subset_sizes = np.append(subset_sizes, x)

    if config.debug:
        print("\nLayers sizes:\n", subset_sizes)

    # Let M be the adjacency matrix nxn initialized as the zero matrix
    M = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if layer(subset_sizes, j) > layer(subset_sizes, i):
                if np.random.rand() < p:
                    M[i][j] = 1

    for i in range(n):
        if (sum(M[i, :]) + sum(M[:, i])) == 0:
            if i == 0:
                M[i, np.random.randint(i+1, n)] = 1
            elif i == n-1:
                M[np.random.randint(0, i-1), i] = 1
            else:
                M[i, np.random.randint(i+1, n)] = 1

    # print("\nAdjacency Matrix:\n", M)

    # Conver the numpy matrix into a directed networkX graph
    G = nx.convert_matrix.from_numpy_array(M, create_using=nx.DiGraph)

    G_grouped = utility.topological_sort_grouped_with_terminal_node_reposition(G)
    subset_sizes = list(map(len, G_grouped))
    G = utility.change_node_start_index(G, G_grouped)

    # print("Number of Nodes: ", G.number_of_nodes())
    if config.debug:
        print("\nNumber of Edges: ", G.number_of_edges())

    # Add information regarding the layer of each node (for graph purposes)
    if plot_graphs:
        for i in range(1, n+1):
            G.nodes[i]['layer'] = layer(subset_sizes, i)

        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=True)
        plt.show()

    return G

def plot_instance(G):
    G_grouped = utility.topological_sort_grouped_with_terminal_node_reposition(G)
    subset_sizes = list(map(len, G_grouped))
    G = utility.change_node_start_index(G, G_grouped)

    # print("Number of Nodes: ", G.number_of_nodes())
    if config.debug:
        print("\nNumber of Edges: ", G.number_of_edges())

    n = len(G.nodes())
    # Add information regarding the layer of each node (for graph purposes)
    for i in range(1, n+1):
        G.nodes[i]['layer'] = layer(subset_sizes, i)

    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True)
    plt.show()


def show_example_instances():
    # check correct way of seeding
    layer_by_layer(config.node_min, np.random.rand(), plot_graphs=True)
    layer_by_layer(config.node_max, np.random.rand(), plot_graphs=True)


def ilp_formulation(instance, processor_count):
    # instance = layer_by_layer(22, 0.20, plot_graphs=True)
    width = utility.get_task_graph_max_width(instance)
    print("Instance Max Width: {}".format(width))
    opt = pyo.SolverFactory('gurobi')
    return ILPWithImplicitProcessors.construct_model(instance, processor_count, opt, initialize_with_greedy=True)


def toggle_self_dependence(adj_matrix):
    for i in range(len(adj_matrix.diagonal())):
        adj_matrix[i, i] = 0 if adj_matrix[i, i] == 1 else 1
    return adj_matrix



# Pyomo model
def ilp_instance_test():
    np.random.seed(config.seed)
    instance = layer_by_layer(100, 0.50, plot_graphs=False)
    processor_count = 2
    timelimit = 60
    lower_bound = utility.get_lower_bound(instance)

    _, greedy_makespan = get_greedy_schedule(instance, processor_count)
    print("Greedy Makespan: {}".format(greedy_makespan))
    width = utility.get_task_graph_max_width(instance)
    print("Instance Max Width: {}".format(width))
    print("Lower bound on makespan: {}".format(lower_bound))

    print("Building Pyomo ILP Model")

    model = ILPWithImplicitProcessors.construct_model(instance, processor_count, lower_bound, get_timeout= lambda: timelimit)

    print("Solving Pyomo ILP Model")
    model.solve()

    print("Pyomo ILP Solution Results")
    print("Task Start Time")
    pp(model.get_schedule())
    print("Makespan")
    pp(model.get_makespan())




    print("Building GurobiPy ILP Model")
    opt = pyo.SolverFactory('gurobi')
    model = ILPWithExplicitProcessors.construct_model(instance, processor_count, lower_bound, get_timelimit= lambda: timelimit)

    print("Solving GurobiPy ILP Model")
    model.solve()

    print("GurobiPy ILP Solution Results")
    print("Task Start Time")
    pp(model.get_schedule())
    print("Makespan")
    pp(model.get_makespan())
