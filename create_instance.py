import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import config
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
import utility
from ILP_Generators import ILPWithImplicitProcessorBound

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


def show_example_instances():
    # check correct way of seeding
    layer_by_layer(config.node_min, np.random.rand(), plot_graphs=True)
    layer_by_layer(config.node_max, np.random.rand(), plot_graphs=True)


def ilp_formulation():
    # instance = layer_by_layer(22, 0.20, plot_graphs=True)
    instance = layer_by_layer(25, 0.30, plot_graphs=True)
    width = utility.get_task_graph_max_width(instance)
    print("Instance Max Width: {}".format(width))
    return ILPWithImplicitProcessorBound.construct_model(instance, 3)


def toggle_self_dependence(adj_matrix):
    for i in range(len(adj_matrix.diagonal())):
        adj_matrix[i, i] = 0 if adj_matrix[i, i] == 1 else 1
    return adj_matrix



np.random.seed(config.seed)
opt = pyo.SolverFactory('gurobi')
print("Building ILP Model")
model = ilp_formulation()
# model.pprint()
print("Solving- ILP Model")
opt.solve(model)
print("ILP Solution Results")
model.T.display()
model.x.display()

