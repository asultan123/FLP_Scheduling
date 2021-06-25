import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import config


def layer(subset_sizes, target):
    for layer_idx, subset_size in enumerate(np.cumsum(subset_sizes)):
        if target <= subset_size:
            return layer_idx


def layer_by_layer(n, p, plot_graphs=False, favor_longer_graphs=True):
    subset_sizes = np.array([])
    while np.sum(subset_sizes) < n-1:
        y = (n - np.sum(subset_sizes))
        if favor_longer_graphs:
            y /= np.random.randint(1,y)
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

    # print("\nAdjacency Matrix:\n", M)

    # Conver the numpy matrix into a directed networkX graph
    G = nx.convert_matrix.from_numpy_array(M, create_using=nx.DiGraph)

    # print("Number of Nodes: ", G.number_of_nodes())
    if config.debug:
        print("\nNumber of Edges: ", G.number_of_edges())

    # Add information regarding the layer of each node (for graph purposes)
    if plot_graphs:
        for i in range(n):
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
    instance = layer_by_layer(20, 0.25, plot_graphs=True)
    generate_ilp_model_from_instance(instance)
    
def generate_ilp_model_from_instance(instance):
    instance_transitive_closure = nx.transitive_closure_dag(instance)
    instance_transitive_reduction = nx.transitive_reduction(instance)
    nodes_with_no_successors = [v for v, d in instance_transitive_reduction.out_degree() if d == 0]
    nodes_with_no_predecessors = [v for v, d in instance_transitive_reduction.in_degree() if d == 0]

    pass    

if __name__ == "__main__":
    np.random.seed(config.seed)
    ilp_formulation()

