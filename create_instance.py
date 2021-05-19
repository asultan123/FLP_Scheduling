import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import config

def layer(subset_sizes, target):
    for layer_idx, subset_size in enumerate(np.cumsum(subset_sizes)):
        if target <= subset_size:
            return layer_idx

def layer_by_layer(n, p, seed):
    np.random.seed(seed) # check correct way of seeding 

    # Distribute n vertices...
    subset_sizes = np.array([])
    while np.sum(subset_sizes) < n-1:
        y = (n - np.sum(subset_sizes))
        x = np.random.randint(1, y + 1)
        subset_sizes = np.append(subset_sizes, x)

    if config.debug:
        print("\nLayers sizes:\n", subset_sizes)

    # Let M be the adjacency matrix nxn initialized as the zero matrix
    M = np.zeros((n,n))

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
    if config.instance_plt:
        for i in range(n):
            G.nodes[i]['layer'] = layer(subset_sizes, i)
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=True)
        plt.show()

    return G

