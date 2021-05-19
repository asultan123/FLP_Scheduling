import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def layer(subset_sizes, k):
    # Let layer(v) be the layer assigned to vertex v
    l = 1
    while(np.sum(subset_sizes[:l]) < k):
        l +=1

    return l

def layer_by_layer(n, p, seed, print_graph=False):
    np.random.seed(seed) # check correct way of seeding 

    # Distribute n vertices...
    subset_sizes = np.array([])
    while np.sum(subset_sizes) < n:
        y = (n - np.sum(subset_sizes) + 1)
        x = np.random.randint(1, y)
        subset_sizes = np.append(subset_sizes, x)

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

    # Add information regarding the layer of each node (for graph purposes) 
    for i in range(n):
        G.nodes[i]['layer'] = layer(subset_sizes, i)
        
    # print("Number of Nodes: ", G.number_of_nodes())
    print("\nNumber of Edges: ", G.number_of_edges())
    # print("Nodes: ", G.nodes())
    # print("\nEdges: ", G.edges())
    
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True)

    return G, plt


graph_instance, plt = layer_by_layer(10, 0.3, 1)

print(list(nx.topological_sort(graph_instance)))

print("STOP")
