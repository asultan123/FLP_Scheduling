import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, product
import copy
from operator import itemgetter

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
    plt.show()

    return G

def topological_sort_grouped(G):
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    while zero_indegree:
        yield zero_indegree
        new_zero_indegree = []
        for v in zero_indegree:
            for _, child in G.edges(v):
                indegree_map[child] -= 1
                if not indegree_map[child]:
                    new_zero_indegree.append(child)
        zero_indegree = new_zero_indegree

def bindings(processors, nodes):
    return product(*[range(processors)]*nodes)

def schedule(binding, grouped_top_sort):
    sched = {}
    time = 0
    group_offset = 0
    for group in grouped_top_sort:
        group_binding = binding[group_offset : group_offset+len(group)]
        sorted_group = sorted(list(zip(group, group_binding)), key=itemgetter(1))
        last_processor = sorted_group[0][1]
        sched[sorted_group[0][0]] = time
        for idx, (task, task_binding) in enumerate(sorted_group):
            if idx == 0:
                continue
            if task_binding == last_processor:
                 time += 1
            sched[task] = time
            last_processor = task_binding
        group_offset += len(group)
        time += 1
    return sched

def max_deadlines(graph_instance):
    bfs = nx.bfs_predecessors(graph_instance)
    pass

processors, nodes = 4, 32

graph_instance = layer_by_layer(nodes, 0.1, 1)
# graph_instance = layer_by_layer(nodes, 0.4, 2)
# graph_instance = layer_by_layer(nodes, 1, 2)

max_deadlines(graph_instance)

grouped_top_sort = list(topological_sort_grouped(graph_instance))
flat_top_sort = list(chain(*grouped_top_sort))
task_bindings = bindings(processors, nodes)
schedule(next(task_bindings), grouped_top_sort)
print("STOP")
