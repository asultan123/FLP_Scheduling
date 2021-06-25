import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import config
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
from utility import topological_sort_grouped

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
    instance = layer_by_layer(20, 0.25, plot_graphs=False)
    generate_ilp_model_from_instance(instance,4)
    
def generate_ilp_model_from_instance(instance, processor_count):
    
    model = pyo.ConcreteModel()

    model.T = pyo.Var(domain=pyo.NonNegativeIntegers)
    model.x = pyo.Var(instance.nodes(), domain=pyo.NonNegativeIntegers, initialize=0)

    n = len(instance.nodes())
    w_indexes = [(i,j) for j in RangeSet(1,n+1) for i in RangeSet(0,n)]
    model.w = pyo.Var(w_indexes, domain=pyo.NonNegativeIntegers, initialize=0)
    model.OBJ = pyo.Objective(expr = model.T, sense = pyo.minimize) 

    instance_transitive_reduction = nx.transitive_reduction(instance)
    nodes_with_no_successors = [v for v, d in instance_transitive_reduction.out_degree() if d == 0]
    nodes_with_no_predecessors = [v for v, d in instance_transitive_reduction.in_degree() if d == 0]

    model.no_predecessors_constraint = pyo.ConstraintList()
    for node in nodes_with_no_predecessors:
        model.no_predecessors_constraint.add(expr = model.x[node] >= 0)
    
    model.precedence_constraints = pyo.ConstraintList()
    for node in instance_transitive_reduction.nodes():
        if node not in nodes_with_no_predecessors:
            for successor in instance_transitive_reduction.successors(node):
                model.precedence_constraints.add(expr = model.x[successor] >= model.x[node] + 1)            

    model.makespan_constraint = pyo.ConstraintList()
    for node in nodes_with_no_successors:
        model.makespan_constraint.add(expr = model.T >= model.x[node] + 1)
        
    model.processor_bound_constraint = pyo.Constraint(expr = sum(model.w[:,n+1]) <= processor_count)

    topologically_sorted_instance = list(topological_sort_grouped(instance))
    widths = map(len, topologically_sorted_instance)
    max_width = max(widths)
    if processor_count < max_width:
        instance_transitive_closure = nx.adjacency_matrix(nx.transitive_closure_dag(instance))
        for idx in range(len(instance_transitive_closure.diagonal())):
            instance_transitive_closure[idx,idx] = 1
            
        # TODO ADD BEFORE TASK ORDERING CONSTRAINT
        # TODO AFTER TASK ORDERING CONSTRAINT
        # TODO ADD RELAXABLE CONSTRAINT ON SEQUENTIAL EXECUTION
        
        print()
    elif processor_count >= max_width:
        # TODO ADD PROCESSOR COUNT >= FORMULATION
        pass
            
    pass    

if __name__ == "__main__":
    np.random.seed(config.seed)
    ilp_formulation()

