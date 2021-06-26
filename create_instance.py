import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import config
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
from utility import change_node_start_index, topological_sort_grouped, topological_sort_grouped_with_terminal_node_reposition
from itertools import chain
from copy import copy

def layer(subset_sizes, target):
    for layer_idx, subset_size in enumerate(np.cumsum(subset_sizes)):
        if target < subset_size:
            return layer_idx


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
            M[i, np.random.randint(i+1, n)] = 1
            

    # print("\nAdjacency Matrix:\n", M)

    # Conver the numpy matrix into a directed networkX graph
    G = nx.convert_matrix.from_numpy_array(M, create_using=nx.DiGraph)

    G_grouped = topological_sort_grouped_with_terminal_node_reposition(G)
    subset_sizes = list(map(len, G_grouped))
    G = change_node_start_index(G, G_grouped)
    
    # print("Number of Nodes: ", G.number_of_nodes())
    if config.debug:
        print("\nNumber of Edges: ", G.number_of_edges())

    # Add information regarding the layer of each node (for graph purposes)
    if plot_graphs:
        for i in range(1, n+1):
            G.nodes[i]['layer'] = layer(subset_sizes, i-1)

        pos = nx.multipartite_layout(G, subset_key="layer")
        nx.draw(G, pos, with_labels=True)
        plt.show()

    return G


def show_example_instances():
    # check correct way of seeding
    layer_by_layer(config.node_min, np.random.rand(), plot_graphs=True)
    layer_by_layer(config.node_max, np.random.rand(), plot_graphs=True)


def ilp_formulation():
    instance = layer_by_layer(20, 0.15, plot_graphs=False)
    generate_ilp_model_from_instance(instance, 4)


def toggle_self_dependence(adj_matrix):
    for i in range(len(adj_matrix.diagonal())):
        adj_matrix[i, i] = 0 if adj_matrix[i, i] == 1 else 1
    return adj_matrix


class TaskGraphILPGenerator():
    @classmethod
    def construct_model(cls, instance):
        model = pyo.ConcreteModel()
        model = cls.create_variables(instance, model)
        
    @classmethod
    def create_variables(cls, instance, model):
        model.T = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        model.x = pyo.Var(instance.nodes(),
                      domain=pyo.NonNegativeIntegers, initialize=0)
        n = len(instance.nodes())
        w_indexes = [(i, j) for j in RangeSet(1, n+1) for i in RangeSet(0, n)]
        model.w = pyo.Var(w_indexes, domain=pyo.NonNegativeIntegers, initialize=0)
        model.OBJ = pyo.Objective(expr=model.T, sense=pyo.minimize)
        return model

    @classmethod
    def add_precedence_constraints(cls, instance_transitive_reduction, nodes_with_no_predecessors, model):
        model.precedence_constraints = pyo.ConstraintList()
        for node in instance_transitive_reduction.nodes():
            if node not in nodes_with_no_predecessors:
                for successor in instance_transitive_reduction.successors(node):
                    model.precedence_constraints.add(
                        expr=model.x[successor] >= model.x[node] + 1)
        return model

    

def generate_ilp_model_from_instance(instance, processor_count):

    model = pyo.ConcreteModel()

    model.T = pyo.Var(domain=pyo.NonNegativeIntegers)
    model.x = pyo.Var(instance.nodes(),
                      domain=pyo.NonNegativeIntegers, initialize=0)

    n = len(instance.nodes())
    w_indexes = [(i, j) for j in RangeSet(1, n+1) for i in RangeSet(0, n)]
    model.w = pyo.Var(w_indexes, domain=pyo.NonNegativeIntegers, initialize=0)
    model.OBJ = pyo.Objective(expr=model.T, sense=pyo.minimize)

    instance_transitive_reduction = nx.transitive_reduction(instance)
    nodes_with_no_successors = [
        v for v, d in instance_transitive_reduction.out_degree() if d == 0]
    nodes_with_no_predecessors = [
        v for v, d in instance_transitive_reduction.in_degree() if d == 0]

    model.no_predecessors_constraint = pyo.ConstraintList()
    for node in nodes_with_no_predecessors:
        model.no_predecessors_constraint.add(expr=model.x[node] >= 0)

    model.precedence_constraints = pyo.ConstraintList()
    for node in instance_transitive_reduction.nodes():
        if node not in nodes_with_no_predecessors:
            for successor in instance_transitive_reduction.successors(node):
                model.precedence_constraints.add(
                    expr=model.x[successor] >= model.x[node] + 1)

    model.makespan_constraint = pyo.ConstraintList()
    for node in nodes_with_no_successors:
        model.makespan_constraint.add(expr=model.T >= model.x[node] + 1)

    model.processor_bound_constraint = pyo.Constraint(
        expr=sum(model.w[:, n+1]) <= processor_count)

    topologically_sorted_instance = list(topological_sort_grouped(instance))
    widths = map(len, topologically_sorted_instance)
    max_width = max(widths)
    if processor_count < max_width:
        instance_adjacency_array = nx.to_numpy_array(
            instance, nodelist=sorted(instance.nodes())).astype('int')

        # TODO ADD BEFORE TASK ORDERING CONSTRAINT
        before_expressions = {}
        for node in instance.nodes():
            before_expressions[node] = model.w[0, node]
            if node not in nodes_with_no_predecessors:
                before_expressions[node] += sum([model.w[i, node]
                                                for i in instance.predecessors(node)])

        def get_concurrent_nodes(node, instance_transitive_closure):
            node_predecessors_list = list(
                instance_transitive_closure.predecessors(node))
            node_sucessors_list = list(
                instance_transitive_closure.successors(node))
            node_rejection_list = list(
                chain([node], node_predecessors_list, node_sucessors_list))
            concurrency_list = []
            for prospective_node in instance_transitive_closure.nodes():
                if prospective_node not in node_rejection_list:
                    concurrency_list.append(prospective_node)
            return concurrency_list

        instance_transitive_closure = nx.transitive_closure_dag(instance)
        for node in sorted(instance.nodes()):
            current_node_list = get_concurrent_nodes(
                node, instance_transitive_closure)
            before_expressions[node] += sum([model.w[i, node]
                                            for i in current_node_list])

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
