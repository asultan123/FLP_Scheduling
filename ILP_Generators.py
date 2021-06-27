from itertools import chain
import networkx as nx
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
from utility import *

class ILPWithImplicitProcessorBound():
    @classmethod
    def construct_model(cls, instance, processor_count, initialize_with_greedy=False):
        model = pyo.ConcreteModel()
        model = cls.create_variables(instance, model, initialize_with_greedy)
        model = cls.add_constraints(instance, model, processor_count)
        return model

    @classmethod
    def create_variables(cls, instance, model, initialize_with_greedy):
        model.T = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        if initialize_with_greedy:
            
        model.x = pyo.Var(instance.nodes(),
                          domain=pyo.NonNegativeIntegers, initialize=0)
        n = len(instance.nodes())
        w_indexes = [(i, j) for j in RangeSet(1, n+1) for i in RangeSet(0, n)]
        model.w = pyo.Var(
            w_indexes, domain=pyo.Binary, initialize=0)
        model.OBJ = pyo.Objective(expr=model.T, sense=pyo.minimize)
        return model

    @classmethod
    def add_constraints(cls, instance, model, processor_count):
        
        instance_transitive_reduction = nx.transitive_reduction(instance)
        nodes_with_no_successors = [
            v for v, d in instance_transitive_reduction.out_degree() if d == 0]
        node_count = len(instance.nodes())
        nodes_with_no_predecessors = [
            v for v, d in instance_transitive_reduction.in_degree() if d == 0]
        
        
        model = cls.add_precedence_constraints(
            instance_transitive_reduction, nodes_with_no_successors, model)
        
        model = cls.add_no_predecessors_constraint(
            nodes_with_no_predecessors, model)
        
        model = cls.add_makespan_constraint(nodes_with_no_successors, model)
        
        model = cls.add_processor_bound_constraint(
            node_count, processor_count, model)

        max_width = utility.get_task_graph_max_width(instance)
        
        if processor_count < max_width:
        
            relaxation_constant = node_count+1
        
            instance_transitive_closure = nx.transitive_closure_dag(instance)
        
            model = cls.add_task_ordering_constraints(
                instance, instance_transitive_closure, nodes_with_no_predecessors, nodes_with_no_successors, model)
        
            model = cls.add_per_processor_ordering_constraint(instance_transitive_closure, relaxation_constant, model)
        
        else:
            raise Exception("Instance width smaller than processor count, not solvable with current ILP Formulation")
            # model = cls.add_task_ordering_constraints_with_excess_processors(
            #     instance, nodes_with_no_predecessors, nodes_with_no_successors, model)

        return model


    @classmethod
    def add_per_processor_ordering_constraint(cls, instance_transitive_closure, relaxation_constant, model):
        sequential_expression = {}

        for node in instance_transitive_closure.nodes():
            for parallel_node in cls.get_concurrent_nodes(node, instance_transitive_closure):
                sequential_expression[(node, parallel_node)] = model.x[parallel_node] >= model.x[node] + \
                    1 - relaxation_constant*(1-model.w[node, parallel_node])
        
        model.force_sequential_scheduling_constraint = pyo.ConstraintList()
        for _, expr in sequential_expression.items():
            model.force_sequential_scheduling_constraint.add(expr)
        
        return model

    @classmethod
    def get_concurrent_nodes(cls, node, instance_transitive_closure):
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


    @classmethod
    def add_task_ordering_constraints(cls, instance, instance_transitive_closure, nodes_with_no_predecessors, nodes_with_no_successors, model):
        model = cls.add_backwards_task_ordering_constraints(
            instance, instance_transitive_closure, nodes_with_no_predecessors, model)
        model = cls.add_forward_task_ordering_constraints(
            instance, instance_transitive_closure, nodes_with_no_successors, model)
        return model

    @classmethod
    def add_backwards_task_ordering_constraints(cls, instance, instance_transitive_closure, nodes_with_no_predecessors, model):
        backwards_expressions = {}
        for node in instance.nodes():
            backwards_expressions[node] = model.w[0, node]
            if node not in nodes_with_no_predecessors:
                backwards_expressions[node] += sum([model.w[i, node]
                                                    for i in instance.predecessors(node)])

        for node in sorted(instance.nodes()):
            current_node_list = cls.get_concurrent_nodes(
                node, instance_transitive_closure)
            backwards_expressions[node] += sum([model.w[i, node]
                                                for i in current_node_list])

        model.backward_task_ordering_constraints = pyo.ConstraintList()

        for _, summation in backwards_expressions.items():
            model.backward_task_ordering_constraints.add(expr=summation == 1)

        return model

    @classmethod
    def add_forward_task_ordering_constraints(cls, instance, instance_transitive_closure, nodes_with_no_successors, model):
        forwards_expressions = {}
        node_count = len(instance.nodes())
        for node in instance.nodes():
            forwards_expressions[node] = model.w[node, node_count+1]
            if node not in nodes_with_no_successors:
                forwards_expressions[node] += sum([model.w[node, i]
                                                   for i in instance.successors(node)])

        for node in sorted(instance.nodes()):
            current_node_list = cls.get_concurrent_nodes(
                node, instance_transitive_closure)
            forwards_expressions[node] += sum([model.w[node, i]
                                               for i in current_node_list])

        model.forward_task_ordering_constraints = pyo.ConstraintList()

        for _, summation in forwards_expressions.items():
            model.forward_task_ordering_constraints.add(expr=summation == 1)

        return model


    @classmethod
    def add_task_ordering_constraints_with_excess_processors(cls, instance, nodes_with_no_predecessors, nodes_with_no_successors, model):
        model = cls.add_backwards_task_ordering_constraints_with_excess_processors(
            instance, nodes_with_no_predecessors, model)
        model = cls.add_forward_task_ordering_constraints_with_excess_processors(
            instance, nodes_with_no_successors, model)
        return model

    @classmethod
    def add_backwards_task_ordering_constraints_with_excess_processors(cls, instance, nodes_with_no_predecessors, model):
        backwards_expressions = {}
        for node in instance.nodes():
            backwards_expressions[node] = model.w[0, node]
            if node not in nodes_with_no_predecessors:
                backwards_expressions[node] += sum([model.w[i, node]
                                                    for i in instance.predecessors(node)])

        model.backward_task_ordering_constraints = pyo.ConstraintList()

        for _, summation in backwards_expressions.items():
            model.backward_task_ordering_constraints.add(expr=summation == 1)

        return model

    @classmethod
    def add_forward_task_ordering_constraints_with_excess_processors(cls, instance, nodes_with_no_successors, model):
        forwards_expressions = {}
        node_count = len(instance.nodes())
        for node in instance.nodes():
            forwards_expressions[node] = model.w[node, node_count+1]
            if node not in nodes_with_no_successors:
                forwards_expressions[node] += sum([model.w[node, i]
                                                   for i in instance.successors(node)])

        model.forward_task_ordering_constraints = pyo.ConstraintList()

        for _, summation in forwards_expressions.items():
            model.forward_task_ordering_constraints.add(expr=summation == 1)

        return model

    @classmethod
    def add_precedence_constraints(cls, instance_transitive_reduction, nodes_with_no_successors, model):
        model.precedence_constraints = pyo.ConstraintList()
        for node in instance_transitive_reduction.nodes():
            if node not in nodes_with_no_successors:
                for successors in instance_transitive_reduction.successors(node):
                    model.precedence_constraints.add(
                        expr=model.x[successors] >= model.x[node] + 1)
        return model

    @classmethod
    def add_no_predecessors_constraint(cls, nodes_with_no_predecessors, model):
        model.no_predecessors_constraint = pyo.ConstraintList()
        for node in nodes_with_no_predecessors:
            model.no_predecessors_constraint.add(expr=model.x[node] >= 0)
        return model

    @classmethod
    def add_makespan_constraint(cls, nodes_with_no_successors, model):
        model.makespan_constraint = pyo.ConstraintList()
        for node in nodes_with_no_successors:
            model.makespan_constraint.add(expr=model.T >= model.x[node] + 1)
        return model

    @classmethod
    def add_processor_bound_constraint(cls, node_count, processor_count, model):
        expr = model.w[1, node_count+1]
        for i in range(2,node_count+1):
            expr += model.w[i, node_count+1]
        model.processor_bound_constraint = pyo.Constraint(
            expr= expr <= processor_count)
        return model

