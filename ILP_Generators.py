from itertools import chain
import networkx as nx
from pyomo.core.base.set import RangeSet
import pyomo.environ as pyo
import utility
from Greedy import get_greedy_schedule
import gurobipy as gp
from gurobipy import GRB
from functools import partial
from collections import OrderedDict
import time
import config
from pyomo.opt import SolverStatus, TerminationCondition

def run_instance_ilp(formulation, graph_instance, processor_count, node_count, monitor, ret_value):
    solve_start = time.time()
    lower_bound = utility.get_lower_bound(graph_instance)
    bindings_solved = 0
    makespan = 0
    schedule = {}
    if processor_count < utility.get_task_graph_max_width(graph_instance):
        model = formulation.construct_model(graph_instance, processor_count, get_timeout = monitor.time_remaining)
        results = model.solve()
        if results is not None:
            makespan = model.get_makespan()
            schedule = model.get_schedule()
            if results.solver.status != SolverStatus.ok:
                if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                    bindings_solved = 0 #equivelent space "searched"
                else:
                    raise Exception("Invalid solver termination state")
            else:
                bindings_solved = processor_count**node_count #equivelent space "searched"

        else:
            # todo check why this execution path may happen (why would solver fail to return in case of other ILP)
            bindings_solved = 0
    else:
        bindings_solved = processor_count**node_count
        schedule, makespan = get_greedy_schedule(graph_instance, processor_count)
    solve_end = time.time()
    if ret_value is not None:
        ret_value["makespan"] = makespan
        ret_value["sched"] = schedule
    return (makespan, schedule), bindings_solved, solve_end-solve_start

class ILPSchedulingModel():
    def __init__(self, model, optimizer, solve_method, get_schedule_method, get_makespan_method, get_model_description_method=None):
        self._model = model
        self._solve_method = solve_method
        self._get_schedule_method = get_schedule_method
        self._get_makespan_method = get_makespan_method
        self._description_method = get_model_description_method
        self._optimizer = optimizer

    def solve(self):
        try:
            return self._solve_method()
        except:
            return None

    def description(self):
        self._description_method()

    def get_makespan(self):
        return self._get_makespan_method()

    def get_schedule(self):
        return self._get_schedule_method()

    def get_model(self):
        return self._model
    
    def get_optimizer(self):
        return self._optimizer


class ILPWithExplicitProcessors():
    @classmethod
    def construct_model(cls, instance, processor_count, lower_bound= None, get_timeout = None):
        edges = list(instance.edges)
        n_edges = instance.number_of_edges()
        model = gp.Model("ilp")
        nodes = list(range(1, 1+len(instance.nodes())))
        T = model.addVar(vtype=GRB.INTEGER, name="makespan")
        t = model.addVars(nodes, vtype=GRB.INTEGER,
                          name="task_start_time")
        p = model.addVars(nodes, vtype=GRB.INTEGER, name="task_processor")
        y = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="y")
        model.setObjective(T, GRB.MINIMIZE)
        # Constraints for relaxed (no constraint on number of processors) version
        model.addConstrs((t[i] + 1 <= T for i in nodes), "c0")
        model.addConstrs((t[i] >= 0 for i in nodes), "c1")
        model.addConstrs((p[i] >= 0 for i in nodes), "c2")
        model.addConstrs((t[edges[i][0]] + 1 <= t[edges[i][1]]
                         for i in range(n_edges)), "c3")
        model.addConstrs(((y[i, j] == 0) >> (p[i] + t[i]*processor_count >= p[j] + t[j]*processor_count + 1) for i in nodes for j in nodes if j != i), "c4")
        model.addConstrs(((y[i, j] == 1) >> (p[i] + t[i]*processor_count <= p[j] + t[j]*processor_count - 1) for i in nodes for j in nodes if j != i), "c5")
        model.addConstrs((p[i] <= processor_count - 1 for i in nodes), "c6")

        def get_schedule_method(start_times, nodes):
            try:
                task_start_times = {
                    node: int(start_times[node].x) for node in nodes}
                sorted_task_start_times = sorted(
                    task_start_times.items(), key=lambda node: node[1])
                return OrderedDict(sorted_task_start_times)
            except:
                print("Solver did not find a solution")
                return None     

        def get_makespan_method(model):
            try:
                return int(model.objVal)
            except:
                print("Solver did not find a solution")
                return None                     
        
        if get_timeout is not None:
            model.Params.TimeLimit = get_timeout()

        if lower_bound is not None:
            model.Params.BestObjStop = lower_bound

        model.Params.Threads = config.gurobi_threads

        model_with_wrapper = ILPSchedulingModel(model,
                                                solve_method=model.optimize,
                                                get_schedule_method=partial(
                                                    get_schedule_method, t, nodes),
                                                get_makespan_method=partial(
                                                    get_makespan_method, model))

        return model_with_wrapper


class ILPWithImplicitProcessors():
    @classmethod
    def construct_model(cls, instance, processor_count, lower_bound = None, get_timeout= None, initialize_with_greedy=False):
        model = pyo.ConcreteModel()
        model = cls.create_variables(
            instance, model, processor_count, initialize_with_greedy)
        model = cls.add_constraints(instance, model, processor_count)

        def get_schedule_method(model):
            task_start_times = {node: int(model.x[node]()) for node in model.x}
            sorted_task_start_times = sorted(
                task_start_times.items(), key=lambda node: node[1])
            return OrderedDict(sorted_task_start_times)

        def get_makespan_method(model):
            return int(model.T())

        optimizer = pyo.SolverFactory('gurobi')
        if get_timeout is not None:
            optimizer.options['TimeLimit'] = get_timeout()

        if lower_bound is not None:
            optimizer.options['BestObjStop'] = lower_bound

        optimizer.options['Threads'] = config.gurobi_threads

        model_with_wrapper = ILPSchedulingModel(model,
                                                optimizer,
                                                solve_method=partial(optimizer.solve, model),
                                                get_schedule_method=partial(
                                                    get_schedule_method, model),
                                                get_makespan_method=partial(
                                                    get_makespan_method, model),
                                                get_model_description_method=model.pprint)

        return model_with_wrapper

    @classmethod
    def create_variables(cls, instance, model, processor_count, initialize_with_greedy):
        model.T = pyo.Var(domain=pyo.NonNegativeIntegers, initialize=0)
        if initialize_with_greedy:
            initialization_sched, _ = get_greedy_schedule(
                instance, processor_count)
        else:
            initialization_sched = 0
        model.x = pyo.Var(instance.nodes(),
                          domain=pyo.NonNegativeIntegers, initialize=initialization_sched)
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

            model = cls.add_per_processor_ordering_constraint(
                instance_transitive_closure, relaxation_constant, model)

        else:
            raise Exception(
                "Instance width smaller than processor count, not solvable with current ILP Formulation")
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
        for i in range(2, node_count+1):
            expr += model.w[i, node_count+1]
        model.processor_bound_constraint = pyo.Constraint(
            expr=expr <= processor_count)
        return model
