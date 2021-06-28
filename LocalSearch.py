from config import *
import time
from utility import *
from Greedy import distributed_load_binding
import numpy as np

def run_instance_steepest_descent(graph_instance, processor_count, node_count, monitor, options):
    solve_start = time.time()
    
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    # TODO Fix instance solved identifier
    bindings_solved = processor_count**node_count #equivelent space "searched"

    if options is not None and "random_init" in options and options["random_init"] == True:
        # Random initial binding
        selected_binding = np.random.randint(processor_count, size=node_count).tolist()
        random_init = True
    else:
        # Initial binding based on a greedy approach
        selected_binding = distributed_load_binding(grouped_top_sort, processor_count)
        random_init = False

    # Makespan (cost) of initial solution
    init_sched = schedule(selected_binding, grouped_top_sort)
    init_makespan = makespan(init_sched)
    print("Initial ({}) solution makespan: {}".format("random" if random_init==True else "greedy", init_makespan))


    makespan_best = node_count # start with a big value
    selected_makespan = init_makespan
    equal_makespans = 0
    # Stop search when no better solutions are found after 20 steps
    while equal_makespans < 20 and not monitor.timeout():
        # Neighborhood search
        for i in range(node_count):
            for j in range(processor_count):
                if selected_binding[i] == j:
                    continue
                # Evaluate neighbor
                selected_binding_n = selected_binding.copy()
                selected_binding_n[i] = j
                sched_n = schedule(selected_binding_n, grouped_top_sort)
                makespan_n = makespan(sched_n)
                # Include equal solutions, for a wider exploration of the neighborhood
                if makespan_best >= makespan_n:
                    makespan_best = makespan_n
                    binding_best = selected_binding_n.copy()
                
                if monitor.timeout():
                    break
        # Move to best neighbor if better or equal to current solution
        if makespan_best < selected_makespan:
            equal_makespans = 0
            selected_makespan = makespan_best
            selected_binding = binding_best.copy()
        elif makespan_best == selected_makespan:
            equal_makespans += 1
            selected_binding = binding_best.copy()
#        print(selected_makespan)

    print("Best solution makespan: {}".format(selected_makespan))
    selected_sched = schedule(selected_binding, grouped_top_sort)
    solve_end = time.time()
    return (selected_makespan, selected_sched), bindings_solved, solve_end-solve_start


def run_instance_genetic(graph_instance, processor_count, node_count, monitor, options):
    solve_start = time.time()
    
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    # TODO Fix instance solved identifier
    bindings_solved = processor_count**node_count #equivelent space "searched"

    if options is not None and "random_init" in options and options["random_init"] == True:
        # Random initial binding
        selected_binding = np.random.randint(processor_count, size=node_count).tolist()
        random_init = True
    else:
        # Initial binding based on a greedy approach
        selected_binding = distributed_load_binding(grouped_top_sort, processor_count)
        random_init = False

    # Makespan (cost) of initial solution
    init_sched = schedule(selected_binding, grouped_top_sort)
    init_makespan = makespan(init_sched)
    print("Initial ({}) solution makespan: {}".format("random" if random_init==True else "greedy", init_makespan))


    makespan_best = node_count # start with a big value
    selected_makespan = init_makespan
    equal_makespans = 0
    # Stop search when no better solutions are found after 20 steps
    while equal_makespans < 20:
        # Neighborhood search
        for i in range(node_count):
            for j in range(processor_count):
                if selected_binding[i] == j:
                    continue
                # Evaluate neighbor
                selected_binding_n = selected_binding.copy()
                selected_binding_n[i] = j
                sched_n = schedule(selected_binding_n, grouped_top_sort)
                makespan_n = makespan(sched_n)
                # Include equal solutions, for a wider exploration of the neighborhood
                if makespan_best >= makespan_n:
                    makespan_best = makespan_n
                    binding_best = selected_binding_n.copy()
        # Move to best neighbor if better or equal to current solution
        if makespan_best < selected_makespan:
            equal_makespans = 0
            selected_makespan = makespan_best
            selected_binding = binding_best.copy()
        elif makespan_best == selected_makespan:
            equal_makespans += 1
            selected_binding = binding_best.copy()
#        print(selected_makespan)

    print("Best solution makespan: {}".format(selected_makespan))
    selected_sched = schedule(selected_binding, grouped_top_sort)
    solve_end = time.time()
    return (selected_makespan, selected_sched), bindings_solved, solve_end-solve_start