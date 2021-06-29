from numpy.lib.function_base import average
from config import *
import time
from utility import *
from Greedy import distributed_load_binding
import numpy as np
from math import floor
from create_instance import layer_by_layer
import config
import utility

def run_instance_steepest_descent(graph_instance, processor_count, node_count, monitor, options, ret_value = None):
    solve_start = time.time()

    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    # TODO Fix instance solved identifier
    bindings_solved = processor_count**node_count  # equivelent space "searched"

    if options is not None and "random_init" in options and options["random_init"] == True:
        # Random initial binding
        selected_binding = np.random.randint(
            processor_count, size=node_count).tolist()
        random_init = True
    else:
        # Initial binding based on a greedy approach
        selected_binding = distributed_load_binding(
            grouped_top_sort, processor_count)
        random_init = False

    # Makespan (cost) of initial solution
    init_sched = schedule(selected_binding, grouped_top_sort)
    init_makespan = makespan(init_sched)
    # print("Initial ({}) solution makespan: {}".format(
    #     "random" if random_init == True else "greedy", init_makespan))

    makespan_best = node_count  # start with a big value
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

        # Move to best neighbor if better or equal to current solution
        if makespan_best < selected_makespan:
            equal_makespans = 0
            selected_makespan = makespan_best
            selected_binding = binding_best.copy()
        elif makespan_best == selected_makespan:
            equal_makespans += 1
            selected_binding = binding_best.copy()
#        print(selected_makespan)

    # print("Best solution makespan: {}".format(selected_makespan))
    selected_sched = schedule(selected_binding, grouped_top_sort)
    solve_end = time.time()
    if ret_value is not None:
        ret_value["makespan"] = selected_makespan
        ret_value["sched"] = selected_sched
    return (selected_makespan, selected_sched), bindings_solved, solve_end-solve_start


def local_search(individual, grouped_top_sort, processor_count, node_count):
    init_sched = schedule(individual, grouped_top_sort)
    init_makespan = makespan(init_sched)

    makespan_best = init_makespan  # start with a big value
    selected_binding = individual
    binding_best = individual
    
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
            if makespan_n <= makespan_best:
                makespan_best = makespan_n
                binding_best = selected_binding_n.copy()

    return binding_best


def eval_individual_fitness(individual, grouped_top_sort):
    sched = schedule(individual, grouped_top_sort)
    makespan_ind = makespan(sched)
    return makespan_ind


def eval_population_fitness(population, grouped_top_sort):
    population_fitness = [(individual, eval_individual_fitness(
        individual, grouped_top_sort)) for individual in population]
    avg_population_fitness = average(
        [individual[1] for individual in population_fitness])
    normalized_population_fitness = [
        (individual[0], individual[1]/avg_population_fitness) for individual in population_fitness]
    return normalized_population_fitness, avg_population_fitness


def mutation_operator(individual, mutation_rate, processor_count, node_count):
    if np.random.random() <= mutation_rate:
        target_binding = np.random.randint(node_count)
        individual[target_binding] = np.random.randint(processor_count)
    return individual


def mutate_population(population, mutation_rate, processor_count, node_count):
    mutated_population = [mutation_operator(
        individual, mutation_rate, processor_count, node_count) for individual in population]
    return mutated_population


def selection_operator(population, cutoff, grouped_top_sort):
    normalized_population_fitness, _ = eval_population_fitness(
        population, grouped_top_sort)
    sorted_population_based_on_fitness = list(
        sorted(normalized_population_fitness, key=lambda entry: entry[1]))
    surviving_population = sorted_population_based_on_fitness[:cutoff]
    surviving_population = [individual[0] for individual in surviving_population]
    return surviving_population


def crossover_operator(population, population_size, node_count):
    first_parent = np.random.randint(population_size)
    second_parent = np.random.randint(population_size)
    while second_parent == first_parent:
        second_parent = np.random.randint(population_size)

    first_parent = population[first_parent]
    second_parent = population[second_parent]

    offspring = [0]*node_count
    for i in range(node_count):
        offspring[i] = first_parent[i] if np.random.random(
        ) <= 0.5 else second_parent[i]
    return offspring


def apply_mating_strategy(population, population_size, node_count):
    new_population = population
    for _ in range(population_size):
        new_population.append(crossover_operator(
            population, population_size, node_count))
    return new_population


def random_population_initialization(processor_count, node_count, population_size):
    population = np.random.randint(processor_count, size=(
        population_size, node_count)).tolist()
    return population


def run_instance_genetic(graph_instance, processor_count, node_count, monitor, options, ret_value = None):
    solve_start = time.time()

    grouped_top_sort = list(topological_sort_grouped(graph_instance))

    population_size = options['population_size']
    cut_off = options['cut_off']
    mutation_rate = options['mutation_rate']
    fitness_tolerance = options['fitness_tolerance']
    max_steps_with_no_change = options['max_steps_with_no_change']

    population = random_population_initialization(
        processor_count, node_count, population_size)
    best_binding = selection_operator(
        population, 1, grouped_top_sort)[0]
    best_sched = schedule(best_binding, grouped_top_sort)
    best_makespan = makespan(best_sched)
    
    steps_with_no_change = 0
    last_avg_population_fitness = 0
    while not monitor.timeout():
        for individual in population:
            individual = local_search(
                individual, grouped_top_sort, processor_count, node_count)
        population = selection_operator(
            population, cut_off, grouped_top_sort)

        population = mutate_population(
            population, mutation_rate, processor_count, node_count)
        population = apply_mating_strategy(population, cut_off, node_count)

        _, avg_population_fitness = eval_population_fitness(
            population, grouped_top_sort)

        if abs(avg_population_fitness - last_avg_population_fitness) <= fitness_tolerance:
            steps_with_no_change += 1
        else:
            steps_with_no_change = 0

        if steps_with_no_change == max_steps_with_no_change:
            break
        
        last_avg_population_fitness = avg_population_fitness
        candidate_best_binding = selection_operator(
            population, 1, grouped_top_sort)[0]
        candidate_best_sched = schedule(candidate_best_binding, grouped_top_sort)
        candidate_best_makespan = makespan(candidate_best_sched)

        if best_makespan >= candidate_best_makespan:
            best_binding = candidate_best_binding
            best_sched = candidate_best_sched
            best_makespan = candidate_best_makespan
            
        # print("Average population fitness: {}".format(avg_population_fitness))

    # print("Best Makespan: {}".format(best_makespan))
    bindings_solved = processor_count**node_count  # equivelent space "searched"
    solve_end = time.time()
    if ret_value is not None:
        ret_value["makespan"] = best_makespan
        ret_value["sched"] = best_sched
    return (best_makespan, best_sched), bindings_solved, solve_end-solve_start


def genetic_instance_test():
    np.random.seed(config.seed)
    node_count = 20
    instance = layer_by_layer(node_count, 0.25, plot_graphs=False)
    processor_count = 4
    lower_bound = utility.get_lower_bound(instance)
    print("Instance Lower Bound {}".format(lower_bound))
    
    options = {}
    options['population_size'] = 50
    options['cut_off'] = 25
    options['mutation_rate'] = 0.25
    options['fitness_tolerance'] = 0.25
    options['max_steps_with_no_change'] = 20

    (best_makespan, best_sched), bindings_solved, solve_time = run_instance_genetic(instance, processor_count, node_count, None, options)
    return best_makespan, best_sched


def steepest_decent_instance_test():
    np.random.seed(config.seed)
    node_count = 100
    instance = layer_by_layer(node_count, 0.25, plot_graphs=False)
    processor_count = 4
    lower_bound = utility.get_lower_bound(instance)
    print("Instance Lower Bound {}".format(lower_bound))
    
    run_instance_steepest_descent(instance, processor_count, node_count, None, None)
