from numpy import sign
from config import *
from create_instance import *
from itertools import islice, product
from operator import itemgetter
from signal import signal, alarm, SIGALRM
from math import log2
import time
import argparse
from multiprocessing import Pool, Event
from functools import partial
from numba import jit
from pprint import pprint as pp


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

# @jit(nopython=True)


def schedule(binding, grouped_top_sort):
    sched = {}
    time = 0
    group_offset = 0
    for group in grouped_top_sort:
        group_binding = binding[group_offset: group_offset+len(group)]
        sorted_group = sorted(
            list(zip(group, group_binding)), key=itemgetter(1))
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


def max_latency(sched):
    return max(sched.values()) + 1

# @jit


def benchmark(processor_max, processor_min, node_max, node_min, instance_timeout, worker_count, iteration_count, start_idx):

    next_instance = False

    def next_binding(signo, frame):
        nonlocal next_instance
        next_instance = True

    signal(SIGALRM, next_binding)

    def run_instance_exhaustive(processors, nodes, edge_prob):
        graph_instance = layer_by_layer(nodes, edge_prob, config.seed)
        solve_start = time.time()
        grouped_top_sort = list(topological_sort_grouped(graph_instance))
        min_max_sched_latency = None
        opt_sched = None
        bindings_solved = 0
        nonlocal next_instance
        for selected_binding in bindings(processors, nodes):
            sched = schedule(selected_binding, grouped_top_sort)
            max_latency_sched = max_latency(sched)
            if min_max_sched_latency is not None and min_max_sched_latency > max_latency_sched:
                min_max_sched_latency = max_latency_sched
                opt_sched = sched
            bindings_solved += 1
            if next_instance:
                break
        solve_end = time.time()
        return (min_max_sched_latency, opt_sched), bindings_solved, solve_end-solve_start

    chunk_size = int(iteration_count/worker_count)
    process_idx = int(start_idx/chunk_size)
    iters = product(range(node_min, node_max+1),
                    range(processor_min, processor_max+1))
    iters = list(islice(iters, start_idx, start_idx+chunk_size))
    solve_log = {(node_count, processor_count): 0 for (
        node_count, processor_count) in iters}
    nodes, processors = zip(*iters)
    nodes, processors = list(set(nodes)), list(set(processors))
    print("Spawned process {} starting at benchmark iteration {}".format(
        process_idx, start_idx))

    current_processor_upper_bound = processor_max
    for node_count in nodes:
        for processor_count in processors:
            if processor_count > current_processor_upper_bound:
                break
            print("Process {}: Current instance n:{}, m:{}".format(
                process_idx, node_count, processor_count))
            solved_count = 0
            cum_solve_time = 0
            alarm(instance_timeout)
            while not next_instance:
                _, bindings_solved, solve_time = run_instance_exhaustive(
                    processor_count, node_count, edge_prob=np.random.rand())
                cum_solve_time += solve_time
                if bindings_solved == processor_count**node_count:
                    solved_count += 1
            # adjust to exclude generation time
            solved_time_ratio = cum_solve_time/instance_timeout
            solved_count = int(solved_count/solved_time_ratio)
            print("Process {}: Solved ~{} instances".format(
                process_idx, solved_count))
            solve_log[(node_count, processor_count)] = solved_count
            next_instance = False
            if(solved_count == 0):
                current_processor_upper_bound -= 1
                if current_processor_upper_bound >= processor_min:
                    print("Process {}: defined upper bound at n:{} m:{} for timeout: {} ... constraining processor upper limit to m:{}".format(
                        process_idx, node_count, processor_count, instance_timeout, current_processor_upper_bound))
                    break
                else:
                    print("Process {}: defined final upper bound at n:{} m:{} for timeout: {} ... terminating...".format(
                        process_idx, node_count, processor_count, instance_timeout))
                    return solve_log
            with open("proc_{}_timeout_{}.bin".format(process_idx, instance_timeout), 'wb') as file:
                import pickle
                pickle.dump(solve_log, file)

    return solve_log


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('timeout', type=int)
    args = parser.parse_args()

    iteration_count = abs(
        config.processor_max-config.processor_min+1)*abs(config.node_max-config.node_min+1)

    benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                 config.node_max, config.node_min, args.timeout, config.core_count, iteration_count)

    aggregate_solve_log = {}

    with Pool(config.core_count) as benchmark_pool:
        start_idx = list(range(0, iteration_count, int(
            iteration_count/config.core_count)))
        worker_solve_log = benchmark_pool.map(benchmark_instance, start_idx)
    for solve_log in worker_solve_log:
        for (nodes, processors), solves in solve_log.items():
            aggregate_solve_log[(nodes, processors)] = solves

    pp(aggregate_solve_log)
    # print(benchmark(processor_max,node_max, args.timeout))
    with open("benchmark_timeout_{}.bin".format(args.timeout), 'wb') as file:
        import pickle
        pickle.dump(aggregate_solve_log, file)

if __name__ == "__main__":
    main()
