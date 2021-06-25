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
import os
import gurobipy

print = partial(print, flush=True)

def bindings(processors, nodes):
    return product(*[range(processors)]*nodes)

def schedule(binding, grouped_top_sort):
    sched = {}
    time = 0
    group_offset = 0
    for group in grouped_top_sort:
        group_time = time
        group_binding = binding[group_offset: group_offset+len(group)]
        sorted_group = sorted(
            list(zip(group, group_binding)), key=itemgetter(1))
        last_processor = sorted_group[0][1]
        sched[sorted_group[0][0]] = group_time
        for idx, (task, task_binding) in enumerate(sorted_group):
            if idx == 0:
                continue
            group_time = group_time + 1 if task_binding == last_processor else time
            sched[task] = group_time
            last_processor = task_binding
        group_offset += len(group)
        time = makespan(sched)
    return sched

def makespan(sched):
    return max(sched.values()) + 1

class Timeout_Monitor:
    next_instance = False
    @classmethod
    def get_handler(cls):
        return partial(Timeout_Monitor.set_state, cls)
    @classmethod
    def timeout(cls):
        return cls.next_instance 
    @classmethod
    def reset_state(cls):
        cls.next_instance = False
    @classmethod
    def set_state(cls, *args):
        cls.next_instance = True
    @classmethod
    def register_signal(cls):
        signal(SIGALRM, cls.get_handler())
    @classmethod
    def set_alarm(cls, timeout):
        alarm(timeout)

def run_instance_exhaustive(graph_instance, processors, nodes, monitor):
    solve_start = time.time()
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    # TODO: Fix optimal_sched_latency return
    opt_sched_latency = None
    opt_sched = None
    bindings_solved = 0
    for selected_binding in bindings(processors, nodes):
        sched = schedule(selected_binding, grouped_top_sort)
        max_latency_sched = makespan(sched)
        if opt_sched_latency is not None and opt_sched_latency > max_latency_sched:
            opt_sched_latency = max_latency_sched
            opt_sched = sched
        bindings_solved += 1
        if monitor.timeout():
            break
    solve_end = time.time()
    return (opt_sched_latency, opt_sched), bindings_solved, solve_end-solve_start

def distributed_load_binding(grouped_top_sort, processors):
    binding = []
    for group in grouped_top_sort:
        binding_counter = 0
        for _ in group:
            binding.append(binding_counter)
            binding_counter += 1
            if binding_counter == processors:
                binding_counter = 0
    return binding
        
def run_instance_naive_greedy(graph_instance, processor_count, node_count, monitor):
    solve_start = time.time()
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    # TODO Fix instance solved identifier
    bindings_solved = processor_count**node_count #equivelent space "searched"
    selected_binding = distributed_load_binding(grouped_top_sort, processor_count)
    greedy_sched = schedule(selected_binding, grouped_top_sort)
    max_latency_sched = makespan(greedy_sched)
    solve_end = time.time()
    return (max_latency_sched, greedy_sched), bindings_solved, solve_end-solve_start

def benchmark(processor_max, processor_min, node_max, node_min, instance_timeout, worker_count, iteration_count, allow_early_termination, method, start_idx):

    monitor = Timeout_Monitor()
    monitor.register_signal()

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
    cur_it = 0
    for node_count in nodes:
        for processor_count in processors:
            if processor_count > current_processor_upper_bound:
                break
            print("Process {}: Current instance n:{}, m:{}".format(
                process_idx, node_count, processor_count))
            solved_count = 0
            cum_solve_time = 0
            monitor.reset_state()
            monitor.set_alarm(instance_timeout)
            while not monitor.timeout():
                graph_instance = layer_by_layer(node_count, np.random.rand(), config.seed)
                _, bindings_solved, solve_time = method(graph_instance, processor_count, node_count, monitor)
                cum_solve_time += solve_time
                # Todo: fix how "solved instances" are identified
                if bindings_solved == processor_count**node_count:
                    solved_count += 1
            # adjust to exclude generation time
            solved_time_ratio = cum_solve_time/instance_timeout
            solved_count = int(solved_count/solved_time_ratio)
            print("Process {}: Solved ~{} instances".format(
                process_idx, solved_count))
            solve_log[(node_count, processor_count)] = solved_count
            if(allow_early_termination and solved_count == 0):
                current_processor_upper_bound -= 1
                if cur_it == 0 or current_processor_upper_bound < processor_min:
                    print("Process {}: defined final upper bound at n:{} m:{} for timeout: {} ... terminating...".format(
                        process_idx, node_count, processor_count, instance_timeout))
                    return solve_log
                if current_processor_upper_bound >= processor_min:
                    print("Process {}: defined upper bound at n:{} m:{} for timeout: {} ... constraining processor upper limit to m:{}".format(
                        process_idx, node_count, processor_count, instance_timeout, current_processor_upper_bound))
                    break
            cur_it += 1
            if config.log_results:
                with open("./Data/proc_{}_timeout_{}.bin".format(process_idx, instance_timeout), 'wb') as file:
                    import pickle
                    pickle.dump(solve_log, file)
    print("Process {}: completed benchmarking ... terminating ".format(
        process_idx, node_count, processor_count, instance_timeout, current_processor_upper_bound))
    return solve_log

def main():
    np.random.seed(config.seed)  # check correct way of seeding
    if not os.path.exists('./Data'):
        os.makedirs('./Data')

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('timeout', type=int)
    parser.add_argument('--method', default='exhaustive')
    args = parser.parse_args()

    iteration_count = abs(
        config.processor_max-config.processor_min+1)*abs(config.node_max-config.node_min+1)

    if args.method == 'exhaustive':
        benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                 config.node_max, config.node_min, args.timeout, config.core_count, iteration_count, True, run_instance_exhaustive)
    elif args.method == 'naive-greedy':
        benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                config.node_max, config.node_min, args.timeout, config.core_count, iteration_count, False, run_instance_naive_greedy)
    elif args.method == 'ilp_1':
        # TODO ADD INSTANCE PRE-CONVERSION TO ILP MODEL
        pass


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
    if config.log_results:
        with open("./Data/benchmark_timeout_{}.bin".format(args.timeout), 'wb') as file:
            import pickle
            pickle.dump(aggregate_solve_log, file)


if __name__ == "__main__":
    main()
