from pyomo import opt
from ILP_Generators import run_instance_ilp
from utility import get_task_graph_max_width
from numpy import sign
from config import *
from create_instance import *
from itertools import islice, product
from operator import itemgetter
from signal import signal, alarm, SIGALRM
from math import log2
import time
import argparse
from multiprocessing import Pool, Event, Process, Manager
from functools import partial
from numba import jit
from pprint import pprint as pp
import os
import gurobipy
from Exhausitve import run_instance_exhaustive
from Greedy import run_instance_naive_greedy
from ILP_Generators import ILPWithImplicitProcessors
from LocalSearch import run_instance_genetic, run_instance_steepest_descent
import math

print = partial(print, flush=True)

class Timeout_Monitor:
    next_instance = False
    timeout_start = 0
    interval = 0
    @classmethod
    def get_handler(cls):
        return partial(Timeout_Monitor.set_state, cls)
    @classmethod
    def timeout(cls):
        return cls.time_remaining() == 0  # Hack to get timer to work when only one exist for all processes
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
    def time_remaining(cls):
        remaining_time = cls.interval - (time.time() - cls.timeout_start)
        remaining_time = 0 if remaining_time < 0 else remaining_time
        return remaining_time
    @classmethod
    def set_alarm(cls, timeout):
        cls.timeout_start = time.time()
        cls.interval = timeout
        alarm(timeout)
    @classmethod
    def cancel_alarm(cls):
        alarm(0)



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
                graph_instance = layer_by_layer(node_count, config.seed)
                _, bindings_solved, solve_time = method(graph_instance, processor_count, node_count, monitor)
                cum_solve_time += solve_time
                # Todo: fix how "solved instances" are identified
                if bindings_solved == processor_count**node_count:
                    solved_count += 1
            # adjust to exclude generation time
            solved_time_ratio = cum_solve_time/instance_timeout
            solved_count = int(solved_count/solved_time_ratio) if solved_time_ratio > 0 else 0
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
    elif args.method == 'ilp_implicit':
        run_instance_ilp_implicit = partial(run_instance_ilp, ILPWithImplicitProcessors)
        benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                config.node_max, config.node_min, args.timeout, config.core_count, iteration_count, False, run_instance_ilp_implicit)
    elif args.method == 'genetic':
        options = {}
        options['population_size'] = 50
        options['cut_off'] = 25
        options['mutation_rate'] = 0.25
        options['fitness_tolerance'] = 0.25
        options['max_steps_with_no_change'] = 20

        run_instance_genetic_with_options = partial(run_instance_genetic, options=options)
        benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                config.node_max, config.node_min, args.timeout, config.core_count, iteration_count, False, run_instance_genetic_with_options)

    elif args.method == 'decent':
        options = {}
        options['random_init'] = False
        run_instance_steepest_descent_with_options = partial(run_instance_steepest_descent, options=options)
        benchmark_instance = partial(benchmark, config.processor_max, config.processor_min,
                                config.node_max, config.node_min, args.timeout, config.core_count, iteration_count, False, run_instance_steepest_descent_with_options)

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

def compare_results():
    np.random.seed(config.seed)  # check correct way of seeding
    instance_timeout = 10
    monitor = Timeout_Monitor()
    monitor.register_signal()
    options = {}

    # Genetic Options
    options['population_size'] = 25
    options['cut_off'] = 10
    options['mutation_rate'] = 0.5
    options['fitness_tolerance'] = 0.25
    
    # Steepest Decent
    options['random_init'] = False

    # Both
    options['max_steps_with_no_change'] = 200    

    manager = Manager()
    ret_value_exhaustive = manager.dict()
    ret_value_greedy = manager.dict()
    ret_value_ilp = manager.dict()
    ret_value_genetic = manager.dict()
    ret_value_descent = manager.dict()
                
    for node_count in range(50, 52):
        for processor_count in range(5, 9):

            graph_instance = layer_by_layer(node_count, 0.20)
            lower_bound = utility.get_lower_bound(graph_instance)
            
            monitor.reset_state()
            
            exhaustive_proc = Process(target = run_instance_exhaustive, args=[graph_instance, processor_count, node_count, monitor, ret_value_exhaustive])
            greedy_proc = Process(target = run_instance_naive_greedy, args=[graph_instance, processor_count, node_count, monitor, ret_value_greedy])
            ilp_proc = Process(target = run_instance_ilp, args=[ILPWithImplicitProcessors, graph_instance, processor_count, node_count, monitor, ret_value_ilp])
            genetic_proc = Process(target = run_instance_genetic, args=[graph_instance, processor_count, node_count, monitor, options, ret_value_genetic])
            descent_proc = Process(target = run_instance_steepest_descent, args=[graph_instance, processor_count, node_count, monitor, options, ret_value_descent])
            
            monitor.set_alarm(instance_timeout)
            
            exhaustive_proc.start()
            greedy_proc.start()
            ilp_proc.start()
            genetic_proc.start()
            descent_proc.start()
            
            
            exhaustive_proc.join()
            greedy_proc.join()
            ilp_proc.join()
            genetic_proc.join()
            descent_proc.join()
            
            pp("ret_value_exhaustive: {}".format(ret_value_exhaustive))
            pp("ret_value_greedy: {}".format(ret_value_greedy))
            pp("ret_value_ilp: {}".format(ret_value_ilp))
            pp("ret_value_genetic: {}".format(ret_value_genetic))
            pp("ret_value_descent: {}".format(ret_value_descent))       
            print("Done")  


                                           

if __name__ == "__main__":
    compare_results()
