from config import *
import time
from utility import *

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
        
def get_greedy_schedule(graph_instance, processor_count):
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    selected_binding = distributed_load_binding(grouped_top_sort, processor_count)
    greedy_sched = schedule(selected_binding, grouped_top_sort)
    max_latency_sched = makespan(greedy_sched)
    return greedy_sched, max_latency_sched
        
def run_instance_naive_greedy(graph_instance, processor_count, node_count, monitor, ret_value = None):
    solve_start = time.time()
    max_latency_sched, greedy_sched = get_greedy_schedule(graph_instance, processor_count)
    # TODO Fix instance solved identifier
    bindings_solved = processor_count**node_count #equivelent space "searched"
    solve_end = time.time()
    if ret_value is not None:
        ret_value["makespan"] = max_latency_sched
        ret_value["sched"] = greedy_sched
    return (max_latency_sched, greedy_sched), bindings_solved, solve_end-solve_start