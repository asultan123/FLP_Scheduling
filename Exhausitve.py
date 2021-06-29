from config import *
from create_instance import *
import time
from utility import *
import math

def run_instance_exhaustive(graph_instance, processors, nodes, monitor, ret_value = None):
    solve_start = time.time()
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    opt_sched_latency = math.inf
    opt_sched = None
    bindings_solved = 0
    for selected_binding in bindings(processors, nodes):
        sched = schedule(selected_binding, grouped_top_sort)
        max_latency_sched = makespan(sched)
        if opt_sched_latency > max_latency_sched:
            opt_sched_latency = max_latency_sched
            opt_sched = sched
        bindings_solved += 1
        if monitor.timeout():
            break
    solve_end = time.time()
    if ret_value is not None:
        ret_value["makespan"] = opt_sched_latency
        ret_value["sched"] = opt_sched
    return (opt_sched_latency, opt_sched), bindings_solved, solve_end-solve_start
