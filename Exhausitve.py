from config import *
from create_instance import *
import time
from utility import *

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
