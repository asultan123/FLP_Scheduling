from config import *
from create_instance import *
from itertools import chain, product
from operator import itemgetter
import signal
from math import log2
import time
next_instance = False

def next_binding(signo, frame):
    global next_instance
    next_instance = True

signal.signal(signal.SIGALRM, next_binding)

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

def schedule(binding, grouped_top_sort):
    sched = {}
    time = 0
    group_offset = 0
    for group in grouped_top_sort:
        group_binding = binding[group_offset : group_offset+len(group)]
        sorted_group = sorted(list(zip(group, group_binding)), key=itemgetter(1))
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

def run_instance_exhaustive(processors, nodes, edge_prob):
    graph_instance = layer_by_layer(nodes, edge_prob, config.seed)
    grouped_top_sort = list(topological_sort_grouped(graph_instance))
    latencies = {}
    solved_count = 0
    for selected_binding in bindings(processors, nodes):
        sched = schedule(selected_binding, grouped_top_sort)
        latencies[solved_count] = (sched, selected_binding, max_latency(sched))
        solved_count += 1
        if next_instance:
            break
    return latencies, solved_count

def benchmark(processor_max, node_max):
    global next_instance
    instance_idx = 0
    instance_schedules = {}
    
    for processor_pow in range(2,int(log2(processor_max))+1):
        for node_pow in range(2,int(log2(node_max))+1):
            processor_count = 2**processor_pow
            node_count = 2**node_pow
            print("Current instance m:{}, n:{}".format(processor_count, node_count))
            signal.alarm(instance_timeout)
            solved_count = 0
            while not next_instance:
                latencies, bindings_solved = run_instance_exhaustive(processor_count, node_count, edge_prob=np.random.rand())
                instance_schedules[instance_idx] = latencies
                if bindings_solved == processor_count**node_count:
                    solved_count += 1
            print("Solved {} instances in {} sec".format(solved_count, end_time-start_time))
            next_instance = False


benchmark(8,16)