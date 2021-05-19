from config import *
from create_instance import *
from itertools import chain, product
from operator import itemgetter

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

processors, nodes = 4, 32


def run_instance():
    graph_instance = layer_by_layer(nodes, 0.1, 1)
    grouped_top_sort = topological_sort_grouped(graph_instance)
    task_bindings = bindings(processors, nodes)
    selected_binding = next(task_bindings)
    sched = schedule(selected_binding, grouped_top_sort)
    return max_latency(sched)

print(run_instance())
