from itertools import chain
import networkx as nx
from itertools import product
from operator import itemgetter

def get_lower_bound(instance):
    return len(list(topological_sort_grouped(instance)))

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

def get_task_graph_max_width(instance):
    topologically_sorted_instance = list(topological_sort_grouped(instance))
    widths = map(len, topologically_sorted_instance)
    max_width = max(widths)
    return max_width

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
        
def topological_sort_grouped_with_terminal_node_reposition(G):
    G_grouped = topological_sort_grouped(G)
    nodes_with_no_successors = [node for node in G.nodes() if len(list(G.successors(node))) == 0]
    adjusted_G_grouped = []
    for group in G_grouped:
        new_group = []
        for node in group:
            if node not in nodes_with_no_successors:
                new_group.append(node)
        adjusted_G_grouped.append(new_group)
    
    adjusted_G_grouped[-1] = nodes_with_no_successors
    
    return adjusted_G_grouped

def change_node_start_index(G, topologically_sorted_group, start_idx = 1):
    G_flattened = list(chain(*topologically_sorted_group))
    
    new_labels = {n: i+start_idx for i, n in enumerate(G_flattened)}
    G = nx.relabel_nodes(G, new_labels)
    
    return G
        