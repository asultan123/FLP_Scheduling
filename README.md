# FLP_Scheduling

A benchmark for testing multiapplication scheduling techniques on heterogeneous platforms like: https://doi.org/10.1007/s11265-015-1058-5. 
Simulation of benchmarked techniques available at: https://github.com/neu-ece-esl/FLP_Scheduling

## Usage

### To install poetry
(Assuming linux like enviornment)

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

### To install virtual enviornment in current directory (useful if vscode isn't picking up on venv)
poetry config virtualenvs.in-project true

### To install dependencies
poetry install

### To start virtual enviornment
poetry shell

## Synthetic task graph generation
The layer-by-layer approach described in [1] is used to generate appropriate task graphs that represent dependency relations that exist in many real world applications.

## Techniques

The technique list below will be updated everytime a new technique is added. 

### Exhaustive Search

The basis for the exhaustive search algorithm is the exploration of all possible task-processor bindings
and the schedules that arise as a consequence of that binding + partial task ordering. Validating that a
schedule meets the required individual task deadlines only serves to eliminate a schedule after it has
been produced; not during. The exhaustive search algorithm does not take into account whether a
partial schedule is valid or not mid generation. That would be an optimization over the exhaustive
search algorithm. Since 1) the schedule validation does not affect exhaustive search exploration and 2)
the generation of individual task deadlines can only be done arbitrarily; the additional validation step
has not been included to the exhaustive search algorithm used in this benchmark. This will likely skew
results to reflect more instance variations solved than what would be realistic due to the lack of
validation overhead in the algorithm. This validation step will likely be necessary for exploration in other
less naïve algorithms that will use it to trim to search space. When those algorithms are added to the
benchmark in the future, for the sake of fairness, this validation step will be added to the exhaustive
search algorithm.

#### The steps taken by the algorithm are:
1. Apply a grouped topological sort to the graph instance to extract tasks that can be executed in
parallel
2. Enumerate all possible bindings for tasks(nodes) and processors
3. For each binding, generate a valid schedule that does not violate partial order using grouped
topological sort output
4. Keep track of schedule with lowest max latency

The number of bindings to explore scales exponentially with the processor count and node count. This is
where the search space size claim given above comes from. For each task(node) there are processor
count possibilities as to where it can execute. That is irrespective of what tasks have already been
assigned to the available processors. As long as the schedule is adjusted to run tasks in time in a way
that respects partial order in the task graph, any binding can be valid from a partial order perspective.
This does not apply to individual task deadlines or overall task graph deadlines. Those can still eliminate
schedules/ bindings. If we assume a scheduling technique that attempts to maximize task parallelism.
Each binding can only have one valid schedule that maximizes parallelism while respecting partial order.
The number of edges in a task graph is irrelevant because it only affects the maximum amount of
parallelism extractable from the task graph. This affects the task/ task graph latency of the schedule
producible from a given binding, but it doesn’t reduce the size of the search space. This is due to not
using the anticipated schedule latency for a given binding to trim the search space during exploration in
exhaustive search. As a result, the search space for a given instance type is then defined solely by the
number of available bindings for the instance type. The number of bindings is then
processor_count^node_count. 

#### Disadvantages
While this approach is guaranteed to find the optimal schedule it becomes infeasible if instance size grows passed and more than 8 nodes in the graph. 

#### Benchmark Results

Below is a snippet of the exhaustive search algorithm python code:
``` Python
 def run_instance_exhaustive(graph_instance, processor_count, node_count):
   grouped_top_sort = list(topological_sort_grouped(graph_instance))
   min_max_sched_latency = math.inf
   opt_sched = None
   bindings_explored = 0
   for selected_binding in enumerate_bindings(processors, nodes):
     sched = schedule(selected_binding, grouped_top_sort)
     max_latency_sched = max_latency(sched)
     if min_max_sched_latency > max_latency_sched:
       min_max_sched_latency = max_latency_sched
       opt_sched = sched
     bindings_explored += 1
 return (min_max_sched_latency, opt_sched), bindings_explored
```

### Greedy 

The previous exhaustive algorithm was modified to prevent exploring the entire
processor_count^vertex_count space and instead a greedy approach to task processor binding was
implemented. The critical parts of the algorithm’s python code are given below:

``` Python
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
 
def run_instance_naive_greedy(graph_instance, processor_count, node_count):
 grouped_top_sort = list(topological_sort_grouped(graph_instance))
 selected_binding = distributed_load_binding(grouped_top_sort, processor_count)
 greedy_sched = schedule(selected_binding, grouped_top_sort)
 max_latency_sched = makespan(greedy_sched)
 return (max_latency_sched, greedy_sched)
```

The algorithm first begins by running a grouped topological sort on the tasks in the task graph. The steps
for the grouped topological sort are as follows:
1) Scan the task graphs for nodes with no predecessors and groups them together
2) Assume grouped tasks have been executed and eliminate them from the graph
3) Repeat 1 and 2 until the graph is empty
4) Return grouped tasks

After performing the grouped topological sort, the algorithm then runs a load distribution function that
takes each group of tasks and distributes them across the available processors. If there aren’t enough
processors for the group of tasks it reuses processors until the binding is complete. After performing the
binding, the scheduling function runs and schedules tasks as a consequence of the chosen binding. So, for
a given group of n tasks if those tasks are bound to the same processor, they are scheduled sequentially. If
a group of m tasks are scheduled on different processors they are scheduled concurrently. Within a single
group of tasks if there are less processors than tasks and there is more than one processor available then
the schedule is going to contain a mix of sequential and concurrent execution of tasks. Task dependencies
are already handled by the grouping of grouped topological sort so there are no dependency implications
as a result of using this load balancing based binding approach.
The below figures show how both the exhaustive and the naïve greedy approaches perform in our
benchmark with a timeout of 60 seconds. Random directed graphs using the layer-by-layer technique
were generated within the timeout period and scheduled based on both approaches. The number of nodes
within them varied from 4 to 80. The number of processors available to schedule was varied from 4 to 8.
From Fig. 1 it can be shown that the exhaustive search algorithm failed to solve any instances of 8 nodes
or more regardless of the number of processors available. This is understandable given the fact that the
exhaustive approach’s complexity is processor_count^number_of_nodes because the search space of
available bindings is massive. Fig. 2 shows the performance of the above naïve greedy algorithm.
Regardless of instance size naïve greedy algorithm managed to schedule all instances; however, the
algorithm is provably non-optimal based on the inherent limitation of not looking ahead within the task
graph. An example that can be used to illustrate this limitation will be given in the following section. An
interesting thing to note from Fig.2 is the fact that the number of instances solved is unaffected by the
number of available processors because the algorithm distributes available tasks on the available
processors without backtracking to find better bindings. The overall complexity of the above algorithm is
O(V+E). 

#### Disadvantages: Naïve Greedy Non-Optimal Instance
Fig.3 is an example of an instance that isn’t scheduled optimally using our greedy naïve algorithm. Tasks
are grouped based on topological sort, bound to different processors and then scheduled as a consequence
of that binding. The grouping based on topological sort is given as follows:

{{0, 1, 2, 3, 4, 5}, {6, 7}, {8, 9, 10, 11}}
used in 
With the following binding (assuming 4 processors)

{{0, 1, 2, 3, 0, 1}, {0, 1}, {0, 1, 2, 3}}

This results in the following schedule (Task, Start Time):

(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 2), (7, 2), (8, 3), (9, 3), (10, 3), (11, 3)

Fig.3 illustrates the above scheduling and binding overlayed on the instance diagram. Because the
algorithm performs the precedence-based grouping once at the beginning the algorithm doesn’t look for
tasks to execute beyond the group of tasks it’s actively scheduling at any given point. This ignores some
optimizations like executing tasks 4, 5, 6, and 7 concurrently as a result of executing tasks 0, 1, 2, 3 which
are task 6 and 7’s predecessors. If the algorithm is allowed to look ahead to expand its current group
based on what has already been scheduled then the algorithm may be able to find a schedule with a lower
makespan. An illustration of this greedy lookahead is given in Fig. 4

#### Benchmark Results

## References
[1] D. Cordeiro, G. Mounié, S. Perarnau, D. Trystram, J.-M. Vincent, and F. Wagner, “Random graph
generation for scheduling simulations,” Proceedings of the 3rd International ICST Conference on
Simulation Tools and Techniques, 2010.
