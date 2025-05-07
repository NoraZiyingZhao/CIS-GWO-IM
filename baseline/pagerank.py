import numpy as np
import time
import networkx as nx
from Evaluator import *
from copy import deepcopy

def pagerank_seed_selection(graph, budget, node_to_comm, total_communities):
    start_time = time.time()
    all_solutions = []

    # Step 1: Compute PageRank
    pagerank_dict = nx.pagerank(graph)

    # Step 2: Sort and select top-k nodes
    sorted_nodes = sorted(pagerank_dict.items(), key=lambda x: x[1], reverse=True)
    seed_set = set(node for node, _ in sorted_nodes[:budget])

    # Step 3: Evaluate fitness
    evaluator = Evaluator(graph, node_to_comm, total_communities)
    fitness_values = evaluator.evaluate(seed_set)

    # Step 4: Record time
    elapsed_time = time.time() - start_time

    # Step 5: Store result
    all_solutions.append((deepcopy(seed_set), deepcopy(fitness_values), deepcopy(elapsed_time)))
    return all_solutions
