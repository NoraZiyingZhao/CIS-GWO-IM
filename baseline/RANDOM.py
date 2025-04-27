import numpy as np
import time
import networkx as nx
import influencediffusion as im
from copy import deepcopy

# result: max spread 0.09,<0.1
def random_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes, total_communities, search_tendency):
    startTime = time.time()
    spread_results = []
    all_solutions = []
    node_list=np.array(list(graph.nodes()))
    seed_set = list(np.random.choice(node_list, budget, replace=False))
    spread_results = im.evaluate_objectives(graph, list(seed_set), node_preferences, num_information, node_to_community, total_nodes, total_communities)
    elapsed_time = time.time() - startTime
    # self.write_results(spread_results)
    all_solutions.append((deepcopy(seed_set),deepcopy(spread_results),deepcopy(elapsed_time)))
    return all_solutions