import numpy as np
import time
import networkx as nx
import Save_Visualize
import influencediffusion as im
from copy import deepcopy

def pagerank_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes, total_communities, search_tendency):
    startTime = time.time()  # 记录开始时间
    all_solutions = []

    # Step 1: Calculate PageRank using networkx
    pagerank_dict = nx.pagerank(graph)

    # Step 2: Sort nodes based on their PageRank scores
    sorted_nodes = sorted(pagerank_dict.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Select top-k nodes based on PageRank as seeds
    top_k_nodes = [node for node, rank in sorted_nodes[:budget]]

    # Step 4: Calculate the fitness of the selected seeds
    # evaluate_objectives returns multiple objectives in a tuple (obj1, obj2, obj3, ...)
    fitness_values = im.evaluate_objectives(
        graph, list(top_k_nodes), node_preferences, num_information,
        node_to_community, total_nodes, total_communities
    )

    # Step 5: Record the time taken for the seed selection process
    elapsed_time = time.time() - startTime

    # Step 6: Save the results
    all_solutions.append((deepcopy(top_k_nodes), deepcopy(fitness_values), deepcopy(elapsed_time)))

    return all_solutions
