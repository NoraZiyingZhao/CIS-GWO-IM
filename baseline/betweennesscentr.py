import networkx as nx
import time
from copy import deepcopy
from Evaluator import *

def betweenness_seed_selection(graph, budget, node_to_comm, total_communities):
    """
    基于 Betweenness Centrality 的种子节点选择算法。
    """
    start_time = time.time()
    all_solutions = []

    # 1. 计算 Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(graph)

    # 2. 排序并选出前 budget 个节点
    sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    seed_set = {node for node, _ in sorted_nodes[:budget]}

    # 3. 调用新的 Evaluator 评估
    evaluator = Evaluator(graph, node_to_comm, total_communities)
    fitness_values = evaluator.evaluate(seed_set)

    # 4. 记录运行时间
    elapsed_time = time.time() - start_time

    # 5. 存储结果
    all_solutions.append((deepcopy(seed_set), deepcopy(fitness_values), deepcopy(elapsed_time)))
    return all_solutions
