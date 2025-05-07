import time
import networkx as nx
from copy import deepcopy
from Evaluator import *

def eigenvector_seed_selection(graph, budget, node_to_comm, total_communities):
    """
    基于 Eigenvector Centrality 的种子节点选择算法。
    """
    start_time = time.time()
    all_solutions = []

    # 1. 计算 Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)

    # 2. 排序后取前 budget 个节点
    sorted_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    seed_set = {node for node, _ in sorted_nodes[:budget]}

    # 3. 调用 Evaluator 进行评估
    evaluator = Evaluator(graph, node_to_comm, total_communities)
    fitness_values = evaluator.evaluate(seed_set)

    # 4. 记录运行时间
    elapsed_time = time.time() - start_time

    # 5. 返回结构统一的结果
    all_solutions.append((deepcopy(seed_set), deepcopy(fitness_values), deepcopy(elapsed_time)))
    return all_solutions
