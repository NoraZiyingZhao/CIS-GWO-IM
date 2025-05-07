import networkx as nx
import numpy as np
import time
from Evaluator import *
from copy import deepcopy

def closeness_seed_selection(graph, budget, node_to_comm, total_communities):
    """
    基于 Closeness Centrality 的种子节点选择算法。
    """
    start_time = time.time()
    all_solutions = []

    # 1. 计算节点的 Closeness Centrality
    closeness_centrality = nx.closeness_centrality(graph)

    # 2. 按照 centrality 从大到小排序
    sorted_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

    # 3. 选出前 budget 个节点作为种子
    seed_set = {node for node, _ in sorted_nodes[:budget]}

    # 4. 执行多目标评估
    evaluator = Evaluator(graph, node_to_comm, total_communities)
    fitness_values = evaluator.evaluate(seed_set)

    # 5. 记录运行时间
    elapsed_time = time.time() - start_time

    # 6. 保存结果
    all_solutions.append((deepcopy(seed_set), deepcopy(fitness_values), deepcopy(elapsed_time)))
    return all_solutions
