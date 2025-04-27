import time
import networkx as nx
from copy import deepcopy
import influencediffusion as im

def eigenvector_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes, total_communities):
    """
    基于 Eigenvector Centrality 的种子节点选择算法。
    """
    startTime = time.time()
    spread_results = []
    all_solutions = []

    # 计算 Eigenvector Centrality
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)

    # 排序节点，根据 Eigenvector Centrality 值降序排序
    sorted_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)

    # 选择前 budget 个节点作为种子节点
    seed_set = [node for node, centrality in sorted_nodes[:budget]]

    # 计算传播效果（fitness 值）
    spread_results = im.evaluate_objectives(graph, list(seed_set), node_preferences, num_information, node_to_community, total_nodes, total_communities)

    # 计算运行时间
    elapsed_time = time.time() - startTime

    # 将结果存储到 all_solutions
    all_solutions.append((deepcopy(seed_set), deepcopy(spread_results), deepcopy(elapsed_time)))

    return all_solutions
