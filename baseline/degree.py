import time
import networkx as nx
from copy import deepcopy
import influencediffusion as im
def degree_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes, total_communities, search_tendency):
    """
    基于出度的节点选择算法。
    :param graph: 有向图 (networkx DiGraph)
    :param budget: 需要选择的种子节点数量
    :return: 所选择的种子节点和对应的扩散结果
    """
    start_time = time.time()
    final_spread = []
    all_solutions = []

    # 获取网络中所有节点的度数，并按照度数从大到小对节点进行排序
    deg = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    seed_set = []

    # 选择 budget 个种子节点
    for _ in range(budget):
        seed_set.append(deg[0][0])  # 将出度最大的节点添加到 seed_set
        deg = deg[1:]  # 从 deg 列表中移除已经添加的节点

        # 使用当前的种子节点集 seed_set 进行一次扩散模拟，并记录结果
    # final_spread=im.evaluate_single_objectives(graph,list(seed_set), node_preferences, num_information,
    #         node_to_community, total_nodes, total_communities)
    final_spread = im.evaluate_objectives(graph, list(seed_set), node_preferences, num_information, node_to_community, total_nodes, total_communities)

    # 计算总运行时间
    elapsed_time = time.time() - start_time

    # 保存最终的解决方案
    all_solutions.append((deepcopy(seed_set), deepcopy(final_spread), deepcopy(elapsed_time)))
    return all_solutions
