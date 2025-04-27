import networkx as nx
import numpy as np
import time
import snap
import influencediffusion as im
from copy import deepcopy


def closeness_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes,
                             total_communities):
    """
    基于 Closeness Centrality 的种子节点选择算法。

    参数：
    - graph: 输入图对象（networkx 格式）。
    - budget: 预算，即种子节点的数量。
    - node_preferences: 节点偏好，用于 evaluate_objectives 函数。
    - num_information: 信息数量，用于 evaluate_objectives 函数。
    - node_to_community: 节点到社区的映射。
    - total_nodes: 图中节点总数。
    - total_communities: 图中社区总数。
    - search_tendency: 搜索倾向，用于 evaluate_objectives 函数。

    返回：
    - all_solutions: 包含 (种子节点集合, fitness 值, 运行时间) 的列表。
    """
    startTime = time.time()
    spread_results = []
    all_solutions = []

    # 计算节点的 Closeness Centrality
    closeness_centrality = nx.closeness_centrality(graph)

    # 排序节点，根据 Closeness Centrality 值降序排序
    sorted_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

    # 选择前 budget 个节点作为种子节点
    seed_set = [node for node, centrality in sorted_nodes[:budget]]

    # 计算传播效果（fitness 值）
    spread_results = im.evaluate_objectives(graph, list(seed_set), node_preferences, num_information, node_to_community,
                                            total_nodes, total_communities)

    # 计算运行时间
    elapsed_time = time.time() - startTime

    # 将结果存储到 all_solutions
    all_solutions.append((deepcopy(seed_set), deepcopy(spread_results), deepcopy(elapsed_time)))

    return all_solutions

# import snap
# import re
#
# def Closeness(d, e):
#     f = open(d)
#     s = f.read()
#     s1 = re.split('\n', s)
#     G1 = snap.PUNGraph.New()
#
#     a = re.split(' ', s1[0])
#
#     for i in range(0, int(a[0])):
# 	   G1.AddNode(i)
#
#     for i in range(1, int(a[1]) + 1):
# 	   b = re.split(' ', s1[i])
# 	   G1.AddEdge(int(b[0]), int(b[1]))
#
#     CloseCentr = dict()
#
#     for NI in G1.Nodes():
# 	   CloseCentr[NI.GetId()] = snap.GetClosenessCentr(G1, NI.GetId())
# 	   # print "node: %d centrality: %f" % (NI.GetId(), CloseCentr)
#
#     EdgePara = dict()
#
#     for i in range(1, int(a[1]) +1):
# 	   c = re.split(' ', s1[i])
# 	   if CloseCentr[int(c[0])] == 0 and CloseCentr[int(c[1])] == 0:
# 		  EdgePara[(int(c[0]), int(c[1]))] = 0
# 		  EdgePara[(int(c[1]), int(c[0]))] = 0
# 	   else:
# 		  EdgePara[(int(c[0]), int(c[1]))] = e * CloseCentr[int(c[0])] / (CloseCentr[int(c[0])] + CloseCentr[int(c[1])])
# 		  EdgePara[(int(c[1]), int(c[0]))] = e * CloseCentr[int(c[1])] / (CloseCentr[int(c[0])] + CloseCentr[int(c[1])])
#
#     return EdgePara