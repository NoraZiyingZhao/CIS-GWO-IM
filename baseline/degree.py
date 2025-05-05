import time
import networkx as nx
from copy import deepcopy
from Evaluator import *

def degree_seed_selection(graph, budget):
    """
    基于出度的节点选择算法。
    :param graph: 有向图 (networkx DiGraph)
    :param budget: 需要选择的种子节点数量
    :return: 所选择的种子节点和对应的扩散结果
    """
    start_time = time.time()
    all_solutions = []

    # 获取网络中所有节点的度数，并按照度数从大到小对节点进行排序
    deg = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    seed_set = set()

    # 选择 budget 个种子节点
    for _ in range(budget):
        seed_set.add(deg[0][0])  # 加入最大度节点
        deg = deg[1:]            # 从候选中移除

    evaluator = Evaluator(graph)
    final_spread = evaluator.evaluate(seed_set)

    # 计算总运行时间
    elapsed_time = time.time() - start_time

    # 保存最终的解决方案
    all_solutions.append((deepcopy(seed_set), deepcopy(final_spread), deepcopy(elapsed_time)))
    return all_solutions
