import numpy as np
import time
import networkx as nx
from Evaluator import *
from copy import deepcopy

def random_seed_selection(graph, budget):
    start_time = time.time()
    all_solutions = []
    evaluator = Evaluator(graph)

    # 随机选 budget 个节点，确保不重复
    node_list = list(graph.nodes())
    seed_set = set(np.random.choice(node_list, budget, replace=False))

    # 评估传播和公平性
    fitness_values = evaluator.evaluate(seed_set)

    # 记录运行时间
    elapsed_time = time.time() - start_time

    # 保存结果
    all_solutions.append((deepcopy(seed_set), deepcopy(fitness_values), deepcopy(elapsed_time)))
    return all_solutions
