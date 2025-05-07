import numpy as np
import time
from copy import deepcopy
from Evaluator import *

def CELF_seed_selection(graph, budget, node_to_comm, total_communities):
    start_time = time.time()
    all_solutions = []
    evaluator = Evaluator(graph, node_to_comm, total_communities)

    # 1. 计算每个节点的边际增益 (单独作为种子)
    marg_gain = []
    for node in graph.nodes():
        cost = evaluator.evaluate([node])  # [spread, fairness]
        marg_gain.append(cost[0])  # 只看 spread 排序

    # marg_gain 是一个列表，包含多个目标值
    # 我们只基于第一个目标值进行排序
    Q = sorted(zip(list(graph.nodes()), marg_gain), key=lambda x: x[1], reverse=True)

    # 2. 初始化第一个节点
    S = {Q[0][0]}
    spread = Q[0][1]
    Q = Q[1:]
    final_cost = evaluator.evaluate(list(S))  # ✅ 初始化，防止 budget=1 出错
    elapsed = time.time() - start_time  # ✅ 初始时间

    # 3. 迭代选择剩余 budget-1 个节点
    for _ in range(budget - 1):
        check = False
        while not check:
            current = Q[0][0]
            candidate_set = S | {current}
            cost = evaluator.evaluate(list(candidate_set))
            new_spread = cost[0] - spread

            Q[0] = (current, new_spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)

        # 更新 S 与 spread
        spread += Q[0][1]
        S.add(Q[0][0])
        Q = Q[1:]
        final_cost = evaluator.evaluate(list(S))
        elapsed = time.time() - start_time

    all_solutions.append((deepcopy(S), deepcopy(final_cost), deepcopy(elapsed)))
    return all_solutions



#
# def diffuse(network, seed_set, episodes=10, time_steps=100):
#     spread = []
#
#     for i in range(episodes):
#         new_active, already_active = seed_set[:], seed_set[:]
#         while new_active:
#             temp = network.loc[network['source'].isin(new_active)]
#             targets = temp['target'].tolist()
#             np.random.seed(i)
#             success = np.random.uniform(0, 1, len(targets)) < temp['weight']
#             new_ones = np.extract(success, targets)
#             new_active = list(set(new_ones) - set(already_active))
#             already_active += new_active
#             print(already_active)
#         spread.append(len(already_active))
#     return np.mean(spread)
#
#
# def seed_selection(network, k):
#     marg_gain = [(user, diffuse(network.real_network, [user])) for user in network.nodes]
#     Q = sorted(marg_gain, key=lambda x: x[1], reverse=True)
#     seed_set, spread_results, s = [Q[0][0]], [Q[0][1]], Q[0][1]
#     for _ in range(k - 1):
#         Q = Q[1:]
#         check = False
#         while not check:
#             current = Q[0][0]
#             Q[0] = (current, diffuse(network.real_network, seed_set + [current]) - s)
#             Q = sorted(Q, key=lambda x: x[1], reverse=True)
#             check = (Q[0][0] == current)
#         s += Q[0][1]
#         seed_set.append(Q[0][0])
#         spread_results.append(s)
#     # self.write_results(spread_results)
#     return seed_set
