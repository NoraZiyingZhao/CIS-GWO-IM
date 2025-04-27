import numpy as np
import time
from copy import deepcopy
import influencediffusion as im

def CELF_seed_selection(graph, budget, node_preferences, num_information, node_to_community, total_nodes, total_communities, search_tendency):
    fitness_calculate = []
    startTime = time.time()
    g = graph
    all_solutions = []

    # 1. 计算每个节点的边际增益 (单独作为种子)
    marg_gain = [
        im.evaluate_objectives(
            g, [node], node_preferences, num_information,
            node_to_community, total_nodes, total_communities
        )[0] for node in list(g.nodes())
    ]
    # marg_gain 是一个列表，包含多个目标值
    # 我们只基于第一个目标值进行排序
    Q = sorted(zip(list(g.nodes()), marg_gain), key=lambda x: x[1], reverse=True)

    # 3. 选择第一个节点，移除候选节点
    S = [Q[0][0]]
    spread = Q[0][1] # 这里 spread 也是第一个适应度值
    # SPREADLIST = [spread]
    Q = Q[1:]
    # LOOKUPS = [len(g.nodes())]
    # timelapse = [time.time() - startTime]

    # 4. 选择剩余 k-1 个节点
    for _ in range(budget - 1):
        check, node_lookup = False, 0

        while not check:
            node_lookup += 1
            current = Q[0][0]

            # 使用 evaluate_single_objectives 评估 S + [current]
            group_fitness_calculate = im.evaluate_objectives(
                g, S + [current], node_preferences, num_information,
                node_to_community, total_nodes, total_communities
            )

            # new_spread 基于第一个适应度值来计算边际增益
            new_spread = group_fitness_calculate[0] - spread

            # 更新 Q 的当前节点
            Q[0] = (current, new_spread)

            # 重新排序 Q，仍然只根据第一个目标进行排序
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            # Q.sort(key=lambda x: x[1][0], reverse=True)

            # 如果排序后依然是当前节点在顶部，则确认选择
            check = (Q[0][0] == current)

        # 选择节点并更新
        # fitness_calculate.append(group_fitness_calculate)
        spread += Q[0][1]  # spread 只累加第一个适应度值的增益
        S.append(Q[0][0])
        # SPREADLIST.append(spread)
        # LOOKUPS.append(node_lookup)
        # timelapse.append(time.time() - startTime)
        # 移除已选择的节点
        Q = Q[1:]
        final_fitness_calculate = group_fitness_calculate
        timelapse=time.time() - startTime
    # 保存最终的解决方案
    all_solutions.append((deepcopy(S), deepcopy(final_fitness_calculate), deepcopy(timelapse)))
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
