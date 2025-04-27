#reference: a multiobjective discrete bat algorithm for community detection in dynamic networks
# include mutation strategy and turbulence operation
# the only difference is the objective problem

import numpy as np
import random
import influencediffusion as im  # 假设是你用于评估目标函数的模块
import paretosolution as ps  # 假设是你的 Pareto 档案管理模块
import HV as hv
import time
from copy import deepcopy

# 使用 sigmoid 映射速度为概率，并生成新解
def sigmoid_mapping(velocity):
    return 1 / (1 + np.exp(-velocity))

# 定义 MODBA 算法
def modba(graph, num_nodes, population_size, budget, node_preferences_dict,
          num_information, node_to_community, total_nodes, total_communities, max_iter):
    """
    基于原始 BA 的多目标离散蝙蝠算法 (MODBA)
    """
    # 参数初始化
    Q_min, Q_max = 0.5, 1.5  # 频率范围
    A_init, A_min = 1.0, 0.1  # 响度初始值和最小值
    r_init, r_max = 0.1, 0.9  # 脉冲发射率初始值和最大值
    alpha, gamma = 0.9, 0.9  # 动态调整系数
    archive_size = 100

    # 初始化种群
    all_solutions = []
    n_bats = population_size
    Q = np.zeros(n_bats)  # 频率
    v = np.zeros((n_bats, total_nodes))  # 速度
    A = np.full(n_bats, A_init)  # 响度
    r = np.full(n_bats, r_init)  # 脉冲发射率

    # 随机生成初始解（节点集合）
    bats = [random.sample(range(total_nodes), budget) for _ in range(n_bats)]

    # Pareto 档案初始化
    archive = []
    reference_point = [0, 0, 0]  # 3个目标，参考点用于计算超体积
    final_HV = []
    each_group_time = []

    # 评估初始解并更新 Pareto 档案
    for bat in bats:
        fitness = im.evaluate_objectives(graph, bat, node_preferences_dict, num_information, node_to_community,
                                         total_nodes, total_communities)
        archive = ps.update_elite_archive(archive, [(bat, fitness)], archive_size)
    print("initialization finished")

    # 主循环
    for t in range(max_iter):
        print(f"iteration: {t}")
        start_time = time.time()
        new_solutions = []  # Temporary storage for new solutions

        for i in range(n_bats):
            # 更新频率
            Q[i] = np.random.uniform(Q_min, Q_max)

            # 更新速度（XOR 操作模拟离散空间中的速度变化）
            v[i] += (np.array([1 if node in bats[i] else 0 for node in range(total_nodes)]) ^
                     np.array([1 if node in archive[0][0] else 0 for node in range(total_nodes)])) * Q[i]

            probabilities = sigmoid_mapping(v[i])
            new_bat = [node for node, prob in enumerate(probabilities) if random.random() < prob]

            # 保持预算约束
            if len(new_bat) > budget:
                new_bat = random.sample(new_bat, budget)

            # 局部搜索（基于脉冲发射率）
            if random.random() > r[i]:
                new_bat = mutation_operator(new_bat, graph)
            else:
                new_bat = turbulence_operator(new_bat, graph)

            # 评估新解
            new_fitness = im.evaluate_objectives(graph, new_bat, node_preferences_dict, num_information,
                                                 node_to_community, total_nodes, total_communities)

            # 根据响度和目标函数更新解
            if random.random() < A[i]:
                bats[i] = new_bat
                archive = ps.update_elite_archive(archive, [(new_bat, new_fitness)], archive_size)

            # 动态调整响度和脉冲发射率
            A[i] = max(A[i] * alpha, A_min)
            r[i] = min(r[i] * (1 - np.exp(-gamma * t)), r_max)

            # Store the new solution
            new_solutions.append((bats[i], new_fitness))

            # Update Pareto archive
        archive = ps.update_elite_archive(archive, new_solutions, archive_size)
        # 计算超体积或其他性能指标
        final_hv_value = hv.calculate_hypervolume(archive, reference_point)
        final_HV.append(final_hv_value)

        # 记录迭代时间
        end_time = time.time()
        iteration_time = end_time - start_time
        each_group_time.append(iteration_time)

    all_solutions.append((deepcopy(archive), deepcopy(each_group_time)))
    return all_solutions, final_HV

# 辅助函数

def mutation_operator(position, graph):
    """
    变异操作：随机替换节点
    """
    for i in range(len(position)):
        if random.random() < 0.2:  # 变异概率
            neighbors = list(graph.neighbors(position[i]))
            if neighbors:
                position[i] = random.choice(neighbors)
    return position

def turbulence_operator(position, graph):
    """
    扰动操作：随机调整节点的分配
    """
    for i in range(len(position)):
        if random.random() < 0.1:  # 扰动概率
            neighbors = list(graph.neighbors(position[i]))
            if neighbors:
                position[i] = random.choice(neighbors)
    return position
