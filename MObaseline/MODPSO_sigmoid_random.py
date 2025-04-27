import networkx as nx
import numpy as np
import random
import influencediffusion as im
import paretosolution as ps
import HV as hv
import time
from copy import deepcopy


class modpso:
    def __init__(self, graph, budget, num_particles, archiveSize):
        """
        多目标离散粒子群优化
        Args:
            graph: 输入的网络图。
            budget: 每个解（种子集合）的大小。
            num_particles: 粒子数量。
            max_iter: 最大迭代次数。
        """
        self.graph = graph  # 输入图
        self.budget = budget  # 每个解（种子集合）的大小
        self.num_particles = num_particles  # 粒子数量
        self.particles = []  # 粒子集合（每个粒子为种子集合）
        self.velocities = []  # 粒子的速度（表示节点替换的概率向量）
        self.pbest = []  # 个体历史最优解
        self.pbest_fitness = []  # 个体历史最优适应度
        self.archive = []  # Pareto 存档
        self.archive_size = archiveSize  # 存档最大容量
        self.start_time=[]
        self.end_time=[]
        self.Time = []

    def initialize_particles(self, node_preferences, num_information, node_to_community, total_nodes, total_communities):
        """初始化粒子位置和速度"""
        nodes = list(self.graph.nodes)
        for _ in range(self.num_particles):
            # 初始化粒子位置为随机种子集合
            particle = random.sample(nodes, self.budget)
            self.particles.append(particle)

            # 初始化速度为全零
            velocity = np.zeros(len(nodes))
            self.velocities.append(velocity)

            # 初始化个体最优解
            self.pbest.append(particle)
            fitness = im.evaluate_objectives(self.graph, particle, node_preferences, num_information, node_to_community, total_nodes, total_communities)
            self.pbest_fitness.append(fitness)

            # 初始化 Pareto 存档
            self.archive = ps.update_elite_archive(self.archive, zip(self.particles, self.pbest_fitness), self.archive_size)

    def update_velocity(self, particle_idx, global_best):
        """更新粒子的速度"""
        w = 1  # 惯性权重0.7
        c1, c2 = 1, 1 # 加速度因子
        r1, r2 = np.random.rand(), np.random.rand()  # 随机数

        current_position = self.particles[particle_idx]
        personal_best = self.pbest[particle_idx]

        # 将解转换为二进制向量
        current_vector = np.array([1 if node in current_position else 0 for node in self.graph.nodes])
        personal_best_vector = np.array([1 if node in personal_best else 0 for node in self.graph.nodes])
        global_best_vector = np.array([1 if node in global_best else 0 for node in self.graph.nodes])

        # 更新速度
        new_velocity = (
            w * self.velocities[particle_idx]
            + c1 * r1 * (personal_best_vector - current_vector)
            + c2 * r2 * (global_best_vector - current_vector)
        )
        self.velocities[particle_idx] = np.clip(new_velocity, -4, 4)

    def update_position(self, particle_idx):
        """更新粒子的位置（种子集合）"""
        velocity = self.velocities[particle_idx]
        probabilities = 1 / (1 + np.exp(-velocity))  # 使用 Sigmoid 函数生成选择概率
        nodes = list(self.graph.nodes)

        # 根据速度的概率更新节点集合
        new_position = [node for node, prob in zip(nodes, probabilities) if random.random() < prob]

        # 保证解的大小满足预算约束
        if len(new_position) > self.budget:
            new_position = random.sample(new_position, self.budget)
        elif len(new_position) < self.budget:
            additional_nodes = random.sample(
                list(set(nodes) - set(new_position)), self.budget - len(new_position)
            )
            new_position.extend(additional_nodes)

        self.particles[particle_idx] = new_position

    def optimize(self, node_preferences, num_information, node_to_community, total_nodes, total_communities, maxIt):
        """主优化过程"""
        all_solutions=[]
        final_HV=[]
        reference_point = [0, 0, 0]
        print("optimization start")
        self.initialize_particles(node_preferences, num_information, node_to_community, total_nodes, total_communities)

        for iteration in range(maxIt):
            self.start_time = time.time()
            global_best = random.choice([archive_item[0] for archive_item in self.archive])  # 从 Pareto 存档中随机选择全局最优解

            for i in range(self.num_particles):
                # 更新速度和位置
                self.update_velocity(i, global_best)
                self.update_position(i)

                # 评估新位置的适应度
                new_fitness = im.evaluate_objectives(self.graph, self.particles[i], node_preferences, num_information, node_to_community, total_nodes, total_communities)

                # 更新个体最优解
                if ps.dominates(new_fitness, self.pbest_fitness[i]):
                    self.pbest[i] = self.particles[i]
                    self.pbest_fitness[i] = new_fitness

            # 更新 Pareto 存档
            self.archive = ps.update_elite_archive(self.archive, zip(self.particles, self.pbest_fitness), self.archive_size)

            print(f"Iteration {iteration + 1}: Archive size = {len(self.archive)}")

            # Calculate hypervolume for this iteration
            final_hv_value = hv.calculate_hypervolume(self.archive, reference_point)
            final_HV.append(final_hv_value)

            # Record iteration time
            self.end_time = time.time()
            self.Time.append(self.end_time - self.start_time)

        all_solutions.append((deepcopy(self.archive), deepcopy(self.Time)))
        return all_solutions, final_HV
