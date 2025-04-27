#reference: influence maximization in social networks based on discrete particle swarm optimization.
#in this paper, a local search strategy exists with high computational conplexity.
# So I removed this part and this code is a multiobjective discrete PSO algorithm with the same logic of original PSO.
# also I removed the degree knowledge part in this code

import networkx as nx
import numpy as np
import random
import influencediffusion as im
import paretosolution as ps
import HV as hv
from ArchiveManager import *
import time
from copy import deepcopy


class modpso:
    def __init__(self, graph, budget, num_particles, archive_size):
        """
        多目标离散粒子群优化
        """
        self.graph = graph  # 网络图
        self.budget = budget  # 种子集合大小
        self.num_particles = num_particles  # 粒子数量
        self.archive_size = archive_size  # 存档大小
        self.particles = []  # 粒子位置（种子集合）
        self.velocities = []  # 粒子速度（添加或移除操作）
        self.pbest = []  # 个体最优位置
        self.pbest_fitness = []  # 个体最优适应度
        self.archive = []  # Pareto 存档
        self.gbest = None  # 全局最优位置
        self.gbest_fitness = None  # 全局最优适应度
        self.start_time = []
        self.end_time = []
        self.Time = []

    def initialize_particles(self, node_preferences, num_information, node_to_community, total_nodes,
                             total_communities):

        nodes = list(self.graph.nodes)


        for _ in range(self.num_particles):
            particle = random.sample(nodes, self.budget)

            self.particles.append(particle)

            # 初始化速度为空
            self.velocities.append([])

            # 初始化个体最优解为当前解
            self.pbest.append(particle)
            fitness = im.evaluate_objectives(self.graph, particle, node_preferences, num_information, node_to_community,
                                             total_nodes, total_communities)
            self.pbest_fitness.append(fitness)

            # 更新 Pareto 档案
            self.archive = ps.update_elite_archive(self.archive, [(particle, fitness)], self.archive_size)

    def select_gbest(self):
        """
        从 Pareto 档案中随机选择全局最优解
        """
        if not self.archive:
            return None, None
        # 随机从 Pareto 档案中选择一个解作为 gbest
        gbest = random.choice(self.archive)
        return gbest[0], gbest[1]
    def update_velocity(self, particle_idx, gbest_position):
        """
        更新粒子的速度
        """
        current_position = set(self.particles[particle_idx])
        pbest_position = set(self.pbest[particle_idx])
        # gbest_position = set(self.gbest)

        # 计算需要添加和移除的节点
        add_operations = list((pbest_position | gbest_position) - current_position)
        remove_operations = list(current_position - (pbest_position & gbest_position))

        # 随机选择部分操作作为新的速度
        add_sample = random.sample(add_operations, min(len(add_operations), self.budget // 2))
        remove_sample = random.sample(remove_operations, min(len(remove_operations), self.budget // 2))

        self.velocities[particle_idx] = add_sample + remove_sample

    def update_position(self, particle_idx):
        """
        更新粒子的位置
        """
        current_position = set(self.particles[particle_idx])
        new_position = current_position.copy()

        # 应用速度更新操作
        for node in self.velocities[particle_idx]:
            if node in new_position:
                new_position.remove(node)  # 移除节点
            else:
                new_position.add(node)  # 添加节点

        # 确保种子集合大小满足预算约束
        if len(new_position) > self.budget:
            new_position = set(random.sample(new_position, self.budget))
        elif len(new_position) < self.budget:
            additional_nodes = random.sample(
                list(set(self.graph.nodes) - new_position), self.budget - len(new_position)
            )
            new_position.update(additional_nodes)

        self.particles[particle_idx] = list(new_position)

    def local_search(self, particle, node_preferences, num_information, node_to_community, total_nodes,
                             total_communities):
        """
        局部搜索：按照论文的 Local_Search 算法优化粒子
        """
        particle_fitness = im.evaluate_objectives(self.graph, particle, node_preferences, num_information,
                                                  node_to_community,
                                                  total_nodes, total_communities)
        improved = True
        while improved:
            improved = False
            for idx, node in enumerate(particle):
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in particle:
                        candidate = particle[:]
                        candidate[idx] = neighbor
                        candidate_fitness = im.evaluate_objectives(self.graph, candidate, node_preferences, num_information, node_to_community,
                                             total_nodes, total_communities)
                        if ps.dominates(candidate_fitness, particle_fitness):
                            particle = candidate
                            particle_fitness = candidate_fitness
                            improved = True
                            break
                if improved:
                    break
        return particle, particle_fitness


    def optimize(self, node_preferences, num_information, node_to_community, total_nodes, total_communities, max_iterations):
        """
        优化主循环
        """
        all_solutions = []
        final_HV = []
        reference_point = [0, 0, 0]
        print("optimization start")
        self.initialize_particles(node_preferences, num_information, node_to_community, total_nodes,
                             total_communities)

        for iteration in range(max_iterations):
            new_solutions = []
            self.start_time = time.time()
            gbest_position, _ = self.select_gbest()

            for i in range(self.num_particles):
                # 更新速度和位置
                self.update_velocity(i, set(gbest_position))
                self.update_position(i)

                # 评估新位置的适应度
                fitness = im.evaluate_objectives(self.graph, self.particles[i], node_preferences, num_information,
                                                 node_to_community,
                                                 total_nodes, total_communities)
                # # 局部搜索优化
                # self.particles[i] , fitness = self.local_search(self.particles[i], node_preferences, num_information, node_to_community, total_nodes,
                #              total_communities)

                # 更新个体最优解
                if ps.dominates(fitness, self.pbest_fitness[i]):
                    self.pbest[i] = self.particles[i]
                    self.pbest_fitness[i] = fitness

                # 添加到新解集合
                new_solutions.append((self.particles[i], fitness))

                # 更新 Pareto 档案
            self.archive = ps.update_elite_archive(self.archive, new_solutions, self.archive_size)

            print(f"Iteration {iteration + 1}: Archive Size = {len(self.archive)}")

            # Calculate hypervolume for this iteration
            final_hv_value = hv.calculate_hypervolume(self.archive, reference_point)
            final_HV.append(final_hv_value)

            # Record iteration time
            self.end_time = time.time()
            self.Time.append(self.end_time - self.start_time)

        all_solutions.append((deepcopy(self.archive), deepcopy(self.Time)))
        return all_solutions, final_HV
