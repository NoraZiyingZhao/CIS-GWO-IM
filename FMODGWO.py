# -----------------------------------------------------------
# This class implements the core logic of the proposed FMODGWO algorithm for solving Fairness-aware
# Discrete Multi-Objective Influence Maximization (DMOIM) problems.
#
# Key Features:
# - Each wolf represents a candidate seed set (solution).
# - Initialization:
# - Optimization: guided by leaders (alpha, beta, delta), updated via intersection-base + scored selection.
# - Dynamic node scoring (closeness + degree) with changing weights.
# - Perturbation: encourages structural exploration based on closeness centrality.
# - Archive manager maintains a set of Pareto-optimal solutions.
# - Returns final archive, history of costs, seedsets, and HV approximations for each iteration.
#

import random
from GreyWolf import GreyWolf
from ArchiveManager import *
# from CandidateSelector import *
from Evaluator import *
from PerturbationHandler import *
from collections import defaultdict
from LeaderManager import *
from StructureMetrics import *
import time

class FMODGWO:
    """
    FMODGWO: Fairness-aware Multi-Objective Grey Wolf Optimizer
    针对双目标最大化问题（如扩散性与公平性）的多目标灰狼优化算法。
    """

    def __init__(self, graph, structure_metrics, budget, pop_size, archive_size):
        """
        初始化优化器。
        :param graph: 输入图（节点与边）
        :param structure_metrics: 节点结构性指标对象
        :param budget: 种子集合大小上限
        :param pop_size: 种群大小（狼的数量）
        :param archive_size: 存档大小（非支配解集上限）
        """
        self.graph = graph
        self.metrics = structure_metrics
        self.budget = budget
        self.pop_size = pop_size
        self.archive_size = archive_size

        # 初始化辅助模块
        self.evaluator = Evaluator(graph)               # 评估器（传播/公平性目标）
        # self.selector = CandidateSelector(graph, self.metrics)  # 候选池生成器
        self.perturb = PerturbationHandler(graph)        # 扰动器
        self.archive_mgr = ArchiveManager(self.archive_size)  # 存档管理器
        self.leader_mgr = LeaderManager()                # 领头狼选择器
        self.StructureMetrics = StructureMetrics(graph)

        self.population = []  # 当前狼群

    def initialize_population(self):
        # 初始化种群，每只狼表示一个种子集合（Position），其大小等于预算 budget
        # 获取图中的所有节点
        nodes = list(self.graph.nodes())
        # 按照 degree 从高到低排序
        deg_sorted = sorted(self.metrics.degree.items(), key=lambda x: x[1], reverse=True)
        # 取 top 20% 的节点数量（至少1个）作为结构启发候选集的大小
        top_k = max(1, len(nodes) // 5)  # Top 20% degree nodes
        # 取出 top_k 个 degree 最大的节点作为启发式候选集合
        top_nodes = [node for node, _ in deg_sorted[:top_k]]

        # 初始化种群列表
        self.population = []
        # 遍历种群大小，初始化每一只狼
        for i in range(self.pop_size):
            # 创建新的灰狼个体
            wolf = GreyWolf()
            # 初始化其节点集合为空
            selected = set()

            # 前 50% 的狼完全随机选择 budget 个节点
            if i < int(0.5 * self.pop_size):
                # 50% wolves randomly select from the whole graph
                selected = set(random.sample(nodes, self.budget))  # 从所有节点中随机采样
            else:  # 后 50% 的狼从 top degree 节点中选出部分，其余从剩下的节点中补全
                # 50% wolves include top degree nodes as part of their seed set
                # 计算可选的 top 节点数量（最多为 budget）
                num_top = min(len(top_nodes), self.budget)
                # 从 top_nodes 中随机选出 1~budget 个节点
                top_sample = random.sample(top_nodes, k=random.randint(1, num_top))
                # 剩余需要随机补充的节点数
                remaining_budget = self.budget - len(top_sample)
                # 构建不包含 top_sample 的候选池
                rest_pool = list(set(nodes) - set(top_sample))
                # 从剩余节点中随机选出 remaining_budget 个节点
                random_sample = random.sample(rest_pool, remaining_budget)
                # 合并为完整的 seed set（无重复）
                selected = set(top_sample + random_sample)

                # 设置狼的种子集合为构造好的节点集
            wolf.Position = selected
            # 评估该种子集合的多目标成本（传播、成本、公平性等）
            wolf.Cost = self.evaluator.evaluate(wolf.Position)
            # 将该个体添加到种群中
            self.population.append(wolf)

    def optimize(self, max_iter):
        """
        主优化流程：基于统一过渡点的动态机制，协同更新每只狼的位置。
        :param max_iter: 最大迭代次数
        :return: 最优非支配解集、每代历史记录、HV变化、每代耗时
        """
        self.initialize_population()
        self.archive_mgr.update(self.population)  # 初始化非支配存档

        archive_costs_history = []  # 每代的Pareto解Cost
        hv_values = []  # 每代的HV
        times = []  # 每代的时间
        transition_point = 0.4  # 设定迭代总次数的40%作为算法由“全局探索”阶段向“局部开发”阶段切换的关键节点。

        for t in range(max_iter):
            start_time = time.time()

            self.leader_mgr.ensure_leader_minimum(self.archive_mgr.archive, self.population)

            # 计算search tendency
            search_tendency = "global" if t < transition_point * max_iter else "local"

            for wolf in self.population:
                # archive = self.archive_mgr.archive

                # 动态选择当前的leaders
                alpha, beta, delta = self.leader_mgr.select_leaders_by_region(self.archive_mgr.get_fronts())

                # 构建三头狼和当前狼的共同节点
                base = wolf.Position & alpha.Position & beta.Position & delta.Position

                if search_tendency == "global":
                    # ---- Global Search ----
                    # 从图中所有节点中，排除 base 和 当前狼已有节点
                    all_nodes = set(self.graph.nodes())
                    forbidden_nodes = base | wolf.Position
                    candidate_nodes = list(all_nodes - forbidden_nodes)
                    # 使用 score() 函数为候选节点打分（小closeness + 大degree优先）
                    candidate_nodes = self.StructureMetrics.score(candidate_nodes,t, max_iter, transition_point)
                    # 选择补充节点
                    needed = self.budget - len(base)
                    new_nodes = set(candidate_nodes[:needed])
                else:
                    # ---- Local Search ----
                    leader_nodes = alpha.Position | beta.Position | delta.Position
                    # 1阶邻居
                    first_neighbors = set()
                    for node in leader_nodes:
                        first_neighbors.update(self.graph.neighbors(node))
                    # 2阶邻居
                    second_neighbors = set()
                    for node in first_neighbors:
                        second_neighbors.update(self.graph.neighbors(node))

                    # 综合1阶和2阶邻居
                    candidate_nodes = (leader_nodes | first_neighbors | second_neighbors) - base - wolf.Position
                    candidate_nodes = list(candidate_nodes)
                    needed = self.budget - len(base)
                    if len(candidate_nodes) <= needed:
                        new_nodes = set(candidate_nodes)
                    else:
                        # 用score函数对local候选节点打分（小closeness + 大degree优先）
                        candidate_nodes = sorted(candidate_nodes,
                                                 key=lambda x: self.StructureMetrics.score(x, t, max_iter, transition_point))
                        new_nodes = set(candidate_nodes[:needed])
                # 合成新的位置
                new_position = base | new_nodes

                # 随迭代增加的概率扰动新位置，增强多样性
                # 动态扰动处理
                p_perturb = self.perturb.dynamic_perturbation_prob(t, max_iter, transition_point)
                perturb_ratio = self.perturb.dynamic_perturbation_ratio(t, max_iter, transition_point)

                if random.random() < p_perturb:
                    new_position = self.perturb.perturb(new_position, ratio=perturb_ratio)
                # 更新狼的位置与适应度
                wolf.Position = new_position
                wolf.Cost = self.evaluator.evaluate(new_position)

            # 更新存档
            self.archive_mgr.update(self.population)

            # 记录每代数据
            archive = self.archive_mgr.archive
            archive_costs_history.append([sol[1] for sol in archive])
            hv = self.archive_mgr.calculate_hypervolume(reference_point=(0, 0))
            hv_values.append(hv)

            end_time = time.time()
            times.append(end_time - start_time)

        return self.archive_mgr.archive, archive_costs_history,  hv_values, times

