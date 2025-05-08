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

class communityStratifiedFMODGWO:
    """
    FMODGWO: Fairness-aware Multi-Objective Grey Wolf Optimizer
    针对双目标最大化问题（如扩散性与公平性）的多目标灰狼优化算法。
    """

    def __init__(self, graph, structure_metrics, budget, pop_size, archive_size, node_to_comm, total_communities):
        """
                初始化优化器。
                :param graph: 输入图（节点与边）
                :param structure_metrics: 节点结构性指标对象
                :param budget: 种子集合大小上限
                :param pop_size: 种群大小（狼的数量）
                :param archive_size: 存档大小（非支配解集上限）
                """
        self.graph = graph
        self.StructureMetrics = structure_metrics
        self.budget = budget
        self.pop_size = pop_size
        self.archive_size = archive_size

        self.node_to_comm = node_to_comm
        self.total_communities = total_communities

        self.evaluator = Evaluator(graph, node_to_comm, total_communities)  # ✅ 准备给 evaluator 使用
        self.perturb = PerturbationHandler(self.StructureMetrics)
        self.archive_mgr = ArchiveManager(self.archive_size)
        self.leader_mgr = LeaderManager()
        self.population = []

    def initialize_population(self):
        """
        初始化种群：
        - 以随机选择为主
        - 若发现社区覆盖数量过少，则从其他社区替换部分节点，增强多样性
        """
        nodes = list(self.graph.nodes())

        # 构建 community -> node 列表映射
        community_to_nodes = {}
        for node, comm in self.node_to_comm.items():
            community_to_nodes.setdefault(comm, []).append(node)

        all_communities = list(community_to_nodes.keys())
        self.population = []

        for i in range(self.pop_size):
            wolf = GreyWolf()

            # Step 1: 随机选择 budget 个节点
            selected = set(random.sample(nodes, self.budget))

            # Step 2: 检查社区覆盖数量
            covered_comms = set(self.node_to_comm[n] for n in selected)
            min_required_communities = max(2, self.budget // 2)

            if len(covered_comms) < min_required_communities:
                # Step 3: 替换部分节点，增加社区覆盖
                # 找出被选中过多的社区
                comm_counts = {}
                for node in selected:
                    comm = self.node_to_comm[node]
                    comm_counts[comm] = comm_counts.get(comm, 0) + 1

                # 找出最多的社区（出现次数最多）
                over_comm = max(comm_counts.items(), key=lambda x: x[1])[0]
                node_to_replace = next(n for n in selected if self.node_to_comm[n] == over_comm)

                # 在未覆盖的社区中随机选择一个节点替换
                uncovered = list(set(all_communities) - covered_comms)
                random.shuffle(uncovered)
                replaced = False

                for comm in uncovered:
                    candidates = community_to_nodes.get(comm, [])
                    random.shuffle(candidates)
                    for cand in candidates:
                        if cand not in selected:
                            selected.remove(node_to_replace)
                            selected.add(cand)
                            replaced = True
                            break
                    if replaced:
                        break  # 替换成功就退出

            # Step 4: 完成种子集合并评估
            wolf.Position = selected
            wolf.Cost = self.evaluator.evaluate(wolf.Position)
            self.population.append(wolf)

            # 可选：打印社区覆盖情况
            covered = set(self.node_to_comm[n] for n in selected)
            print(f"Wolf {i+1} covers {len(covered)} communities.")

    def stratified_sample(self, candidate_nodes, degree_dict, sample_size=200):
        """
        对candidate_nodes按degree分层采样。

        candidate_nodes: list of node ids
        degree_dict: dict of {node: degree}
        sample_size: 总采样数量
        """

        # 分层
        low_degree = [node for node in candidate_nodes if degree_dict[node] <= 10]
        mid_degree = [node for node in candidate_nodes if 10 < degree_dict[node] <= 30]
        high_degree = [node for node in candidate_nodes if degree_dict[node] > 30]

        # 每层采样比例（你可以微调，比如高层多采一点）
        low_num = int(0.4 * sample_size)  # 低度节点占40%
        mid_num = int(0.4 * sample_size)  # 中度节点占40%
        high_num = sample_size - low_num - mid_num  # 剩下给高度节点

        # 采样
        sampled = []
        if low_degree:
            sampled += random.sample(low_degree, min(low_num, len(low_degree)))
        if mid_degree:
            sampled += random.sample(mid_degree, min(mid_num, len(mid_degree)))
        if high_degree:
            sampled += random.sample(high_degree, min(high_num, len(high_degree)))

        return sampled

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
        transition_point = 0.6  # 设定迭代总次数的40%作为算法由“全局探索”阶段向“局部开发”阶段切换的关键节点。

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
                    if candidate_nodes:
                        sample_size = min(200, len(candidate_nodes))
                        sampled_nodes = self.stratified_sample(candidate_nodes, self.StructureMetrics.degree, sample_size)

                        scores = []
                        for node in sampled_nodes:
                            score = self.StructureMetrics.score(node, t, max_iter, transition_point)
                            scores.append((node, score))
                        scores.sort(key=lambda x: x[1], reverse=True)
                        candidate_nodes = [node for node, _ in scores]
                    # # 为所有候选节点打分
                    # scores = []
                    # for node in candidate_nodes:
                    #     score = self.StructureMetrics.score(node, t, max_iter, transition_point)
                    #     scores.append((node, score))
                    # scores.sort(key=lambda x: x[1], reverse=True)
                    # candidate_nodes = [node for node, _ in scores]

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
                    if candidate_nodes:
                        sample_size = min(500, len(candidate_nodes))
                        sampled_nodes = self.stratified_sample(candidate_nodes, self.StructureMetrics.degree, sample_size)

                        scores = []
                        for node in sampled_nodes:
                            score = self.StructureMetrics.score(node, t, max_iter, transition_point)
                            scores.append((node, score))
                        scores.sort(key=lambda x: x[1], reverse=True)
                        candidate_nodes = [node for node, _ in scores]
                    needed = self.budget - len(base)
                    if len(candidate_nodes) <= needed:
                        new_nodes = set(candidate_nodes)
                    else:
                        new_nodes = set(candidate_nodes[:needed])
                    # needed = self.budget - len(base)
                    # if len(candidate_nodes) <= needed:
                    #     new_nodes = set(candidate_nodes)
                    # else:
                    #     # 用score函数对local候选节点打分（小closeness + 大degree优先）
                    #     scores = []
                    #     for node in candidate_nodes:
                    #         score = self.StructureMetrics.score(node, t, max_iter, transition_point)
                    #         scores.append((node, score))
                    #     scores.sort(key=lambda x: x[1], reverse=True)
                    #     candidate_nodes = [node for node, _ in scores]
                    #     new_nodes = set(candidate_nodes[:needed])
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
            archive_costs_history.append([sol.Cost for sol in archive])
            hv = self.archive_mgr.calculate_hypervolume(reference_point=(0, 0))
            hv_values.append(hv)

            end_time = time.time()
            times.append(end_time - start_time)
        print("time=",times)

        return self.archive_mgr.archive, archive_costs_history,  hv_values, times



