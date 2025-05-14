import random
from GreyWolf import GreyWolf
from ArchiveManager import *
from Evaluator import *
from PerturbationHandler import *
from LeaderManager import *
from StructureMetrics import *
import time

class communityStratifiedFMODGWO:
    """
    社区感知多目标灰狼优化算法：集成社区结构、静态中心性评分、动态扰动机制（含反馈调节）以解决多目标图优化问题。
    """

    def __init__(self, graph, structure_metrics, budget, pop_size, archive_size, node_to_comm, total_communities):
        self.graph = graph
        self.StructureMetrics = structure_metrics
        self.budget = budget
        self.pop_size = pop_size
        self.archive_size = archive_size

        self.node_to_comm = node_to_comm
        self.total_communities = total_communities

        # 初始化功能模块
        self.evaluator = Evaluator(graph, node_to_comm, total_communities)
        self.perturb = PerturbationHandler(self.StructureMetrics)
        self.archive_mgr = ArchiveManager(self.archive_size)
        self.leader_mgr = LeaderManager()
        self.population = []

    def initialize_population(self):
        """
        初始化种群：
        - 从整个图中随机抽取 budget 个节点形成解；
        - 若发现社区覆盖数量过少，则从“节点数多的社区”中尝试替换一个节点；
        - 替换条件：该社区在当前解中至少出现 2 次（防止被完全替换掉）；
        - 替换目标：未被当前解覆盖的社区中的任意节点。
        """
        nodes = list(self.graph.nodes())

        # 构建 community → node 列表映射
        community_to_nodes = {}
        for node, comm in self.node_to_comm.items():
            community_to_nodes.setdefault(comm, []).append(node)

        all_communities = list(community_to_nodes.keys())
        self.population = []

        for i in range(self.pop_size):
            wolf = GreyWolf()

            # === Step 1: 随机选出 budget 个节点 ===
            selected = set(random.sample(nodes, self.budget))

            # === Step 2: 检查社区覆盖数量是否足够 ===
            covered_comms = set(self.node_to_comm[n] for n in selected)
            min_required_communities = max(2, self.budget // 2)

            if len(covered_comms) < min_required_communities:
                # === 统计当前 selected 中每个社区的出现次数 ===
                comm_counts = {}
                for node in selected:
                    comm = self.node_to_comm[node]
                    comm_counts[comm] = comm_counts.get(comm, 0) + 1

                # === 找出在 selected 中出现 ≥2 次的社区，降序排列 ===
                over_communities = sorted(
                    [comm for comm, count in comm_counts.items() if count >= 2],
                    key=lambda c: comm_counts[c],
                    reverse=True
                )

                # === 尝试从这些社区中替换一个节点，引入未覆盖社区的节点 ===
                replaced = False
                for over_comm in over_communities:
                    replace_candidates = [n for n in selected if self.node_to_comm[n] == over_comm]
                    if not replace_candidates:
                        continue  # 保险判断

                    node_to_replace = random.choice(replace_candidates)

                    # 寻找未覆盖的社区
                    uncovered = list(set(all_communities) - covered_comms)
                    random.shuffle(uncovered)

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
                            break
                    if replaced:
                        break  # 替换完成，退出循环

            # === Step 3: 完成初始化个体评估 ===
            wolf.Position = selected
            wolf.Cost = self.evaluator.evaluate(wolf.Position)
            self.population.append(wolf)

            # ✅ 可选调试输出（可注释）
            covered_now = set(self.node_to_comm[n] for n in selected)
            print(f"Wolf {i+1}: covers {len(covered_now)} communities.")

    def optimize(self, max_iter):
        self.initialize_population()
        self.archive_mgr.update(self.population)

        archive_costs_history = []
        hv_values = []
        times = []
        transition_point = 0.6  # 60% 前为 global，后为 local
        stagnation_counter = 0
        stagnation_threshold = 10 #monitor the variant of HV value
        hv_tolerance = 1e-6

        # === 获取结构评分（使用封装方法） ===
        score_global, score_local = self.StructureMetrics.compute_scores()

        # === 构建社区样本池（用于 global / local） ===
        community_samples_global = {i: [] for i in range(self.total_communities)}
        community_samples_local = {i: [] for i in range(self.total_communities)}

        for node in self.graph.nodes():
            comm = self.node_to_comm[node]
            community_samples_global[comm].append((node, score_global[node]))
            community_samples_local[comm].append((node, score_local[node]))

        max_ratio = 0.2
        for comm_id in community_samples_global:
            community_samples_global[comm_id].sort(key=lambda x: x[1], reverse=True)
            community_samples_local[comm_id].sort(key=lambda x: x[1], reverse=True)
            k = max(5, int(len(community_samples_global[comm_id]) * max_ratio))
            community_samples_global[comm_id] = [n for n, _ in community_samples_global[comm_id][:k]]
            community_samples_local[comm_id] = [n for n, _ in community_samples_local[comm_id][:k]]

        # === Hub 节点提取 ===
        # 在主算法初始化阶段使用 eigenvector 提取 hub 节点
        hub_score = self.StructureMetrics.eigenvector
        hub_nodes = sorted(hub_score, key=hub_score.get, reverse=True)[:int(0.05 * len(hub_score))]
        self.hub_nodes = hub_nodes

        for t in range(max_iter):
            start_time = time.time()
            self.leader_mgr.ensure_leader_minimum(self.archive_mgr.archive, self.population)
            search_tendency = "global" if t < transition_point * max_iter else "local"

            for wolf in self.population:
                alpha, beta, delta = self.leader_mgr.select_leaders_by_region(self.archive_mgr.get_fronts())
                # ✅ 使用并集代替三头交集，提升多样性、增强探索能力
                base = wolf.Position & (alpha.Position | beta.Position | delta.Position)
                # 控制 base 不超过预算的一定比例（后期 base 越大）
                max_base_len = int(self.budget * (0.3 + 0.4 * (t / max_iter)))
                if len(base) > max_base_len:
                    base = set(random.sample(base, max_base_len))
                needed = self.budget - len(base)

                candidate_nodes = set()
                if search_tendency == "global":
                    for comm_id, candidates in community_samples_global.items():
                        #(每个社区抽样 20% 的 top 节点),样本来源基于 global score，偏好外围连接节点
                        candidate_nodes.update(random.sample(candidates, max(1, int(len(candidates) * 0.2))))
                else:
                    # --- 1. leader 所在社区 ---
                    leader_nodes = alpha.Position | beta.Position | delta.Position
                    leader_comms = set(self.node_to_comm[n] for n in leader_nodes if n in self.node_to_comm)

                    candidate_nodes = set()
                    for comm in leader_comms:
                        candidate_nodes.update(community_samples_local.get(comm, []))

                    # --- 2. 其他社区比例采样 ---
                    extra_comms = list(set(community_samples_local.keys()) - leader_comms)

                    # 动态计算其他社区数量
                    extra_comm_ratio = 0.3
                    num_extra_comm = max(1, int(extra_comm_ratio * needed))
                    num_extra_comm = min(num_extra_comm, len(extra_comms))  # 不超过可选数量

                    random.shuffle(extra_comms)
                    for comm in extra_comms[:num_extra_comm]:
                        extra_candidates = community_samples_local.get(comm, [])
                        if extra_candidates:
                            candidate_nodes.update(random.sample(extra_candidates, 1))

                    # --- 3. 加入 hub 节点比例 ---
                    hub_ratio = 0.3
                    num_hubs = max(1, int(hub_ratio * needed))
                    num_hubs = min(num_hubs, len(self.hub_nodes))

                    candidate_nodes.update(random.sample(self.hub_nodes, num_hubs))

                #从候选中选择补全节点: 排除 base 和原始 Position；随机选取若干个，填满预算；
                candidate_nodes = list(candidate_nodes - base - wolf.Position)
                random.shuffle(candidate_nodes)
                new_nodes = set(candidate_nodes[:needed])
                new_position = base | new_nodes

                # === [构造新解后] 统一进行扰动判断和扰动操作 ===
                # - Global：偏探索性扰动（只加 random 节点）
                # - Local：偏跳跃性扰动（加 hub + random 节点）
                # - 调用 apply_perturbation 自动判断策略类型

                if random.random() < self.perturb.dynamic_perturbation_prob(t, max_iter, transition_point,stagnation_counter):
                    new_position = self.perturb.apply_perturbation(
                        position=new_position,
                        base=base,
                        t=t,
                        max_iter=max_iter,
                        transition_point=transition_point,
                        graph=self.graph,
                        search_tendency=search_tendency,
                        stagnation_counter=stagnation_counter,
                        hub_nodes=self.hub_nodes
                    )

                wolf.Position = new_position
                wolf.Cost = self.evaluator.evaluate(new_position)

            # === 更新存档与记录 ===
            self.archive_mgr.update(self.population)
            archive = self.archive_mgr.archive
            archive_costs_history.append([sol.Cost for sol in archive])
            hv = self.archive_mgr.calculate_hypervolume(reference_point=(0, 0))
            hv_values.append(hv)

            # === 停滞检测 ===
            if t > 1 and abs(hv_values[-1] - hv_values[-2]) < hv_tolerance:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            times.append(time.time() - start_time)

        print("time=", times)
        return self.archive_mgr.archive, archive_costs_history, hv_values, times