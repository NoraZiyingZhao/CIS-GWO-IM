import random
import numpy as np
from collections import Counter
import math

class Evaluator:
    """
    Evaluator类（简化版串行版本）：
    - 计算扩散性（Spread）
    - 计算基于社区的公平性（Fairness）
    """

    def __init__(self, graph, node_to_comm, total_communities, num_information=1, simulations=10):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.total_nodes = len(self.nodes)
        self.num_information = num_information
        self.simulations = simulations

        # ✅ 添加社区信息
        self.node_to_comm = node_to_comm
        self.total_communities = total_communities

    def reductionp(self, similarity_value=1.0, reduction_factor=0.1):
        """
        简化传播概率函数，目前无节点偏好
        """
        return similarity_value * reduction_factor

    def IC_model(self, S, max_steps=2):
        """
        信息扩散过程（串行版本）
        """
        node_info_counts = {node: [0] * self.num_information for node in self.graph.nodes()}

        for _ in range(self.simulations):
            local_counts = {node: [0] * self.num_information for node in self.graph.nodes()}
            for info_index in range(self.num_information):
                new_active = set(S)
                activated_nodes = set(S)
                step = 0

                while new_active and step < max_steps:
                    next_active = set()
                    for node in new_active:
                        for neighbor in self.graph.neighbors(node):
                            if neighbor not in activated_nodes:
                                prob = self.reductionp()
                                if random.random() < prob:
                                    next_active.add(neighbor)
                                    activated_nodes.add(neighbor)
                                    local_counts[neighbor][info_index] += 1
                    new_active = next_active
                    step += 1

            for node in local_counts:
                for info_index in range(self.num_information):
                    node_info_counts[node][info_index] += local_counts[node][info_index]

        return node_info_counts



    def fairness(self, seed_set, w_cov=0.5, w_entropy=0.5):
        """
        coverage：这些种子覆盖了多少个不同的社区（越多越好）；
        entropy：这些种子在社区之间分布得是否均匀（越均衡越好）
        基于社区的公平性计算
        - coverage_score：覆盖社区比例
        - entropy_score：社区分布均匀性
        """
        if not seed_set:
            return 0.0

        covered_comms = set(self.node_to_comm.get(n) for n in seed_set if n in self.node_to_comm)#获取所有种子节点所涉及的社区编号集合
        coverage_score = len(covered_comms) / self.total_communities #计算覆盖率，即种子集覆盖了多少比例的社区

        comm_counts = Counter(self.node_to_comm[n] for n in seed_set if n in self.node_to_comm) #统计每个社区中有多少个节点被选作种子
        probs = [count / len(seed_set) for count in comm_counts.values()] #把各社区中的种子数量转化为“比例分布”
        entropy = -sum(p * math.log(p + 1e-9) for p in probs) #计算这些种子在不同社区中的信息熵（衡量分布的均匀性）
        #熵越高，表示越分散（均衡），熵越低，表示集中在少数社区。
        max_entropy = math.log(len(comm_counts)) if comm_counts else 1.0 #对熵进行归一化，范围变成 0 到 1
        entropy_score = entropy / (max_entropy + 1e-9)

        fairness_score = w_cov * coverage_score + w_entropy * entropy_score #综合两个指标，加权求和得到最终公平性得分
        return fairness_score

    def evaluate(self, seed_set):
        """
        评估：扩散性 + 社区公平性
        """
        node_info_counts = self.IC_model(seed_set)

        # 目标1：扩散率
        activated_nodes = sum(1 for node, counts in node_info_counts.items() if any(count > 0 for count in counts))
        spread = activated_nodes / self.total_nodes

        # 目标2：社区公平性
        fair = self.fairness(seed_set)

        return spread, fair
