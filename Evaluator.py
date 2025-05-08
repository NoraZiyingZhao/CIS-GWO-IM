import random
import numpy as np
from collections import Counter
import math

class Evaluator:
    """
    Evaluator类（社区感知版本）：
    - 评估扩散性（Spread）
    - 评估社区公平性（Fairness）：加权社区覆盖 + 节点分布均衡性
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

        # ✅ 计算每个社区的节点数（用于加权）
        self.community_sizes = Counter(node_to_comm.values())
        self.total_nodes_in_communities = sum(self.community_sizes.values())

    def IC_model(self, seed_set, max_steps=2, prob=0.1):
        """
        独立级联模型，运行多次模拟，返回平均激活节点数
        :param seed_set: 初始种子节点集合
        :param max_steps: 最大扩散轮数
        :param prob: 每条边的传播概率
        :return: 平均被激活的节点集合大小（或激活率）
        """
        total_activated = 0

        for _ in range(self.simulations):
            activated = set(seed_set)
            new_active = set(seed_set)

            for _ in range(max_steps):
                next_active = set()
                for node in new_active:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in activated and random.random() < prob:
                            next_active.add(neighbor)
                            activated.add(neighbor)
                if not next_active:
                    break
                new_active = next_active

            total_activated += len(activated)

        avg_activated = total_activated / self.simulations
        return avg_activated

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
        # ✅ 加权覆盖率：按社区大小进行加权
        total_weight = self.total_nodes_in_communities #被覆盖的社区的节点总数,作为该种子集合的“加权覆盖度”。
        covered_weight = sum(self.community_sizes.get(comm, 0) for comm in covered_comms)
        coverage_score = covered_weight / (total_weight + 1e-9) #相对加权覆盖率，更真实地反映了“我们覆盖了多大比例的网络社区”。

        # === 节点分布均衡性（信息熵）
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
        综合评估：扩散性 + 社区公平性
        """
        avg_activated = self.IC_model(seed_set)
        spread = avg_activated / self.total_nodes
        fair = self.fairness(seed_set)
        return spread, fair

