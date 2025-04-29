import random
import numpy as np

class Evaluator:
    """
    Evaluator类（简化版串行版本）：
    - 只计算扩散性（Spread）
    - Fairness 占位返回0
    - 传播过程串行执行，保证一致性
    """

    def __init__(self, graph, num_information=1, simulations=10):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.total_nodes = len(self.nodes)
        self.num_information = num_information
        self.simulations = simulations

    def reductionp(self, similarity_value=1.0, reduction_factor=0.1):
        """
        当前没有节点偏好，默认 similarity_value=1.0
        """
        return similarity_value * reduction_factor

    def IC_model(self, S, max_steps=2):
        """
        简化版IC传播模型（纯串行）。
        """
        node_info_counts = {node: [0] * self.num_information for node in self.graph.nodes()}

        for _ in range(self.simulations):
            local_counts = {node: [0] * self.num_information for node in self.graph.nodes()}
            for info_index in range(self.num_information):
                new_active = set(S)  # make sure new_active is a set
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

            # 汇总这次仿真结果
            for node in local_counts:
                for info_index in range(self.num_information):
                    node_info_counts[node][info_index] += local_counts[node][info_index]

        return node_info_counts

    def evaluate(self, seed_set):
        """
        评估给定seed set的扩散性和占位fairness。
        """
        node_info_counts = self.IC_model(seed_set)

        # 目标1：扩散率
        activated_nodes = sum(1 for node, counts in node_info_counts.items() if any(count > 0 for count in counts))
        spread = activated_nodes / self.total_nodes

        # 目标2：公平性（占位）
        fairness = 0.5

        return spread, fairness
