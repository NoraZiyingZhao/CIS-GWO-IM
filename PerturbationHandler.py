import random

class PerturbationHandler:
    def __init__(self, structure_metrics):
        self.metrics = structure_metrics

    def perturb(self, position, ratio=0.1):
        """
        根据扰动比例ratio，动态替换seed set中的部分节点。
        """
        if not position:
            return position

        position = set(position)  # 确保集合操作

        num_to_perturb = max(1, int(ratio * len(position)))

        # Step 1: 随机移除部分节点
        remove_nodes = random.sample(list(position), min(num_to_perturb, len(position)))
        position -= set(remove_nodes)

        # Step 2: 构建candidates，排除现有节点和刚移除的节点
        candidates = set(self.metrics.graph.nodes()) - position - set(remove_nodes)

        if len(candidates) < num_to_perturb:
            add_nodes = candidates
        else:
            # 用closeness最大来选择新的节点
            add_nodes = sorted(candidates, key=lambda v: self.metrics.closeness[v], reverse=True)[:num_to_perturb]
            add_nodes = set(add_nodes)

        position |= add_nodes

        return position

    def dynamic_perturbation_prob(self, t, max_iter, transition_point):
        """
        根据迭代进程动态控制每次更新时是否触发扰动操作。
        - 过渡点之前：扰动缓慢增长，从0.05起步，线性缓慢增长至0.15；稳定探索，避免过早扰动。
        - 过渡点之后：扰动快速增长，从0.15快速增长到最大0.45；增强局部多样性，打破收敛停滞，加速发现新优解。
        """
        if t / max_iter < transition_point:
            return 0.05 + 0.1 * (t / (transition_point * max_iter))  # 平缓增长
        else:
            return 0.15 + 0.3 * ((t - transition_point * max_iter) / (max_iter * (1 - transition_point)))  # 加速增长

    def dynamic_perturbation_ratio(self, t, max_iter, transition_point):
        """
        根据迭代进程动态计算扰动比例（影响一次扰动替换多少节点）。
        """
        if t / max_iter < transition_point:
            return 0.05 + 0.05 * (t / (transition_point * max_iter))  # 前期慢慢增长从0.05增长到0.1
        else:
            return 0.1 + 0.1 * ((t - transition_point * max_iter) / (max_iter * (1 - transition_point)))  # 后期快速增长从0.1增长到0.2
