import random

# └── dynamic_perturbation_prob(...) → 是否扰动？
# └── dynamic_perturbation_ratio(...) → 扰动强度
# └── apply_perturbation(...) → 具体替换哪些节点、加哪些节点

class PerturbationHandler:
    def __init__(self, structure_metrics):
        self.metrics = structure_metrics  # 保留引用以防后续需要扩展

    def dynamic_perturbation_prob(self, t, max_iter, transition_point, stagnation_counter):
        """
        动态扰动概率（带反馈增强）：
        - 前期缓慢增长，后期加速；
        - 若连续停滞，扰动概率进一步增加（最多增加到 +0.15）。
        """
        if t / max_iter < transition_point:
            base_prob = 0.05 + 0.1 * (t / (transition_point * max_iter))
        else:
            base_prob = 0.15 + 0.3 * ((t - transition_point * max_iter) / (max_iter * (1 - transition_point)))

        feedback = min(0.015 * stagnation_counter, 0.15)  # 每代增加 1.5%，最多提升 0.15
        return min(base_prob + feedback, 0.9)

    def dynamic_perturbation_ratio(self, t, max_iter, transition_point):
        """
        动态扰动强度（扰动比例）：
        - 前期 5% → 10%，后期 10% → 20%
        - 后续还可在 optimize() 中加入 feedback_boost
        """
        if t / max_iter < transition_point:
            return 0.05 + 0.05 * (t / (transition_point * max_iter))
        else:
            return 0.1 + 0.1 * ((t - transition_point * max_iter) / (max_iter * (1 - transition_point)))


    def apply_perturbation(self, position, base, t, max_iter, transition_point,
                           graph, search_tendency, stagnation_counter, hub_nodes):
        """
        扰动机制：根据当前阶段与停滞状态，对当前解 position 执行扰动（替换部分非 base 节点）

        参数：
        - position: 当前解的节点集（set）
        - base: 遗传保留部分（不参与扰动）
        - t, max_iter, transition_point: 用于计算扰动比例和概率
        - graph: 当前图结构
        - search_tendency: 当前为 'global' 还是 'local'
        - stagnation_counter: 当前 HV 停滞的代数
        - hub_nodes: 预提取的 hub 节点集合

        返回：
        - new_position: 扰动后的节点集
        """
        position = set(position)
        base_ratio = self.dynamic_perturbation_ratio(t, max_iter, transition_point)
        feedback_boost = 0.05 * stagnation_counter
        perturb_ratio = min(base_ratio + feedback_boost, 0.6)

        num_replace = max(1, int(len(position) * perturb_ratio))
        # 动态保留概率：连续停滞越久，越降低保守性（最低为 0.5）
        safe_keep_prob = max(0.5, 0.8 - 0.03 * stagnation_counter)
        # 当前 replace_candidates 是非 base 的节点
        replace_candidates = [
            node for node in (position - base)
            if random.random() > safe_keep_prob
        ]
        # replace_candidates = list(position - base)
        if not replace_candidates:
            # fallback：如果所有非 base 节点都被保留，那就允许随机选一个
            replace_candidates = list(position - base)

        replace_nodes = random.sample(replace_candidates, min(len(replace_candidates), num_replace))

        # 选择新节点来源（global 更随机，local 加入 hub）
        if search_tendency == "local":
            num_hubs = min(len(hub_nodes), int(num_replace * 0.5))
            new_hubs = set(random.sample(hub_nodes, num_hubs))
            new_rand = set(random.sample(list(set(graph.nodes()) - position), num_replace - len(new_hubs)))
            new_nodes = new_hubs | new_rand
        else:
            # global 仅加入 random 节点
            new_nodes = set(random.sample(list(set(graph.nodes()) - position), num_replace))

        # 更新节点集
        position -= set(replace_nodes)
        position |= new_nodes
        # Debug log（可注释）
        print(f"[Perturbation] Gen {t}, mode: {search_tendency}, replacing {num_replace} nodes with {len(new_nodes)} new nodes.")

        return position
