import networkx as nx

class StructureMetrics:
    def __init__(self, graph):
        self.graph = graph
        self.closeness = nx.closeness_centrality(graph)  # 计算所有节点的接近中心性 closeness centrality：每个节点 v 到所有其他节点的平均最短路径距离的倒数。
        self.degree = dict(graph.degree())               # 获取所有节点的度（degree）作为 degree centrality

    def score(self, node, t, max_iter, transition_point):
        # 动态权重系数：w1 随迭代逐渐减小（表示 closeness 权重）
        if t / max_iter < transition_point:
            w1 = 0.8 - 0.3 * (t / (transition_point * max_iter))  # 前期closeness为主
        else:
            w1 = 0.5 - 0.3 * ((t - transition_point * max_iter) / (max_iter * (1 - transition_point)))  # 后期degree占主导
        w2 = 1.0 - w1  # degree 的权重随迭代逐渐增加
        closeness_norm = self._normalize(self.closeness)  # 归一化 closeness 值到 0~1
        degree_norm = self._normalize(self.degree)        # 归一化 degree 值到 0~1
        # 返回结构评分：w1 × closeness + w2 × degree，动态组合
        return w1 * closeness_norm[node] + w2 * degree_norm[node]

    def _normalize(self, values):
        min_val, max_val = min(values.values()), max(values.values())  # 找出最小最大值
        # 返回所有值的归一化结果，避免除以零加上 1e-9
        return {k: (v - min_val) / (max_val - min_val + 1e-9) for k, v in values.items()}
# 40%之前：
# w1=0.8, closenness权重占主导；
# 随着迭代推进，w1逐步线性下降到0.5，w2上升到0.5；
# 表现为从**偏向传播能力（closeness）向平衡传播和连通性（degree）**过渡。
# 40%之后：
# w1进一步下降至0.2左右，w2升高到0.8左右；
# 后期以**局部连通性（degree）**为主要参考指标，强化局部开发。
# 数值变化示意：
# 初期强调“远程扩散”（大closeness，小degree节点优先）；
# 后期强调“局部控制”（高degree节点优先）。