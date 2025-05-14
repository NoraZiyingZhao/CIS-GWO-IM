import networkx as nx

class StructureMetrics:
    def __init__(self, graph):
        self.graph = graph

        # === 中心性指标 ===
        self.degree = dict(graph.degree())                           # 度中心性
        self.closeness = nx.closeness_centrality(graph)             # 接近中心性（路径平均距离倒数）
        self.eigenvector = nx.eigenvector_centrality(graph, max_iter=500)  # ✅ 替代 betweenness，用于 hub

        # 如果需要切换为 harmonic：
        # self.closeness = nx.harmonic_centrality(graph)

    def _normalize(self, values):
        """
        将中心性字典归一化至 [0, 1] 区间，避免数值尺度影响。
        """
        min_val, max_val = min(values.values()), max(values.values())
        return {
            k: (v - min_val) / (max_val - min_val + 1e-9)
            for k, v in values.items()
        }

    def compute_scores(self):
        """
        结构评分函数（一次性输出两类评分）：
        - global：探索性评分，倾向于选 degree 高且离中心远的节点（低 closeness）；
        - local：开发性评分，倾向于选 degree 高且结构中心节点（高 closeness）。
        """
        deg_norm = self._normalize(self.degree)
        clo_norm = self._normalize(self.closeness)

        score_global = {}
        score_local = {}

        for node in self.graph.nodes():
            # Global 偏探索：高 degree + 小 closeness（越远越好）
            score_global[node] = 0.8 * deg_norm[node] + 0.2 * (1 - clo_norm[node])
            # Local 偏开发：高 degree + 高 closeness（越中心越好）
            score_local[node] = 0.5 * deg_norm[node] + 0.5 * clo_norm[node]

        return score_global, score_local
