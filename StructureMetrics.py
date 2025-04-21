import networkx as nx

class StructureMetrics:
    def __init__(self, graph):
        self.graph = graph
        self.closeness = nx.closeness_centrality(graph)
        self.degree = dict(graph.degree())

        self.closeness_norm = self._normalize(self.closeness, reverse=True)
        self.degree_norm = self._normalize(self.degree)

    def _normalize(self, values, reverse=False):
        vals = list(values.values())
        min_val, max_val = min(vals), max(vals)
        norm = {}
        for k, v in values.items():
            score = (v - min_val) / (max_val - min_val + 1e-9)
            norm[k] = 1 - score if reverse else score
        return norm

    def score(self, v, t, max_iter):
        w1 = 0.8 - 0.6 * (t / max_iter)
        w2 = 1 - w1
        return w1 * self.closeness_norm[v] + w2 * self.degree_norm[v]
