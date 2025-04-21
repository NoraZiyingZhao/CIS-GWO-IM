import random

class CandidateSelector:
    def __init__(self, graph, structure_metrics):
        self.graph = graph
        self.metrics = structure_metrics

    def generate_pool(self, strategy, base, current, alpha, beta, delta):
        if strategy == 'global':
            return set(self.graph.nodes()) - base - current
        else:
            leaders = alpha | beta | delta
            non_base = leaders - base
            neighbors = set()
            for node in non_base:
                neighbors.update(self.graph.neighbors(node))
            return (non_base | neighbors) - base - current

    def select_nodes(self, pool, k, t, max_iter):
        ranked = sorted(pool, key=lambda v: self.metrics.score(v, t, max_iter), reverse=True)
        return set(ranked[:k])