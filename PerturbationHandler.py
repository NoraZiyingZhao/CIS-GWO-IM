import random
class PerturbationHandler:
    def __init__(self, structure_metrics):
        self.metrics = structure_metrics

    def perturb(self, position):
        if not position:
            return position
        out_node = random.choice(list(position))
        position.remove(out_node)
        candidates = set(self.metrics.graph.nodes()) - position
        far_node = max(candidates, key=lambda v: self.metrics.closeness[v])
        position.add(far_node)
        return position

    def get_probability(self, t, max_iter):
        return 0.05 + 0.25 * (t / max_iter)
