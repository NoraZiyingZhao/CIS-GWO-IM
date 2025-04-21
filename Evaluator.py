class Evaluator:
    def __init__(self, graph, node_costs):
        self.graph = graph
        self.node_costs = node_costs

    def evaluate(self, seed_set):
        spread = self._simulate_spread(seed_set)
        cost = sum(self.node_costs[v] for v in seed_set)
        fairness = self._evaluate_fairness(seed_set)
        return [spread, cost, fairness]

    def _simulate_spread(self, seed_set):
        return len(seed_set) * 10  # Dummy spread

    def _evaluate_fairness(self, seed_set):
        return 1.0  # Placeholder for actual fairness computation

