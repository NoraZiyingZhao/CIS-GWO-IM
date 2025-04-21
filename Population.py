import random
from GreyWolf import *
class Population:
    def __init__(self, graph, evaluator, structure_metrics, candidate_selector, perturb_handler, archive_manager, budget, population_size):
        self.graph = graph
        self.evaluator = evaluator
        self.metrics = structure_metrics
        self.selector = candidate_selector
        self.perturb = perturb_handler
        self.archive_mgr = archive_manager
        self.budget = budget
        self.population = [self._create_wolf() for _ in range(population_size)]

    def _create_wolf(self):
        wolf = GreyWolf()
        nodes = list(self.graph.nodes())
        wolf.Position = set(random.sample(nodes, self.budget))
        wolf.Cost = self.evaluator.evaluate(wolf.Position)
        return wolf

    def iterate(self, t, max_iter):
        # Update archive
        self.archive_mgr.update(self.population)
        archive = self.archive_mgr.archive
        alpha, beta, delta = archive[0], archive[min(1, len(archive)-1)], archive[min(2, len(archive)-1)]

        for wolf in self.population:
            base = wolf.Position & alpha.Position & beta.Position & delta.Position
            strategy = random.choices(['global', 'leader_neighbor'], weights=[0.6, 0.4])[0]
            pool = self.selector.generate_pool(strategy, base, wolf.Position, alpha.Position, beta.Position, delta.Position)
            new_nodes = self.selector.select_nodes(pool, self.budget - len(base), t, max_iter)
            new_position = base | new_nodes

            # Optional perturbation
            if random.random() < self.perturb.get_probability(t, max_iter):
                new_position = self.perturb.perturb(new_position)

            wolf.Position = new_position
            wolf.Cost = self.evaluator.evaluate(new_position)