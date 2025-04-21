# StructMOGWO - Main entry point
from GreyWolf import *
from ArchiveManager import *
from CandidateSelector import *
from Evaluator import *
from Population import *
from PerturbationHandler import *
from StructureMetrics import *
from visualise import *

if __name__ == '__main__':
    import networkx as nx

    # Load example graph
    G = nx.erdos_renyi_graph(n=100, p=0.05, seed=42)
    node_costs = {v: 1 for v in G.nodes()}  # Uniform cost for simplicity

    # Parameters
    budget = 5
    pop_size = 30
    archive_size = 50
    max_iter = 50

    # Instantiate components
    metrics = StructureMetrics(G)
    selector = CandidateSelector(G, metrics)
    perturb = PerturbationHandler(metrics)
    evaluator = Evaluator(G, node_costs)
    archive = ArchiveManager(archive_size)
    visual = Visualizer()

    # Initialize population
    pop = Population(G, evaluator, metrics, selector, perturb, archive, budget, pop_size)

    # Run optimization
    for t in range(max_iter):
        pop.iterate(t, max_iter)
        visual.record_hv(archive.archive)
        print(f"Iteration {t+1}, Archive size: {len(archive.archive)}")

    # Output final solutions
    print("\nFinal Pareto Archive:")
    for wolf in archive.archive:
        print(f"Seeds: {wolf.Position}, Cost: {wolf.Cost}")

    visual.plot_hv()
    visual.save_archive_csv(archive.archive)
