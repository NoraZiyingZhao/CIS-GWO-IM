import numpy as np
import random
import paretosolution as ps
import time
from copy import deepcopy
from Evaluator import *

def sigmoid_mapping(velocity):
    return 1 / (1 + np.exp(-velocity))

class MODBA:
    def __init__(self, graph, budget, pop_size, archive_size, node_to_comm, total_communities):
        self.graph = graph
        self.budget = budget
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.node_to_comm = node_to_comm
        self.total_communities = total_communities
        self.total_nodes = len(graph.nodes)
        self.evaluator = Evaluator(graph, node_to_comm, total_communities)

    def optimize(self, max_iter):
        Q_min, Q_max = 0.5, 1.5
        A_init, A_min = 1.0, 0.1
        r_init, r_max = 0.1, 0.9
        alpha, gamma = 0.9, 0.9

        n_bats = self.pop_size
        Q = np.zeros(n_bats)
        v = np.zeros((n_bats, self.total_nodes))
        A = np.full(n_bats, A_init)
        r = np.full(n_bats, r_init)

        bats = [random.sample(list(self.graph.nodes), self.budget) for _ in range(n_bats)]
        archive = []
        reference_point = [0, 0]  # Adjust to [0, 0, 0] for 3 objectives
        final_HV = []
        each_group_time = []
        all_solutions = []

        for bat in bats:
            fitness = self.evaluator.evaluate(bat)
            archive = ps.update_elite_archive(archive, [(bat, fitness)], self.archive_size)

        print("Initialization finished.")

        for t in range(max_iter):
            print(f"Iteration {t + 1}")
            start_time = time.time()
            new_solutions = []

            for i in range(n_bats):
                Q[i] = np.random.uniform(Q_min, Q_max)
                leader = archive[0][0] if archive else bats[i]

                xi = np.array([1 if node in bats[i] else 0 for node in range(self.total_nodes)])
                x0 = np.array([1 if node in leader else 0 for node in range(self.total_nodes)])
                v[i] += (xi ^ x0) * Q[i]

                probabilities = sigmoid_mapping(v[i])
                new_bat = [node for node, prob in enumerate(probabilities) if random.random() < prob]

                if len(new_bat) > self.budget:
                    new_bat = random.sample(new_bat, self.budget)
                elif len(new_bat) < self.budget:
                    additional = list(set(self.graph.nodes) - set(new_bat))
                    new_bat += random.sample(additional, self.budget - len(new_bat))

                if random.random() > r[i]:
                    new_bat = mutation_operator(new_bat, self.graph)
                else:
                    new_bat = turbulence_operator(new_bat, self.graph)

                new_fitness = self.evaluator.evaluate(new_bat)

                if random.random() < A[i]:
                    bats[i] = new_bat
                    archive = ps.update_elite_archive(archive, [(new_bat, new_fitness)], self.archive_size)

                A[i] = max(A[i] * alpha, A_min)
                r[i] = min(r[i] * (1 - np.exp(-gamma * t)), r_max)

                new_solutions.append((bats[i], new_fitness))

            archive = ps.update_elite_archive(archive, new_solutions, self.archive_size)

            hv_value = ps.calculate_hypervolume(archive, reference_point)
            final_HV.append(hv_value)
            each_group_time.append(time.time() - start_time)

        all_solutions.append((deepcopy(archive), deepcopy(each_group_time)))
        return archive, all_solutions, final_HV, each_group_time


# --- 辅助函数保持不变 ---
def mutation_operator(position, graph):
    for i in range(len(position)):
        if random.random() < 0.2:
            neighbors = list(graph.neighbors(position[i]))
            if neighbors:
                position[i] = random.choice(neighbors)
    return position

def turbulence_operator(position, graph):
    for i in range(len(position)):
        if random.random() < 0.1:
            neighbors = list(graph.neighbors(position[i]))
            if neighbors:
                position[i] = random.choice(neighbors)
    return position
