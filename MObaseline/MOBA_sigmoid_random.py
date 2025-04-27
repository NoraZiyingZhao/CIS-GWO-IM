import numpy as np
import random
import influencediffusion as im
import paretosolution as ps
import HV as hv
import time
from copy import deepcopy

def bat_algorithm_moba(graph, num_nodes, population_size, budget, node_preferences_dict,
                       num_information, node_to_community, total_nodes, total_communities, max_iter):
    """
    Multi-Objective Bat Algorithm (MOBA) for influence maximization with dynamic loudness and pulse rate.
    """
    Q_min = 0.5
    Q_max = 1.5
    A_init = 1.0 #初始响度,调高以增强探索能力
    A_min = 0.1 #最小响度限制,防止过度收敛导致搜索停滞
    r_init = 0.1 #初始脉冲发射率,增大初始值增强局部搜索能力
    r_max = 0.9 #最大脉冲发射率,控制局部搜索的上限强度
    alpha = 0.9  #响度递减因子,降低值加速收敛；提高值增强探索能力。
    gamma = 0.9   #脉冲发射率增长因子,增加值提升局部搜索能力的增强速率
    archive_size = 100

    all_solutions = []
    n_bats = population_size
    Q = np.zeros(n_bats)  # Frequency
    v = np.zeros((n_bats, total_nodes))  # Velocity

    # Initialize loudness (A) and pulse rate (r)
    A = np.full(n_bats, A_init)  # Initial loudness for each bat
    r = np.full(n_bats, r_init)  # Initial pulse rate for each bat

    # Initialize bats (solutions)
    bats = [random.sample(range(total_nodes), budget) for _ in range(n_bats)]

    # Archive for Pareto optimal solutions
    archive = []
    reference_point = [0, 0, 0]  # Hypervolume reference point for 3 objectives maximization
    final_HV = []
    each_group_time = []

    # Evaluate initial solutions and update archive
    for bat in bats:
        fitness = im.evaluate_objectives(graph, bat, node_preferences_dict, num_information, node_to_community,
                                         total_nodes, total_communities)
        archive = ps.update_elite_archive(archive, [(bat, fitness)], archive_size)
    print("initialization finished")

    # Iterative optimization
    for t in range(max_iter):
        print("iteration:", t)
        start_time = time.time()
        new_solutions = []  # Temporary storage for new solutions

        for i in range(n_bats):
            # Update frequency
            Q[i] = np.random.uniform(Q_min, Q_max)

            # Update velocity
            v[i] += (np.random.rand(total_nodes) - np.array([1 if node in bats[i] else 0 for node in range(total_nodes)])) * Q[i]

            # Generate new solution based on velocity (probabilistic node selection)
            probabilities = 1 / (1 + np.exp(-v[i]))  # Sigmoid function
            new_bat = [node for node, prob in enumerate(probabilities) if random.random() < prob]

            # Enforce budget constraint
            if len(new_bat) > budget:
                new_bat = random.sample(new_bat, budget)

            # Local search with probability (Pulse rate)
            if random.random() > r[i]:
                # Add random perturbation to a representative solution from the archive
                best_bat = random.choice([solution for solution, fitness in archive])
                new_bat = random.sample(best_bat, min(budget, len(best_bat)))

            # Evaluate the new solution
            new_fitness = im.evaluate_objectives(graph, new_bat, node_preferences_dict, num_information,
                                                 node_to_community, total_nodes, total_communities)

            # Accept the new solution based on loudness (A)
            if random.random() < A[i]:
                bats[i] = new_bat
                fitness = new_fitness
                archive = ps.update_elite_archive(archive, [(new_bat, fitness)], archive_size)

            # Update loudness and pulse rate dynamically
            A[i] = max(A[i] * alpha, A_min)  # Loudness decreases but not below A_min
            r[i] = min(r[i] * (1 - np.exp(-gamma * t)), r_max)  # Pulse rate increases but not above r_max

            # Store the new solution
            new_solutions.append((bats[i], new_fitness))

        # Update Pareto archive
        archive = ps.update_elite_archive(archive, new_solutions, archive_size)

        # Calculate hypervolume for this iteration
        final_hv_value = hv.calculate_hypervolume(archive, reference_point)
        final_HV.append(final_hv_value)

        # Record iteration time
        end_time = time.time()
        iteration_time = end_time - start_time
        each_group_time.append(iteration_time)

    all_solutions.append((deepcopy(archive), deepcopy(each_group_time)))
    return all_solutions, final_HV