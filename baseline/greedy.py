import numpy as np


def diffuse(network, seed_set, episodes=100, time_steps=100):
    spread = []
    for i in range(episodes):
        new_active, already_active = seed_set[:], seed_set[:]
        for _ in range(time_steps):
            if len(new_active) == 0:
                break
            temp = network.loc[network['source'].isin(new_active)]
            targets = temp['target'].tolist()
            success = np.random.uniform(0, 1, len(targets)) < temp['weight']
            new_ones = np.extract(success, targets)
            new_active = list(set(new_ones) - set(already_active))
            already_active += new_active
        spread.append(len(already_active))
    return np.mean(spread)


def seed_selection(network, k):
    seed_set, spread_results = [], []
    for _ in range(k):
        best_spread = 0
        for user in set(network.nodes()) - set(seed_set):
            s = diffuse(network, seed_set + [user])
            if s > best_spread:
                best_spread, best_node = s, user
        seed_set.append(best_node)
        spread_results.append(best_spread)
    return spread_results, seed_set