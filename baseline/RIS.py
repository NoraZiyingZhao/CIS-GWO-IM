import numpy as np
import time
import random
from collections import Counter


def get_RRS(network):
    source = random.choice(np.unique(network['source']))
    g = network.copy().loc[np.random.uniform(0, 1, network.shape[0]) < network['weight']]
    new_nodes, RRS0 = [source], [source]
    while new_nodes:
        temp = g.loc[g['target'].isin(new_nodes)]
        temp = temp['source'].tolist()
        RRS = list(set(RRS0 + temp))
        new_nodes = list(set(RRS) - set(RRS0))
        RRS0 = RRS[:]
    return RRS


def ris(network, k, mc=1000):
    start_time = time.time()
    R = [get_RRS(network) for _ in range(mc)]
    SEED, timelapse = [], []
    for _ in range(k):
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
        R = [rrs for rrs in R if seed not in rrs]
        timelapse.append((time.time() - start_time))

    return sorted(SEED), timelapse