import random
import multiprocessing as mp
import math
import numpy as np
from collections import Counter

worker = []
node_num = 0


class Worker(mp.Process):
    def __init__(self, inQ, outQ, G):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        self.G = G

    def run(self):
        while True:
            theta = self.inQ.get()
            # print(theta)
            while self.count < theta:
                v = random.choice(np.unique(self.G['target']))
                rr = generate_rr_set(self.G, v)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num, G):
    global worker
    for i in range(num):
        # print(i)
        worker.append(Worker(mp.Queue(), mp.Queue(), G))
        worker[i].start()


def finish_worker():
    global worker
    for w in worker:
        w.terminate()
    worker = []


def IMM(network, n, k, epsilon=0.5, l=1):
    global node_num
    node_num = n
    l = l * (1 + math.log(2) / math.log(node_num))
    R = sampling(network, n, k, epsilon=epsilon, l=l)
    seed_set, _ = seed_selection(R, k, n)
    return list(seed_set)


def seed_selection(R, k, node_num):
    Sk = set()
    R_len = len(R)
    for _ in range(k):
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        Sk.add(seed)
        R = [rrs for rrs in R if seed not in rrs]
    return Sk, len(R) / R_len
    # Sk = set()
    # rr_degree = [0 for ii in range(node_num + 1)]
    # node_rr_set = dict()
    # # node_rr_set_copy = dict()
    # matched_count = 0
    # for j in range(0, len(R)):
    #     rr = R[j]
    #     for rr_node in rr:
    #         rr_node = int(rr_node)
    #         rr_degree[rr_node] += 1
    #         if rr_node not in node_rr_set:
    #             node_rr_set[rr_node] = list()
    #             # node_rr_set_copy[rr_node] = list()
    #         node_rr_set[rr_node].append(j)
    #         # node_rr_set_copy[rr_node].append(j)
    # for i in range(k):
    #     max_point = rr_degree.index(max(rr_degree))
    #     Sk.add(max_point)
    #     if max_point in node_rr_set:
    #         matched_count += len(node_rr_set[max_point])
    #         index_set = []
    #         for node_rr in node_rr_set[max_point]:
    #             index_set.append(node_rr)
    #         for jj in index_set:
    #             rr = R[jj]
    #             for rr_node in rr:
    #                 rr_node = int(rr_node)
    #                 rr_degree[rr_node] -= 1
    #                 node_rr_set[rr_node].remove(jj)
    # return Sk, matched_count / len(R)


def generate_rr_set(G, v):
    g = G.loc[np.random.uniform(0, 1) < G['weight']]

    new_nodes, RRS0 = [v], [v]
    while new_nodes:
        temp = g.loc[g['target'].isin(new_nodes)]
        temp = temp['source'].tolist()
        RRS = list(set(RRS0 + temp))
        new_nodes = list(set(RRS) - set(RRS0))
        RRS0 = RRS[:]
    return RRS


def sampling(G, n, k, epsilon=0.5, l=1):
    R = []
    LB = 1
    worker_num = 4
    create_worker(worker_num, G)
    epsilon_ = epsilon * math.sqrt(2)
    log_n_k = np.sum([math.log(i) for i in range(n - k + 1, n + 1)]) + np.sum([-math.log(i) for i in range(1, k + 1)])
    lambda_ = ((2 + epsilon_ * 2 / 3) * (log_n_k + l * math.log(n) + math.log(math.log2(n))) * n) / math.pow(epsilon_, 2)
    for i in range(1, int(np.log2(n) - 1) + 1):
        x = n / math.pow(2, i)
        theta_i = lambda_ / x
        for ii in range(worker_num):
            worker[ii].inQ.put((math.floor(theta_i) - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        seed_set_i, fr = seed_selection(R, k, n)
        if n * fr >= (1 + epsilon_) * x:
            LB = n * fr / (1 + epsilon_)
            break

    a = math.sqrt((l * math.log(n) + math.log(2)))
    b = math.sqrt((1 - 1 / math.e) * (log_n_k + l * math.log(n) + math.log(2)))
    lambda_star = 2 * n * (math.pow((1 - 1 / math.e) * a + b, 2)) * math.pow(epsilon, -2)
    theta = lambda_star / LB

    diff = int(theta - len(R))
    if diff > 0:
        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    finish_worker()
    # R.extend([self.generate_rr_set(G) for _ in range(math.floor(theta) - len(R))])
    return R


if __name__ == '__main__':
    a = [1 for _ in range(10)]
    print(np.sum(a))