#based on Iran author: Ahmad Zareie
#完全不分组结果
#BMOGWOgood code smell
import MObaseline.GFMOGWOpackage.Grid
from MObaseline.GFMOGWOpackage.solution import solution
import time
import influencediffusion as im
# from fitness_function import *
from copy import deepcopy
import networkx as nx
import MObaseline.GFMOGWOpackage.init
import MObaseline.GFMOGWOpackage.determination
import MObaseline.GFMOGWOpackage.Leader_Hypervolume as LH
import numpy as np

"""
Input:  instance, budget
Output: optimal seed set, resulting spread, time for each iteration
"""

class gfmogwo:
    def __init__(self, instance, greyWolvesNum, archiveSize, alpha, beta, gamma, nGrid):
        self.instance = instance
        self.dim = instance.num_items # number of vertices of graph -- self.dim
        self.budget = instance.budget
        # Lower bound and upper bound
        self.lb = np.zeros(self.budget, dtype=int) * self.dim
        self.ub = np.ones(self.budget, dtype=int) * self.dim
        # self.attribute = instance.attribute
        # self.config = instance.config = mc.Configuration()
        self.graph = instance.graph
        # self.edge_threshold = instance.edge_threshold
        self.greyWolvesNum = greyWolvesNum
        self.node_list = np.array(list(instance.graph.nodes()))

        self.archiveSize = archiveSize
        self.nGrid = nGrid

        self.alpha = alpha  # Grid inflation parameter
        self.beta = beta  # Leader selection pressure parameter
        self.gamma = gamma  # Repository Member Selection Pressure

        self.greyWolves = MObaseline.GFMOGWOpackage.init.createWolves(self.greyWolvesNum)
        self.archive = []
        self.archive_costs_history = []
        self.solution = solution()
        self.Time = []


    def optimize(self, node_preferences, num_information, node_to_community, total_nodes, total_communities, maxIt):
        print('GFMOGWO' + " is optimizing  \"" + "multi_fobj \"")
        HV_value = []
        all_solutions = []
        each_group_time = []
        self.solution.startTime = time.time()
        greyWolvesNum = self.greyWolvesNum
        beta = self.beta
        gamma = self.gamma
        archiveSize = self.archiveSize
        budget = self.budget
        graph = self.graph


        # graph setting
        self_loops = list(nx.selfloop_edges(self.graph))  # 删除所有自环
        self.graph.remove_edges_from(self_loops)
        low_degree_nodes = [node for node, degree in self.graph.degree() if degree <= 1]  # 删除度数小于等于1的节点
        self.graph.remove_nodes_from(low_degree_nodes)
        self.node_list = np.array(list(self.graph.nodes()))  # 更新节点列表和维度
        self.dim = len(self.node_list)

        # initialize
        MObaseline.GFMOGWOpackage.init.initialize(self, node_preferences, num_information, node_to_community, total_nodes, total_communities)
        MObaseline.GFMOGWOpackage.determination.determineDomination(self)
        nonDominatedSolutions = MObaseline.GFMOGWOpackage.determination.getNonDominatedWolves(self)
        self.archive = np.array([deepcopy(sol) for sol in nonDominatedSolutions])
        # print(len(self.archive))

        # 获取非支配解cost集合
        archiveCosts = MObaseline.GFMOGWOpackage.determination.getCosts(self)
        # 构建非支配解超立方体网格分布
        grid = MObaseline.GFMOGWOpackage.Grid.createHypercubes(archiveCosts, self.nGrid, self.alpha)

        for i in range(len(self.archive)):
            self.archive[i].gridIndex, self.archive[i].gridSubIndex = MObaseline.GFMOGWOpackage.Grid.getGridIndex(self.archive[i], grid)

        # Main Loop counter
        for it in range(maxIt):
            a = 2 - it * ((2) / maxIt)
            print(f'\nIteration: {it}')
            self.solution.executionTime = time.time()

            # Update positions
            for w in self.greyWolves:

                Delta = LH.selectLeader(self, beta)
                Beta = LH.selectLeader(self, beta)
                Alpha = LH.selectLeader(self, beta)
                # print(f'Leaders Selected - Alpha: {Alpha.position}, Beta: {Beta.position}, Delta: {Delta.position}')

                addBack = 0
                if (len(self.archive) > 1):
                    self.archive = np.delete(self.archive, np.where(self.archive == Delta))
                    Beta = LH.selectLeader(self, beta)
                    addBack += 1

                if (len(self.archive) > 2):
                    self.archive = np.delete(self.archive, np.where(self.archive == Beta))
                    Alpha = LH.selectLeader(self, beta)
                    addBack += 1

                if addBack == 2:
                    self.archive = np.append(self.archive, [Delta, Beta])
                elif addBack == 1:
                    self.archive = np.append(self.archive, [Delta])

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                A1 = a * ((2 * r1) - 1)
                C1 = 2 * r2

                D_alpha = abs((C1 * Alpha.position) - w.position)
                X1 = Alpha.position - A1 * abs(D_alpha)

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                A2 = a * ((2 * r1) - 1)
                C2 = 2 * r2

                D_beta = abs((C2 * Beta.position) - w.position)
                X2 = Beta.position - A2 * abs(D_beta)

                # r1 & r2 are random vectors in [0, 1]
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                A3 = a * ((2 * r1) - 1)
                C3 = 2 * r2

                D_delta = abs((C3 * Delta.position) - w.position)
                X3 = Delta.position - A3 * abs(D_delta)

                # updated_position = ((sigmoid((X1 + X2 + X3) / 3) >= np.random.rand(dim))) * 1
                updated_position = np.divide((X1 + X2 + X3), 3)
                w.position = updated_position

                # 根据 position 获取前 self.budget 个最大值的索引  #是否需要？
                top_indices = np.argsort(w.position)[-self.budget:]  # 逆序排列获取最大值
                w.corresponded_nodes = [self.node_list[idx] for idx in top_indices]  # 根据索引选取对应的节点
                w.cost = im.evaluate_objectives(self.graph, w.corresponded_nodes, node_preferences, num_information, node_to_community, total_nodes, total_communities) # 计算成本或适应度

            # 构造新的非支配解
            MObaseline.GFMOGWOpackage.determination.determineDomination(self)
            nonDominatedSolutions = MObaseline.GFMOGWOpackage.determination.getNonDominatedWolves(self)

            # 获取在超立方体下的位置
            for i in range(len(self.archive)):
                self.archive[i].gridIndex, self.archive[i].gridSubIndex = MObaseline.GFMOGWOpackage.Grid.getGridIndex(self.archive[i], grid)

              # 重新构建的非支配解集合中，再次确定非支配解，整合，获得当前迭代 maxIt，当前group的最终archive
            for sol in nonDominatedSolutions:

                case3 = True
                for archMem in self.archive:

                    # new soln dominated by any one in archive
                    if archMem.dominates(sol):
                        case3 = False
                        # print("case3 = ",case3)
                        break

                    # new soln dominates any one in archive
                    if sol.dominates(archMem):
                        # print("sol.dominates archive")
                        sol.gridIndex, sol.gridSubIndex = MObaseline.GFMOGWOpackage.Grid.getGridIndex(sol, grid)
                        self.archive = np.delete(self.archive, np.where(self.archive == archMem))
                        self.archive = np.append(self.archive, deepcopy(sol))

                        case3 = False
                        break

                if case3:
                    if (len(self.archive) < self.archiveSize):
                        self.archive = np.append(self.archive, sol)
                    else:
                        EXTRA = len(self.archive) - self.archiveSize
                        LH.deleteFromRepo(self, EXTRA, gamma)

                        # LH.deleteFromRepo(1, gamma) 可能总有问题，是这里没有重新生成新的grid，
                        # new added solutions located outside the hypercubes, need to update the grids to cover them
                        sol.gridIndex, sol.gridSubIndex = MObaseline.GFMOGWOpackage.Grid.getGridIndex(sol, grid)
                        self.archive = np.append(self.archive, deepcopy(sol))
            print('Updated Archive after handling dominance:', self.archive)

            archiveCosts = MObaseline.GFMOGWOpackage.determination.getCosts(self)
            grid = MObaseline.GFMOGWOpackage.Grid.createHypercubes(archiveCosts, self.nGrid, self.alpha)

            # Collecting corresponded nodes for each Grey Wolf in the archive
            archive_corresponded_nodes_list = []
            for grey_wolf in self.archive:
                archive_corresponded_nodes_list.append(grey_wolf.corresponded_nodes)
            print('Corresponded Nodes List:', archive_corresponded_nodes_list)

            self.solution.endTime = time.time()
            self.Time = self.solution.endTime - self.solution.executionTime
            # print('Iteration Time : ' + str(self.Time))

            each_group_time.append(self.Time)
            hv = MObaseline.GFMOGWOpackage.Grid.calculate_hypervolume(archiveCosts)
            HV_value.append(hv)
            # print(archiveCosts)
                # position = []
                # archive_fitness =[]
                # for wolf in self.archive:
                #     position.append(wolf.position)
                #     archive_fitness=archiveCosts
                # plot.Plot_pareto.show(self.in_, self.fitness, np.array(position), archive_fitness)
        # 最后一次迭代的结果Store archive solutions, costs, and corresponded nodes list
        all_solutions.append(
            (deepcopy(self.archive), archiveCosts, deepcopy(archive_corresponded_nodes_list),deepcopy(each_group_time),deepcopy(HV_value)))


        return all_solutions, archiveCosts, HV_value






