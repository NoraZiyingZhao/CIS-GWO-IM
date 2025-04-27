from MObaseline.GFMOGWOpackage.GreyWolf import *
# from fitness_function import *
from MObaseline.GFMOGWOpackage.GFMOGWO import *

def initialize(self,node_preferences, num_information, node_to_community, total_nodes, total_communities):
    # 为每只灰狼分配一个唯一的起始位置，计算该位置的成本或适应度，并记录最佳状态。
    for wolf in self.greyWolves:
        # 生成每个节点的位置，其中位置是基于随机
        wolf.position_dict = {node: np.random.rand() for node in self.node_list}
        # 转换字典为向量形式的位置信息
        wolf.position = np.array([wolf.position_dict[node] for node in self.node_list])
        # 根据 position 获取前 self.budget 个最大值的索引
        top_indices = np.argsort(wolf.position)[-self.budget:]  # 逆序排列获取最大值
        wolf.corresponded_nodes = [self.node_list[idx] for idx in top_indices]  # 根据索引选取对应的节点
        # 计算成本或适应度
        wolf.cost = im.evaluate_objectives(self.graph, wolf.corresponded_nodes, node_preferences, num_information, node_to_community, total_nodes, total_communities) # 计算成本或适应度
        wolf.best_position = np.copy(wolf.position)
        wolf.best_seed_set = np.copy(wolf.corresponded_nodes)
        wolf.best_cost = np.copy(wolf.cost)

def createWolves(greyWolvesNum):
    wolves = []
    for i in range(greyWolvesNum):
        wolves.append(GreyWolf())
    wolves = np.array(wolves, dtype=object)
    return wolves


#这段代码实现了:假设greywolves=100，attributes=age,将图分为7组，
# 首先获取每个小组的节点集subgroup={}，初始化整个狼群，在全图范围内确定position和对应的corresponded nodes
# calculate fitness function 求得对每个小组的影响最大的初始解集（influence spread）
# ============================ 截止目前，找到了能够影响每个小组的initial节点集。
# 然后通过GFMOGWO迭代，获得每一个小组的archive(i)，再从archive中提取一定比例的解，作为最终的解集。
# 获取每组节点数nodes和占总节点数的比重proportion，计算出后面在每个小组解中获取archive个体的比例。如：archive=archive(1+2+3+4+5+6+7) <=budget

# def initialize(self,count):
#     # 为每只灰狼分配一个唯一的起始位置，计算该位置的成本或适应度，并记录最佳状态。
#     for wolf in self.greyWolves:
#         # 生成每个节点的位置，其中位置是随机生成的
#         wolf.position_dict = {node: np.random.rand() for node in self.node_list}
#         # 转换字典为向量形式的位置信息
#         wolf.position = np.array([wolf.position_dict[node] for node in self.node_list])
#         # 根据 position 获取前 self.budget 个最大值的索引
#         top_indices = np.argsort(wolf.position)[-self.budget:]  # 逆序排列获取最大值
#         wolf.corresponded_nodes = [self.node_list[idx] for idx in top_indices]  # 根据索引选取对应的节点
#         # 计算成本或适应度
#         wolf.cost = calculate_fitness(self.instance, wolf.corresponded_nodes,self.graph,count)
#         wolf.best_position = np.copy(wolf.position)
#         wolf.best_seed_set = np.copy(wolf.corresponded_nodes)
#         wolf.best_cost = np.copy(wolf.cost)


#这段代码实现了:假设greywolves=100，attributes=age,将图分为5组，
# 首先获取每组节点数nodes和占总节点数的比重proportion，计算出应该分配给每个小组进行搜索的狼的个数group_size={组1:2只；组2:3只；组3:4只；组4:1只；组5:2只}
# 其次，确保按比例分配的狼没有超过总狼数
# 然后，对其中一个组初始化group_wolves，用于存储当前组的灰狼；边界检查，确定没有超出总狼数，
# 对每组group_size个狼在全图范围内确定position和对应的corresponded nodes，在group_wolves中存储当前组的灰狼
# ============================ 截止目前，找到了每个组分配的狼和对应的节点，即能够影响每个小组的initial节点集。
# 然后通过GFMOGWO迭代，获得每一个小组的archive(i)，再从archive中提取一定比例的解，作为最终的解集。

# def initialize(self, group_info):
#     # 计算每个组应分配的灰狼数量
#     total_wolves = len(self.greyWolves)
#     # 计算每个组分配的灰狼数量，按比例分配,dictionary
#     group_sizes = {attr_val: int(round(total_wolves * info['proportion'])) for attr_val, info in group_info.items()}
#
#     # 确保分配的总数量等于 total_wolves
#     allocated_wolves = sum(group_sizes.values())
#     if allocated_wolves != total_wolves:
#         difference = total_wolves - allocated_wolves
#         # 按照每个组的比例从大到小排序
#         sorted_groups = sorted(group_info.items(), key=lambda x: x[1]['proportion'], reverse=True)
#         for i in range(abs(difference)):
#             attr_val, _ = sorted_groups[i % len(sorted_groups)]
#             group_sizes[attr_val] += np.sign(difference) # 根据差异调整组大小
#
#     wolf_index = 0
#     group_wolf_info = {attr_val: {'wolves': [], 'positions': [], 'nodes': [], 'costs': []} for attr_val in
#                        group_info.keys()}
#
#     for attr_val, group_size in group_sizes.items():
#         group_wolves = [] #store grey wolves inside current group
#         for _ in range(group_size):
#             #boundary check
#             if wolf_index >= total_wolves:
#                 break
#
#             wolf = self.greyWolves[wolf_index]
#             wolf.position = np.random.rand(self.dim)
#             top_indices = np.argsort(wolf.position)[-self.budget:]
#             wolf.corresponded_nodes = [self.node_list[idx] for idx in top_indices]
#             wolf.cost = calculate_fitness(self.instance, wolf.corresponded_nodes, self.graph)
#             wolf.best_position = np.copy(wolf.position)
#             wolf.best_seed_set = np.copy(wolf.corresponded_nodes)
#             wolf.best_cost = np.copy(wolf.cost)
#
#             group_wolf_info[attr_val]['wolves'].append(wolf)
#             group_wolf_info[attr_val]['positions'].append(wolf.position)
#             group_wolf_info[attr_val]['nodes'].append(wolf.corresponded_nodes)
#             group_wolf_info[attr_val]['costs'].append(wolf.cost)
#
#             wolf_index += 1
#     return group_wolf_info



