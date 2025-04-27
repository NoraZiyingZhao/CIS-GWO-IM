import numpy as np


class EmptyGrid:
    def __init__(self):
        self.lower = []
        self.upper = []


def createHypercubes(costs, ngrid, alpha):
    nobj = costs.shape[0]
    G = []
    for i in range(nobj):
        grid = EmptyGrid()
        G.append(grid)

    for j in range(nobj):
        min_cj = min(costs[j])
        max_cj = max(costs[j])
        dcj = alpha * (max_cj - min_cj)
        min_cj = min_cj - dcj
        max_cj = max_cj + dcj
        gx = np.linspace(min_cj, max_cj, ngrid - 1)
        G[j].lower = np.insert(gx, 0, float('-inf'))
        G[j].upper = np.append(gx, float('inf'))

    return G

def calculate_hypervolume_2d(archiveCosts):
    pf = archiveCosts#archive_cost = number of grey wolves * number of fitness function
    repoint=np.zeros(pf.shape[0]) # maximize问题，参考点设置为0,0，求谁的面积最大
    popsize = pf.shape[1]#pf的列数，即archivecost中grey wolf的个数
    temp_index = np.argsort(pf[0,:])# 对帕累托前沿解集的第一行（第一个目标）进行升序排序
    sorted_pf = pf[:,temp_index]
    # 将参考点与排序后的帕累托前沿解集合并
    pointset = np.hstack([repoint.reshape(-1, 1), sorted_pf])
    # 初始化超体积为0
    hyp = 0.0
    if popsize == 1:
        point = pf[:,0]
        hyp = np.prod(point - repoint)
    else:
        for i in range(popsize-1):
            cubei = (pointset[1, i + 1] - pointset[0, 0]) * (pointset[0, i + 1] - pointset[0, i])
            hyp += cubei
    return hyp

def calculate_hypervolume(archiveCosts):
    pf = np.array(archiveCosts)
    reference_point = np.zeros(pf.shape[0])  # 三维问题的参考点为 [0, 0, 0]

    dimensions = pf.shape[0]
    if dimensions != 3:
        raise ValueError("This implementation only supports 3-dimensional hypervolume.")

    # 排序（按第一维目标升序）
    temp_index = np.argsort(pf[0, :])
    sorted_pf = pf[:, temp_index]

    # 初始化超体积
    hyp = 0.0

    # 计算三维超体积
    for i in range(sorted_pf.shape[1]):
        if i == 0:
            dx = sorted_pf[0, i] - reference_point[0]
        else:
            dx = sorted_pf[0, i] - sorted_pf[0, i - 1]

        dy = sorted_pf[1, i] - reference_point[1]
        dz = sorted_pf[2, i] - reference_point[2]

        hyp += dx * dy * dz  # 增量体积

    return hyp


# GetGridIndex
def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    if ind < 0 or ind >= array_shape[0] * array_shape[1]:
        return -1
    return ind


def find_first_bigger(a, U, U_size):
    for i in range(U_size):
        if (a < U[i]):
            return i


def getGridIndex(particle, G):
    # num_of_obj=len(particle.cost)
    c = particle.cost

    nobj = len(c)  # nobj = 2
    ngrid = len(G[0].upper)

    ones_array = np.ones(nobj) * ngrid
    SubIndex = np.zeros(nobj)

    for j in range(nobj):
        U = G[j].upper
        U_size = len(U)
        i = find_first_bigger(c[j], U, U_size)
        SubIndex[j] = i

    Index = sub2ind(ones_array, SubIndex[0], SubIndex[1])

    SubIndex = np.array(SubIndex)

    # print(f'particel_cost: {c} Index: {Index}')

    return Index, SubIndex