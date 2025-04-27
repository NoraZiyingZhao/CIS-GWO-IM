from MObaseline.GFMOGWOpackage.Grid import *


# from fitness_function import *

class DefaultLeader:
    def __init__(self):
        self.gridIndex = -1
        self.fitness = float('-inf')  # 最差解的占位值

def selectLeader(self, beta):

    repo = self.archive
    # print("selectleader archive: ", repo)

    ##new 2024/12/19 modified
    if repo is None or len(repo) == 0:
        print("Archive is empty. Selecting a random default leader.")
        return DefaultLeader()  # 默认

    # Unique occupied grids and no.of objects in each
    occ_cell_index, occ_cell_member_count = getOccupiedCells(self)
    # print("occ_cell_member_count:", occ_cell_member_count)

    ##new 2024/12/19 modified
    if len(occ_cell_member_count) == 0 or np.all(occ_cell_member_count == 0):
        print("No occupied cells found. Selecting a random leader.")
        return np.random.choice(repo)

    # Leader has to be selected from least crowded area.
    # Probability inversely related to count

    p = np.power(occ_cell_member_count.astype(float), -beta)
    ##new 2024/12/19 modified
    p = np.maximum(p, 1e-10)  # 防止概率值为0
    p = p / sum(p)
    # print(p)

    # Selecting the grid by rouletteWheelSelection
    selected_cell_index = occ_cell_index[rouletteWheelSelection(p)]

    # Selecting objects from selected grid
    selected_cell_members = np.array([obj for obj in repo if obj.gridIndex == selected_cell_index])

    ##new 2024/12/19 modified
    if len(selected_cell_members) == 0:
        print("No members in the selected cell. Selecting a random leader.")
        return np.random.choice(repo)

    # Selecting random object from selected grid
    return np.random.choice(selected_cell_members)

def deleteFromRepo(self, extra, gamma):

    for k in range(extra):
        repo = self.archive

        ##new 2024/12/19 modified
        if len(repo) <= 1:  # 设置最小存档数量
            print("Archive size is too small. Skipping deletion.")
            return

        # Unique occupied grids and no.of objects in each
        occ_cell_index, occ_cell_member_count = getOccupiedCells(self)

        ##new 2024/12/19 modified
        if len(occ_cell_member_count) == 0 or np.all(occ_cell_member_count == 0):
            print("No occupied cells found. Skipping deletion.")
            return
        # print(occ_cell_member_count)

        # Deletion has to be done from most crowded area
        p = np.power(occ_cell_member_count.astype(float), gamma)
        p = np.maximum(p, 1e-10)  # 防止概率值为0
        p = p / sum(p)

        # Selecting the grid by rouletteWheelSelection
        selected_cell_index = occ_cell_index[rouletteWheelSelection(p)]

        # Selecting objects from selected grid
        selected_cell_members_index = np.array(
            [i for i in range(len(repo)) if repo[i].gridIndex == selected_cell_index])

        ##new 2024/12/19 modified
        if len(selected_cell_members_index) == 0:
            print("No members in the selected cell. Skipping deletion.")
            continue

        # Selecting object to be deleted from selected grid
        del_obj_index = np.random.choice(selected_cell_members_index)

        self.archive = np.delete(repo, del_obj_index)

# def getOccupiedCells(self):
#
#     repo = self.archive
#
#     # gridIndices = np.array([obj.gridIndex for obj in repo], dtype=object)
#     gridIndices = []
#     for i in range(len(repo)):
#         gridIndices.append(repo[i].gridIndex)
#
#     occ_cell_ind = np.unique(gridIndices)
#
#     occ_cell_member_count = np.zeros(len(occ_cell_ind))
#
#     for k in range(len(occ_cell_ind)):
#         occ_cell_member_count[k] = np.count_nonzero(gridIndices == occ_cell_ind[k])
#
#     return occ_cell_ind, occ_cell_member_count

def getOccupiedCells(self):
    repo = self.archive

    if repo is None or len(repo) == 0:
        print("Archive is empty in getOccupiedCells.")
        return np.array([0]), np.array([1])  # 默认返回一个占据的网格

    # print("getOccupiedCells archive: ", repo)
    # 提取 GridIndex 属性并展平成一维列表
    gridIndices = []
    for obj in repo:
        # Debug: 打印每个对象的gridIndex
        # print("Object gridIndex: ", obj.gridIndex)
        # print("Object subgridIndex: ", obj.gridSubIndex)
        if isinstance(obj.gridIndex, (list, np.ndarray)):
            gridIndices.extend(obj.gridIndex)
        elif obj.gridIndex is not None:
            gridIndices.append(obj.gridIndex)

    # 转换为 NumPy 数组
    gridIndices = np.array(gridIndices)
    # 检查是否有任何有效的 gridIndex
    if gridIndices.size == 0:
        print("No valid grid indices in archive. Using default grid index.")
        return np.array([0]), np.array([1])  # 默认返回一个占据的网格

    # 找到唯一值
    occ_cell_ind = np.unique(gridIndices)

    # 计算每个唯一值的出现次数
    occ_cell_member_count = np.zeros(len(occ_cell_ind))
    for k in range(len(occ_cell_ind)):
        occ_cell_member_count[k] = np.sum(gridIndices == occ_cell_ind[k])

    return occ_cell_ind, occ_cell_member_count

def rouletteWheelSelection(p):
    if len(p) == 0:
        print("Empty probability array. Returning default index.")
        return -1  # 返回默认索引 -1 表示无效选择
    p = np.maximum(p, 1e-10)  # 确保概率值不为零
    p = p / sum(p)
    return np.random.choice(np.arange(len(p)), p=p)