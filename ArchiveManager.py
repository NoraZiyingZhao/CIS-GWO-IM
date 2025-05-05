import random
import copy
from collections import defaultdict

class ArchiveManager:
    """
    管理非支配解存档，支持多层次fronts管理（Fast Non-dominated Sort）
    """
    def __init__(self, archive_size):
        self.archive = []      # 第一层非支配解（最优层）
        self.fronts = []       # 所有层次fronts
        self.archive_size = archive_size

    def dominates(self, wolf1, wolf2):
        """判断wolf1是否支配wolf2（双目标最大化）"""
        return (wolf1.Cost[0] >= wolf2.Cost[0] and wolf1.Cost[1] >= wolf2.Cost[1]) and \
               (wolf1.Cost[0] > wolf2.Cost[0] or wolf1.Cost[1] > wolf2.Cost[1])

    def update(self, wolves):
        """
        快速非支配排序 + 拥挤距离更新 Archive 和 Fronts。
        """
        # === 🧠 Step 1: # 去重：保留 Position 相同但 Cost 更优的狼
        position_map = dict()  # key = frozenset(Position), value = best wolf

        for wolf in self.archive + wolves:
            key = frozenset(wolf.Position)
            if key not in position_map:
                position_map[key] = wolf
            else:
                existing = position_map[key]
                # 比较谁的 Cost 更优（传播和公平性更大）
                if self.dominates(wolf, existing):
                    position_map[key] = wolf

        # 过滤后的候选集
        candidates = list(position_map.values())

        # === Step 2: 快速非支配排序 ===
        # new_solutions = [wolf for wolf in wolves]
        # candidates = self.archive + new_solutions

        S = defaultdict(list)
        n = dict()
        ranks = dict()
        fronts = [[]]

        for i, p in enumerate(candidates):
            S[i] = []
            n[i] = 0
            for j, q in enumerate(candidates):
                if i == j:
                    continue
                if self.dominates(p, q):
                    S[i].append(j)
                elif self.dominates(q, p):
                    n[i] += 1

            if n[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        if not fronts[-1]:
            fronts.pop()

        self.fronts = [[candidates[i] for i in front] for front in fronts]

        # === Step 3: 拥挤距离选择保留第一层 ===
        first_front = self.fronts[0]
        if len(first_front) > self.archive_size:
            first_front = self.calculate_crowding_distance(first_front)
            first_front = first_front[:self.archive_size]

        self.archive = first_front

    def calculate_crowding_distance(self, archive):
        """计算拥挤距离并根据距离排序（从大到小）。"""
        num_solutions = len(archive)
        distances = [0.0] * num_solutions

        for m in range(2):
            archive.sort(key=lambda w: w.Cost[m])
            distances[0] = distances[-1] = float('inf')
            min_m = archive[0].Cost[m]
            max_m = archive[-1].Cost[m]
            if max_m == min_m:
                continue

            for i in range(1, num_solutions - 1):
                distances[i] += (archive[i + 1].Cost[m] - archive[i - 1].Cost[m]) / (max_m - min_m)

        sorted_archive = [w for _, w in sorted(zip(distances, archive), key=lambda pair: pair[0], reverse=True)]
        return sorted_archive

    def calculate_hypervolume(self, reference_point):
        """计算当前第一层存档的HV（最大化问题）。"""
        hv = 0.0
        for wolf in self.archive:
            f1, f2 = wolf.Cost[:2]
            r1, r2 = reference_point
            vol = max(0, f1 - r1) * max(0, f2 - r2)
            hv += vol
        return hv

    def get_archive(self):
        """返回当前第一层存档（深拷贝）。"""
        return copy.deepcopy(self.archive)

    def get_fronts(self):
        """返回所有fronts（多层次）。"""
        return copy.deepcopy(self.fronts)
