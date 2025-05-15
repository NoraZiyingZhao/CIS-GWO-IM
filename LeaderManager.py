import math
import random
from collections import defaultdict

class LeaderManager:
    """
    使用极角三分区 + 拥挤距离选择 leader1-4。
    """

    def __init__(self, crowding_distance_fn):
        """
        crowding_distance_fn: 函数引用，用于计算拥挤距离并返回排序后的个体列表（从高到低）。
        """
        self.crowding_distance_fn = crowding_distance_fn

    def normalize_costs(self, wolves):
        """将目标值归一化到 [0, 1] 区间"""
        objs = list(zip(*[w.Cost[:2] for w in wolves]))
        norm_objs = []
        for i in range(2):
            min_val = min(objs[i])
            max_val = max(objs[i])
            norm_objs.append([(v - min_val) / (max_val - min_val + 1e-9) for v in objs[i]])
        return list(zip(norm_objs[0], norm_objs[1]))

    def classify_region(self, norm_cost):
        f1, f2 = norm_cost
        if f1 == 0 and f2 == 0:
            return 'tradeoff_zone'
        angle = math.atan2(f2, f1)  # angle ∈ [0, π/2]

        if angle <= math.pi / 6:
            return 'spread_zone'
        elif angle <= math.pi / 3:
            return 'tradeoff_zone'
        else:
            return 'fair_zone'

    def select_leaders_with_tradeoff_explorer(self, fronts):
        """
        多层 front 逐层扫描 + 区域划分 + 拥挤距离选探索 leader
        返回 4 个 leader：
        - spread_zone 最优
        - fair_zone 最优
        - tradeoff_zone 最优
        - tradeoff_zone 中稀疏区域探索 leader
        """
        leaders = {}
        zone_order = ['spread_zone', 'fair_zone', 'tradeoff_zone']
        zone_map = {'spread_zone': [], 'fair_zone': [], 'tradeoff_zone': []}

        # === Step 1: 多层 fronts 扫描 + 区域划分 ===
        for front in fronts:
            norm_costs = self.normalize_costs(front)
            for wolf, norm in zip(front, norm_costs):
                region = self.classify_region(norm)
                if region not in zone_map:
                    continue
                zone_map[region].append(wolf)

            # Step 2: 每个区域按排序逻辑选最优（若未选过）
            for zone in zone_order:
                if zone in leaders:
                    continue
                if zone_map[zone]:
                    sort_key = {
                        'spread_zone': lambda w: -w.Cost[0],
                        'fair_zone': lambda w: -w.Cost[1],
                        'tradeoff_zone': lambda w: -(w.Cost[0] + w.Cost[1])
                    }
                    zone_map[zone].sort(key=sort_key[zone])
                    leaders[zone] = zone_map[zone][0]

            if len(leaders) >= 3:
                break  # 三个主 leader 已选够

        # === Step 3: Leader4 - 从 tradeoff_zone 中选择稀疏区域个体 ===
        tradeoff_pool = zone_map['tradeoff_zone']
        if len(tradeoff_pool) > 1:
            ranked = self.crowding_distance_fn(tradeoff_pool)
            explorer = random.choice(ranked[len(ranked) // 2:])  # 从拥挤度低的后半部分选
        elif tradeoff_pool:
            explorer = tradeoff_pool[0]
        else:
            all_pool = [w for front in fronts for w in front if w not in leaders.values()]
            explorer = random.choice(all_pool) if all_pool else None

        # === Step 4: 补全 leader 并返回 ===
        final_leaders = [leaders.get('spread_zone'), leaders.get('fair_zone'), leaders.get('tradeoff_zone'), explorer]

        # 若不足 4 个有效个体，补充已有或随机
        for i in range(4):
            if final_leaders[i] is None:
                pool = [w for front in fronts for w in front if w not in final_leaders and w is not None]
                final_leaders[i] = random.choice(pool) if pool else random.choice(
                    [l for l in final_leaders if l is not None])

        return final_leaders[0], final_leaders[1], final_leaders[2], final_leaders[3]

    def ensure_leader_minimum(self, archive, population, required_num=4):
        """
        保证 archive 中至少有 required_num 个个体用于引导 leader。
        若不足，从 population 中按传播性目标补足。
        """
        if len(archive) >= required_num:
            return  # 足够，无需补充

        # 按传播性（Cost[0]）降序从 population 中选出补充
        supplement = sorted(
            [w for w in population if w not in archive],
            key=lambda w: -w.Cost[0]
        )
        needed = required_num - len(archive)
        archive.extend(supplement[:needed])


# import random
# import copy
# from collections import defaultdict
#
# class LeaderManager:
#     """
#     基于区域偏好从多层Fronts中选择引导个体。
#     """
#     def __init__(self):
#         pass
#
#     def normalize_costs(self, wolves):
#         objs = list(zip(*[w.Cost[:2] for w in wolves]))
#         norm_objs = []
#         for i in range(2):
#             min_val = min(objs[i])
#             max_val = max(objs[i])
#             norm_objs.append([(v - min_val) / (max_val - min_val + 1e-9) for v in objs[i]])
#         return list(zip(norm_objs[0], norm_objs[1]))
#
#     def classify_region(self, norm_cost):
#         spread, fair = norm_cost
#         if spread >= fair and (spread + fair) >= 1:
#             return 'spread_zone'
#         elif fair >= spread and (spread + fair) >= 1:
#             return 'fair_zone'
#         else:
#             return 'middle_zone'
#
#     def select_leaders_by_region(self, fronts):
#         """
#         从多层Fronts中选出3个leaders（优先从高层选）。
#         """
#         leaders = []
#         for front in fronts:
#             if len(leaders) >= 3:
#                 break
#             norm_costs = self.normalize_costs(front)
#             region_map = defaultdict(list)
#
#             for wolf, norm in zip(front, norm_costs):
#                 region = self.classify_region(norm)
#                 region_map[region].append(wolf)
#
#             for key in ['spread_zone', 'fair_zone', 'middle_zone']:
#                 candidates = region_map.get(key, [])
#                 if not candidates:
#                     continue
#                 candidates.sort(key=lambda w: -w.Cost[0])
#                 selected = candidates[0]
#                 if selected not in leaders:
#                     leaders.append(selected)
#                 if len(leaders) >= 3:
#                     break
#
#         if len(leaders) < 3:
#             all_candidates = [wolf for front in fronts for wolf in front if wolf not in leaders]
#             all_candidates.sort(key=lambda w: -w.Cost[0])
#             leaders += all_candidates[:3 - len(leaders)]
#
#         return leaders[:3]
#
#     def ensure_leader_minimum(self, archive, population):
#         """
#         保证至少有3个个体可选leader。
#         """
#         if len(archive) < 3:
#             sorted_by_obj1 = sorted(population, key=lambda w: w.Cost[0], reverse=True)
#             extra = [w for w in sorted_by_obj1 if w not in archive][:3 - len(archive)]
#             archive.extend(extra)
'''
使用极角划分区域

从各区域选择最优个体作为 Leader1–3

从 trade-off 区域中选一个拥挤距离较低的个体作为 Leader4（探索引导）

通过外部传入的 calculate_crowding_distance 函数调用（解耦设计）

Front 1 -> spread_zone / fair_zone / middle_zone
↓
如果不足 -> Front 2 -> spread_zone / fair_zone / middle_zone
↓
如果还不足 -> Front 3 ...
↓
如果全部用完还不足 -> 传播值高的补足

'''