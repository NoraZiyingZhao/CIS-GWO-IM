import random
import copy
from collections import defaultdict

class LeaderManager:
    """
    基于区域偏好从多层Fronts中选择引导个体。
    """
    def __init__(self):
        pass

    def normalize_costs(self, wolves):
        objs = list(zip(*[w.Cost[:2] for w in wolves]))
        norm_objs = []
        for i in range(2):
            min_val = min(objs[i])
            max_val = max(objs[i])
            norm_objs.append([(v - min_val) / (max_val - min_val + 1e-9) for v in objs[i]])
        return list(zip(norm_objs[0], norm_objs[1]))

    def classify_region(self, norm_cost):
        spread, fair = norm_cost
        if spread >= fair and (spread + fair) >= 1:
            return 'spread_zone'
        elif fair >= spread and (spread + fair) >= 1:
            return 'fair_zone'
        else:
            return 'middle_zone'

    def select_leaders_by_region(self, fronts):
        """
        从多层Fronts中选出3个leaders（优先从高层选）。
        """
        leaders = []
        for front in fronts:
            if len(leaders) >= 3:
                break
            norm_costs = self.normalize_costs(front)
            region_map = defaultdict(list)

            for wolf, norm in zip(front, norm_costs):
                region = self.classify_region(norm)
                region_map[region].append(wolf)

            for key in ['spread_zone', 'fair_zone', 'middle_zone']:
                candidates = region_map.get(key, [])
                if not candidates:
                    continue
                candidates.sort(key=lambda w: -w.Cost[0])
                selected = candidates[0]
                if selected not in leaders:
                    leaders.append(selected)
                if len(leaders) >= 3:
                    break

        if len(leaders) < 3:
            all_candidates = [wolf for front in fronts for wolf in front if wolf not in leaders]
            all_candidates.sort(key=lambda w: -w.Cost[0])
            leaders += all_candidates[:3 - len(leaders)]

        return leaders[:3]

    def ensure_leader_minimum(self, archive, population):
        """
        保证至少有3个个体可选leader。
        """
        if len(archive) < 3:
            sorted_by_obj1 = sorted(population, key=lambda w: w.Cost[0], reverse=True)
            extra = [w for w in sorted_by_obj1 if w not in archive][:3 - len(archive)]
            archive.extend(extra)
'''
Front 1 -> spread_zone / fair_zone / middle_zone
↓
如果不足 -> Front 2 -> spread_zone / fair_zone / middle_zone
↓
如果还不足 -> Front 3 ...
↓
如果全部用完还不足 -> 传播值高的补足

'''