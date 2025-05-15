#only for existing MOBaseline use.


import random
#这部分包含：帕累托支配比较,拥挤距离，筛选出种群中的帕累托最优解和精英继承解。

def dominates(fitness1, fitness2):
    """
    比较两个解的适应度，基于多目标的严格帕累托支配关系。
    如果 fitness1 和 fitness2 的长度相同，且 fitness1 在所有目标上都不劣于 fitness2，
    且至少在一个目标上优于 fitness2，则认为 fitness1 支配 fitness2。
    支持单目标和多目标的情况。
    """
    # 检查输入是否为单目标情况
    if isinstance(fitness1, (int, float)) and isinstance(fitness2, (int, float)):
        return fitness1 > fitness2  # 对于单个数值，直接比较即可

    # 检查输入是否合法（多目标情况）
    if not (isinstance(fitness1, tuple) and isinstance(fitness2, tuple)):
        raise ValueError("Fitness values must be tuples of numeric types.")
    if len(fitness1) != len(fitness2):
        raise ValueError("Both fitness values must have the same number of elements.")
    if not all(isinstance(f, (int, float)) for f in fitness1 + fitness2):
        raise ValueError("Fitness values must be numeric.")

    better_in_all = True  # 假设在所有目标上都不劣于
    better_in_one = False  # 是否至少在一个目标上优于

    # 对多目标情况进行比较
    for f1, f2 in zip(fitness1, fitness2):
        if f1 < f2:  # 对于最大化问题，fitness1 应该不小于 fitness2
            better_in_all = False  # 如果某个目标上更差，则无法支配
        if f1 > f2:  # 至少有一个目标上更好
            better_in_one = True

    # 只有在所有目标上不劣于，并且至少一个目标上优于时，才认为支配
    return better_in_all and better_in_one


def update_elite_archive(elite_archive, new_solutions, elite_size):
    """
    更新精英存档，通过保留基于拥挤距离的最佳解。
    确保精英存档不会超过定义的最大精英大小。
    """
    # Step 1: 批量添加新解并更新精英存档
    for new_solution, new_fitness in new_solutions:
        # 标记当前解是否应加入精英存档
        to_add = True

        # 检查新解是否支配现有存档中的解
        elite_archive = [
            (existing_solution, existing_fitness)
            for existing_solution, existing_fitness in elite_archive
            if not dominates(new_fitness, existing_fitness)
        ]

        # 检查新解是否被现有解支配
        for _, existing_fitness in elite_archive:
            if dominates(existing_fitness, new_fitness):
                to_add = False
                break

        if to_add:
            elite_archive.append((new_solution, new_fitness))

    # Step 2: 如果存档大小超过阈值，基于 crowding distance 策略进行裁剪
    if len(elite_archive) > elite_size:
        elite_archive = calculate_crowding_distance(elite_archive)
        elite_archive = elite_archive[:elite_size]  # 保留基于拥挤距离的最佳解

    return elite_archive


"""
归一化处理:

在计算拥挤距离时，对每个目标的差值进行归一化处理 (archive[i + 1][1][obj_index] - archive[i - 1][1][obj_index]) / (max_obj - min_obj)。
这可以确保每个目标的变化范围得到公平的考虑，避免某个目标的绝对差值过大而主导拥挤距离计算的问题。
累计拥挤距离:

每个解的最终拥挤距离是所有目标上的距离之和，这样可以衡量解在所有目标上的分布情况。
将边界解设置为无穷大，以确保这些解不容易被裁剪掉。
排序逻辑:

最终根据累积的拥挤距离从大到小排序，确保更分散（多样性更好）的解在前面。
"""

def calculate_crowding_distance(archive):
    """
    计算 archive 解决方案的拥挤距离，以确保多样性。
    :param archive: [(solution, fitness)] 列表，每个 solution 有多个目标
    :return: 根据拥挤距离排序的 archive 列表 (从大到小)
    """
    num_solutions = len(archive)
    if num_solutions == 0:
        return archive

    # 初始化每个解的拥挤距离
    distances = [0.0] * num_solutions
    num_objectives = len(archive[0][1])  # 假设每个 solution 都有多个目标

    # 针对每一个目标进行排序和计算
    for obj_index in range(num_objectives):
        # 根据当前目标对 archive 进行排序
        archive.sort(key=lambda x: x[1][obj_index])

        # 边界解的拥挤距离设为无穷大
        distances[0] = float('inf')
        distances[-1] = float('inf')

        # 对中间的解计算拥挤距离
        min_obj = archive[0][1][obj_index]
        max_obj = archive[-1][1][obj_index]

        # 避免除以零的问题，如果 min_obj == max_obj 说明所有解在这个目标上都相同
        if max_obj - min_obj == 0:
            continue

        for i in range(1, num_solutions - 1):
            # 归一化拥挤距离计算
            distance = (archive[i + 1][1][obj_index] - archive[i - 1][1][obj_index]) / (max_obj - min_obj)
            distances[i] += distance

    # 根据累计的拥挤距离排序，从大到小
    sorted_archive = [x for _, x in sorted(zip(distances, archive), key=lambda x: x[0], reverse=True)]

    return sorted_archive

def calculate_hypervolume(archive, reference_point):
    """
    Accurate hypervolume calculation for two-objective maximization problems.
    """
    # Extract fitnesses and sort by the first objective in descending order
    fitnesses = sorted([fit for _, fit in archive], key=lambda x: x[0], reverse=True)
    hv = 0.0
    prev_f2 = reference_point[1]

    for f1, f2 in fitnesses:
        width = max(f1 - reference_point[0], 0)
        height = max(f2 - prev_f2, 0)
        hv += width * height
        prev_f2 = f2

    return hv

