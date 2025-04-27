import numpy as np
import matplotlib.pyplot as plt
import os
from ArchiveManager import ArchiveManager

def generate_true_pareto_front(all_algorithms_solutions):
    """
    从所有算法的所有解集中生成全局真实Pareto Front。
    """
    combined = np.vstack(all_algorithms_solutions)

    manager = ArchiveManager(archive_size=1000000)  # 临时用来调用dominates，不真正用它管理
    pareto_mask = [not any(manager.dominates(other, sol) for other in combined) for sol in combined]
    return combined[pareto_mask]

#计算一个给定的Pareto前沿 pf 相对于参考Pareto前沿 reference_pf 的IGD指标（Inverted Generational Distance）。
# IGD越小，说明算法找到的解越靠近真实Pareto前沿，整体性能越好。
def calculate_igd(pf, reference_pf):
    """
    计算单个Pareto Front相对于真实Pareto Front的IGD值。
    """
    distances = []
    for ref_point in reference_pf:
        min_distance = np.min([np.linalg.norm(sol - ref_point) for sol in pf])
        distances.append(min_distance)
    return np.mean(distances)

# 针对每一个运行（run），分别计算其Pareto Front相对于参考Pareto Front的IGD值。
def calculate_igd_per_run(run_solutions_list, reference_pf):
    """
    计算每个run的IGD值列表。
    :param run_solutions_list: List of ndarray，每个run的一组解
    """
    igd_values = []
    for pf in run_solutions_list:
        igd = calculate_igd(pf, reference_pf)
        igd_values.append(igd)
    return igd_values

# 绘制每次Run对应的IGD曲线图，并保存为PNG/EPS格式。
def plot_igd_per_run(igd_data, output_dir, filename):
    """
    绘制每个run对应的IGD变化。
    :param igd_data: dict {algorithm_name: igd_list}
    """
    plt.figure(figsize=(10, 8))

    for algo_name, igd_values in igd_data.items():
        plt.plot(range(1, len(igd_values) + 1), igd_values, label=algo_name, linewidth=2)

    plt.xlabel("Run")
    plt.ylabel("IGD")
    plt.title("IGD per Run")
    plt.legend()
    plt.grid()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
    plt.savefig(os.path.join(output_dir, filename + ".eps"), dpi=600)
    plt.close()

# def plot_igd_per_iteration(igd_data, output_dir, filename):
#     """
#     绘制每一代/每一次迭代的IGD变化。
#     :param igd_data: dict {algorithm_name: igd_list}
#     """
#     plt.figure(figsize=(10, 8))
#
#     for algo_name, igd_values in igd_data.items():
#         plt.plot(range(1, len(igd_values) + 1), igd_values, label=algo_name, linewidth=2)
#
#     plt.xlabel("Iteration")
#     plt.ylabel("IGD")
#     plt.title("IGD per Iteration")
#     plt.legend()
#     plt.grid()
#
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
#     plt.savefig(os.path.join(output_dir, filename + ".eps"), dpi=600)
#     plt.close()
