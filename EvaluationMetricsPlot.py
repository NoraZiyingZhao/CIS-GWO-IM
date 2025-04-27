import os
import numpy as np
import matplotlib.pyplot as plt
from ArchiveManager import ArchiveManager

def generate_true_pareto_front(all_algorithms_solutions):
    """
    从所有算法的所有解集中生成全局真实Pareto Front。
    :param all_algorithms_solutions: List of ndarray，每个算法所有runs的解
    """
    combined = np.vstack(all_algorithms_solutions)

    manager = ArchiveManager(archive_size=1000000)  # 临时用来调用dominates，不真正用它管理
    pareto_mask = [not any(manager.dominates(other, sol) for other in combined) for sol in combined]
    return combined[pareto_mask]

def calculate_igd(pf, reference_pf):
    """
    计算单个Pareto Front相对于参考PF的IGD值。
    IGD越小表示越接近真实PF。
    """
    distances = []
    for ref_point in reference_pf:
        min_distance = np.min([np.linalg.norm(sol - ref_point) for sol in pf])
        distances.append(min_distance)
    return np.mean(distances)

def calculate_igd_per_run(run_solutions_list, reference_pf):
    """
    针对每个run分别计算IGD。
    :param run_solutions_list: List[ndarray]，每个run的一组解
    """
    igd_values = []
    for pf in run_solutions_list:
        igd = calculate_igd(pf, reference_pf)
        igd_values.append(igd)
    return igd_values

def plot_igd_per_run(igd_data, output_dir, filename):
    """
    绘制每个算法在每次Run的IGD对比曲线。
    :param igd_data: dict {algorithm_name: igd_list}
    """
    plt.figure(figsize=(10, 8))

    for algo_name, igd_values in igd_data.items():
        plt.plot(range(1, len(igd_values)+1), igd_values, label=algo_name, linewidth=2)

    plt.xlabel("Run")
    plt.ylabel("IGD")
    plt.title("IGD per Run Comparison")
    plt.legend()
    plt.grid()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
    plt.savefig(os.path.join(output_dir, filename + ".eps"), dpi=600)
    plt.close()

# （可选）如果以后需要画 per iteration 的 IGD 曲线，可以解开这个函数
# def plot_igd_per_iteration(igd_data, output_dir, filename):
#     """
#     绘制每次迭代的IGD变化。
#     :param igd_data: dict {algorithm_name: igd_list}
#     """
#     plt.figure(figsize=(10, 8))
#
#     for algo_name, igd_values in igd_data.items():
#         plt.plot(range(1, len(igd_values)+1), igd_values, label=algo_name, linewidth=2)
#
#     plt.xlabel("Iteration")
#     plt.ylabel("IGD")
#     plt.title("IGD per Iteration Comparison")
#     plt.legend()
#     plt.grid()
#
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
#     plt.savefig(os.path.join(output_dir, filename + ".eps"), dpi=600)
#     plt.close()

def plot_igd_boxplot(igd_data, output_dir, filename):
    """
    绘制不同算法在所有Run上的IGD分布（Boxplot图）。
    :param igd_data: dict {algorithm_name: igd_list}
    """
    plt.figure(figsize=(8, 6))

    data = [igd_values for igd_values in igd_data.values()]
    labels = list(igd_data.keys())

    plt.boxplot(data, labels=labels)
    plt.ylabel('IGD')
    plt.title('IGD Distribution Across Algorithms')
    plt.grid()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=600)
    plt.savefig(os.path.join(output_dir, filename + ".eps"), dpi=600)
    plt.close()
