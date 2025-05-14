import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_and_plot_multiobj(self, all_runs_results, algo_name='Algorithm', dataset_name='dataset'):
        """
        保存数据并绘制图像（适用于多目标优化算法，如FMODGWO）
        all_runs_results: List[Dict], 每个 run 的结果（带 final_archive, hv_values, times）
        algo_name: 算法名称（用于区分保存文件）
        """
        runs = len(all_runs_results)
        iterations = len(all_runs_results[0]['hv_values'])

        all_hv_curves = []
        all_times = []
        all_final_hv = []
        all_final_costs = []
        all_solutions_detailed = []

        for run_idx, result in enumerate(all_runs_results):
            hv_values = result['hv_values']
            times = result['times']
            archive = result['final_archive']

            all_hv_curves.append(hv_values)
            all_times.append(times)

            final_hv = hv_values[-1]  # ✅ Use precomputed HV from final archive
            all_final_hv.append(final_hv)

            for sol_idx, wolf in enumerate(archive):
                cost = getattr(wolf, 'Cost', [None, None])
                seed_set = getattr(wolf, 'Position', None)
                all_final_costs.append([run_idx + 1, cost[0], cost[1]])
                all_solutions_detailed.append({
                    'Run': run_idx + 1,
                    'Solution_Index': sol_idx + 1,
                    'Fitness1_Spread': cost[0],
                    'Fitness2_Fairness': cost[1],
                    'Seed_Set': str(seed_set),
                    'Time': times[-1] if times else None,
                    'Final_HV': final_hv
                })

            # Build file base
        file_prefix = f"{algo_name}_{dataset_name}"

        # === 保存 Excel 文件 ===
        hv_df = pd.DataFrame(all_hv_curves).T
        hv_df.insert(0, 'Iteration', np.arange(hv_df.shape[0]))  # 添加Iteration列
        hv_df.to_excel(os.path.join(self.save_dir, f'{file_prefix}_hv_per_iteration.xlsx'), index=False)

        times_df = pd.DataFrame(all_times).T
        times_df.insert(0, 'Iteration', np.arange(times_df.shape[0]))  # 添加Iteration列
        times_df.to_excel(os.path.join(self.save_dir, f'{file_prefix}_times_per_iteration.xlsx'), index=False)

        final_costs_df = pd.DataFrame(all_final_costs, columns=['Run', 'Fitness1_Spread', 'Fitness2_Fairness'])
        # final_costs_df.to_excel(os.path.join(self.save_dir, f'{file_prefix}_final_pareto_costs.xlsx'),
        #                         index=False)

        final_hv_df = pd.DataFrame({
            'Run': np.arange(1, len(all_final_hv) + 1),
            'Final_HV': all_final_hv
        })
        #
        # final_hv_df.to_excel(os.path.join(self.save_dir, f'{file_prefix}_final_hv_values.xlsx'), index=False)

        detailed_df = pd.DataFrame(all_solutions_detailed)
        detailed_df.to_excel(os.path.join(self.save_dir, f'{file_prefix}_final_pareto_solutions_detailed.xlsx'),
                             index=False)

        # === 绘制图像 ===
        # Plot figures with dataset-aware names
        self.plot_hv_vs_iteration(hv_df, algo_name, dataset_name)
        self.plot_final_hv_boxplot(all_final_hv, algo_name, dataset_name)
        self.plot_final_pareto_scatter(final_costs_df, algo_name, dataset_name)

        print(f"✅ {algo_name} on {dataset_name}: results saved and plotted!\n")

    def plot_hv_vs_iteration(self, hv_df, algo_name, dataset_name):
        mean_hv = np.mean(hv_df.values, axis=1)
        std_hv = np.std(hv_df.values, axis=1)

        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(mean_hv) + 1), mean_hv, label='Mean HV', color='blue')
        plt.fill_between(range(1, len(mean_hv) + 1), mean_hv - std_hv, mean_hv + std_hv, color='blue', alpha=0.3,
                         label='Std Deviation')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.title(f'{algo_name} HV vs Iteration ({dataset_name})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_{dataset_name}_hv_vs_iteration.png'), dpi=600)
        plt.close()

    def plot_final_hv_boxplot(self, all_final_hv, algo_name, dataset_name):
        plt.figure(figsize=(8, 6))
        plt.boxplot(all_final_hv, labels=[algo_name])
        plt.ylabel('Final Hypervolume')
        plt.title(f'{algo_name} Final HV Distribution ({dataset_name})')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_{dataset_name}_final_hv_boxplot.png'), dpi=600)
        plt.close()

    def plot_final_pareto_scatter(self, final_costs_df, algo_name, dataset_name):
        plt.figure(figsize=(8, 6))
        plt.scatter(final_costs_df['Fitness1_Spread'], final_costs_df['Fitness2_Fairness'], c='red', edgecolors='black')
        plt.xlabel('Fitness 1 (Spread)')
        plt.ylabel('Fitness 2 (Fairness)')
        plt.title(f'{algo_name} Final Pareto Front ({dataset_name})')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_{dataset_name}_final_pareto_scatter.png'), dpi=600)
        plt.close()

import matplotlib.pyplot as plt
import os

def plot_combined_pareto_front(multiobj_results, singleobj_results, save_dir, dataset_name, filename):
    """
    绘制包含多目标与单目标方法的 Pareto 前沿图，使用唯一 (颜色, marker) 组合区分每个算法。
    Draws a combined Pareto Front figure with multi-objective and single-objective solutions.

    :param multiobj_results: Dict {algorithm_name: list of run dicts} for multi-objective methods
    :param singleobj_results: Dict {algorithm_name: list of runs}, each run is [(seed_set, (f1, f2), time)]
    :param save_dir: Directory to save the figure
    :param dataset_name: Name of dataset (used in title and filename)
    :param filename: Name of the saved plot file
    """

    plt.figure(figsize=(10, 7))

    # 定义 marker 和颜色组合（循环使用也不重复）
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    colors = ['tab:purple', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:gray', 'tab:cyan']

    # 构造算法 → 样式映射表
    algo_names = list(multiobj_results.keys()) + list(singleobj_results.keys())
    style_map = {
        algo: {'marker': marker_styles[i % len(marker_styles)], 'color': colors[i % len(colors)]}
        for i, algo in enumerate(algo_names)
    }

    # === Multi-objective Algorithms ===
    for algo_name, runs in multiobj_results.items():
        style = style_map[algo_name]
        for run in runs:
            archive = run['final_archive']
            for sol in archive:
                cost = getattr(sol, 'Cost', [None, None])
                plt.scatter(cost[0], cost[1], label=algo_name,
                            marker=style['marker'], color=style['color'], alpha=0.8)

    # === Single-objective Algorithms ===
    for algo_name, runs in singleobj_results.items():
        style = style_map[algo_name]
        for run in runs:
            for sol in run:
                f1, f2 = sol[1]
                plt.scatter(f1, f2, label=algo_name,
                            marker=style['marker'], color=style['color'], alpha=0.8)

    # === 去除重复标签 ===
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title(f"Combined Pareto Front — {dataset_name}")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=600)
    plt.close()

    print(f"✅ Combined PF figure saved to: {os.path.join(save_dir, filename)}")

