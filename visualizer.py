import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_and_plot_multiobj(self, all_runs_results, algo_name='Algorithm'):
        """
        保存数据并绘制图像（适用于多目标优化算法，如FMODGWO等）
        all_runs_results: List[Dict], 每个 run 的结果（带 final_archive, hv_values, times）
        algo_name: 算法名称（用于区分保存文件）
        """
        runs = len(all_runs_results)
        iterations = len(all_runs_results[0]['hv_values'])

        all_hv_curves = []
        all_times = []
        all_final_hv = []
        all_final_costs = []

        for run_idx, result in enumerate(all_runs_results):
            all_hv_curves.append(result['hv_values'])    # 每次 run 的 HV 曲线
            all_times.append(result['times'])            # 每次 run 的时间记录
            all_final_hv.append(result['hv_values'][-1]) # 每次 run 的最后一代HV

            # final_archive 是 [(fitness1, fitness2)] 的列表
            for cost in result['final_archive']:
                all_final_costs.append([run_idx + 1, cost[0], cost[1]])

        # === 保存 Excel 文件 ===
        hv_df = pd.DataFrame(all_hv_curves).T
        hv_df.to_excel(os.path.join(self.save_dir, f'{algo_name}_hv_per_iteration.xlsx'), index=False)

        times_df = pd.DataFrame(all_times).T
        times_df.to_excel(os.path.join(self.save_dir, f'{algo_name}_times_per_iteration.xlsx'), index=False)

        final_costs_df = pd.DataFrame(all_final_costs, columns=['Run', 'Fitness1_Spread', 'Fitness2_Fairness'])
        final_costs_df.to_excel(os.path.join(self.save_dir, f'{algo_name}_final_pareto_costs.xlsx'), index=False)

        final_hv_df = pd.DataFrame({'Run': np.arange(1, runs + 1), 'Final_HV': all_final_hv})
        final_hv_df.to_excel(os.path.join(self.save_dir, f'{algo_name}_final_hv_values.xlsx'), index=False)

        # === 绘制图 ===
        self.plot_hv_vs_iteration(hv_df, algo_name)
        self.plot_final_hv_boxplot(all_final_hv, algo_name)
        self.plot_final_pareto_scatter(final_costs_df, algo_name)

        print(f"✅ {algo_name} 多目标结果保存并绘制完成！\n")

    def save_and_plot_singleobj(self, all_runs_results, algo_name='Algorithm'):
        """
        保存数据并绘制图像（适用于单目标优化算法，如degree、random等）
        all_runs_results: List[Dict], 每个 run 的结果（仅有final_solutions）
        """
        runs = len(all_runs_results)
        all_final_spreads = []  # 这里只关心final_spread

        for run_idx, result in enumerate(all_runs_results):
            for sol in result['final_solutions']:
                spread_value = sol[1]  # (seed_set, spread_value, time)
                all_final_spreads.append(spread_value)

        # 保存 spread 到 Excel
        spread_df = pd.DataFrame({'Run': np.arange(1, len(all_final_spreads)+1), 'Spread': all_final_spreads})
        spread_df.to_excel(os.path.join(self.save_dir, f'{algo_name}_spread_values.xlsx'), index=False)

        # 绘制 spread boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot(all_final_spreads, labels=[algo_name])
        plt.ylabel('Spread')
        plt.title(f'Spread Distribution Across Runs ({algo_name})')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_spread_boxplot.png'), dpi=600)
        plt.close()

        print(f"✅ {algo_name} 单目标结果保存并绘制完成！\n")

    def plot_hv_vs_iteration(self, hv_df, algo_name):
        """绘制每次迭代的平均 HV 曲线"""
        mean_hv = np.mean(hv_df.values, axis=1)
        std_hv = np.std(hv_df.values, axis=1)

        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(mean_hv)+1), mean_hv, label='Mean HV', color='blue')
        plt.fill_between(range(1, len(mean_hv)+1), mean_hv - std_hv, mean_hv + std_hv,
                         color='blue', alpha=0.3, label='Std Deviation')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.title(f'{algo_name} HV vs Iteration (Mean ± Std)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_hv_vs_iteration.png'), dpi=600)
        plt.close()

    def plot_final_hv_boxplot(self, all_final_hv, algo_name):
        """绘制Final HV的boxplot"""
        plt.figure(figsize=(8, 6))
        plt.boxplot(all_final_hv, labels=[algo_name])
        plt.ylabel('Final Hypervolume')
        plt.title(f'{algo_name} Final HV Distribution')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_final_hv_boxplot.png'), dpi=600)
        plt.close()

    def plot_final_pareto_scatter(self, final_costs_df, algo_name):
        """绘制最后收敛的 Pareto 解的散点图"""
        plt.figure(figsize=(8, 6))
        plt.scatter(final_costs_df['Fitness1_Spread'], final_costs_df['Fitness2_Fairness'],
                    c='red', edgecolors='black')
        plt.xlabel('Fitness 1 (Spread)')
        plt.ylabel('Fitness 2 (Fairness)')
        plt.title(f'{algo_name} Final Pareto Front Scatter')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f'{algo_name}_final_pareto_scatter.png'), dpi=600)
        plt.close()
