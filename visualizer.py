import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_and_plot(self, all_runs_results):
        """
        保存关键数据并绘制图像。
        输入：
            all_runs_results = [
                {
                    'final_archive': [...],
                    'archive_costs_history': [...],
                    'hv_values': [...],
                    'times': [...]
                },
                {...}, {...}
            ]
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

            # 注意：这里result['final_archive']是完整的狼个体，所以要取Cost
            for wolf in result['final_archive']:
                all_final_costs.append([run_idx + 1, wolf[0], wolf[1]])  # Fitness1, Fitness2

        # === 保存 Excel 文件 ===
        hv_df = pd.DataFrame(all_hv_curves).T
        hv_df.to_excel(os.path.join(self.save_dir, 'hv_per_iteration.xlsx'), index=False)

        times_df = pd.DataFrame(all_times).T
        times_df.to_excel(os.path.join(self.save_dir, 'times_per_iteration.xlsx'), index=False)

        final_costs_df = pd.DataFrame(all_final_costs, columns=['Run', 'Fitness1_Spread', 'Fitness2_Fairness'])
        final_costs_df.to_excel(os.path.join(self.save_dir, 'final_pareto_costs.xlsx'), index=False)

        final_hv_df = pd.DataFrame({'Run': np.arange(1, runs + 1), 'Final_HV': all_final_hv})
        final_hv_df.to_excel(os.path.join(self.save_dir, 'final_hv_values.xlsx'), index=False)

        # === 绘制图 ===
        self.plot_hv_vs_iteration(hv_df)
        self.plot_final_hv_boxplot(all_final_hv)
        self.plot_final_pareto_scatter(final_costs_df)

        print(f"✅ All results saved and plots generated under {self.save_dir}")

    def plot_hv_vs_iteration(self, hv_df):
        """绘制每次迭代的平均 HV 曲线"""
        mean_hv = np.mean(hv_df.values, axis=1)
        std_hv = np.std(hv_df.values, axis=1)

        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(mean_hv)+1), mean_hv, label='Mean HV', color='blue')
        plt.fill_between(range(1, len(mean_hv)+1), mean_hv - std_hv, mean_hv + std_hv,
                         color='blue', alpha=0.3, label='Std Deviation')
        plt.xlabel('Iteration')
        plt.ylabel('Hypervolume')
        plt.title('HV vs Iteration (Mean ± Std)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, 'hv_vs_iteration.png'), dpi=600)
        plt.close()

    def plot_final_hv_boxplot(self, all_final_hv):
        """绘制Final HV的boxplot"""
        plt.figure(figsize=(8, 6))
        plt.boxplot(all_final_hv, labels=['FMODGWO'])
        plt.ylabel('Final Hypervolume')
        plt.title('Final HV Distribution Across Runs')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, 'final_hv_boxplot.png'), dpi=600)
        plt.close()

    def plot_final_pareto_scatter(self, final_costs_df):
        """绘制最后收敛的 Pareto 解的散点图"""
        plt.figure(figsize=(8, 6))
        plt.scatter(final_costs_df['Fitness1_Spread'], final_costs_df['Fitness2_Fairness'],
                    c='red', edgecolors='black')
        plt.xlabel('Fitness 1 (Spread)')
        plt.ylabel('Fitness 2 (Fairness)')
        plt.title('Final Pareto Front Scatter Plot')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, 'final_pareto_scatter.png'), dpi=600)
        plt.close()
