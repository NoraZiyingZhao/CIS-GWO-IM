import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hv_vs_iteration_comparison(algorithms, save_dir, output_name="hv_vs_iteration_comparison"):
    """
    绘制不同算法的 HV vs Iteration 平均曲线对比图
    """
    plt.figure(figsize=(10, 7))

    for algo_name, algo_path in algorithms.items():
        hv_df = pd.read_excel(os.path.join(algo_path, 'hv_per_iteration.xlsx'))
        mean_hv = np.mean(hv_df.values, axis=1)
        std_hv = np.std(hv_df.values, axis=1)

        iterations = range(1, len(mean_hv)+1)

        plt.plot(iterations, mean_hv, label=f"{algo_name} Mean HV")
        plt.fill_between(iterations, mean_hv - std_hv, mean_hv + std_hv, alpha=0.2)

    plt.xlabel('Iteration')
    plt.ylabel('Hypervolume')
    plt.title('HV vs Iteration Comparison')
    plt.legend()
    plt.grid()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{output_name}.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, f"{output_name}.eps"), dpi=600)
    plt.close()

def plot_final_hv_boxplot_comparison(algorithms, save_dir, output_name="final_hv_boxplot_comparison"):
    """
    绘制不同算法的 Final HV Boxplot 对比图
    """
    all_final_hvs = []
    labels = []

    for algo_name, algo_path in algorithms.items():
        hv_df = pd.read_excel(os.path.join(algo_path, 'final_hv_values.xlsx'))
        all_final_hvs.append(hv_df['Final_HV'].values)
        labels.append(algo_name)

    plt.figure(figsize=(10, 6))
    plt.boxplot(all_final_hvs, labels=labels)
    plt.ylabel('Final Hypervolume')
    plt.title('Final HV Distribution Comparison')
    plt.grid()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{output_name}.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, f"{output_name}.eps"), dpi=600)
    plt.close()

if __name__ == "__main__":
    # ==== 设置不同算法保存的结果路径 ====
    algorithms = {
        "FMODGWO": "results/FMODGWO_results",
        "MODBA": "results/MODBA_results",
        "MODPSO": "results/MODPSO_results",
        "GFMOGWO": "results/GFMOGWO_results",
    }

    # 总体保存路径
    save_dir = "results/ComparisonPlots"

    # ==== 调用绘图函数 ====
    plot_hv_vs_iteration_comparison(algorithms, save_dir)
    plot_final_hv_boxplot_comparison(algorithms, save_dir)

    print(f"✅ Comparison plots saved under {save_dir}")
