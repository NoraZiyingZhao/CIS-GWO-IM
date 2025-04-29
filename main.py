import networkx as nx
import pandas as pd
import os
import EvaluationMetricsPlot
from GreyWolf import *
from ArchiveManager import *
from Evaluator import *
from PerturbationHandler import *
from StructureMetrics import *
# from FMODGWO import *
from StratifiedFMODGWO import *
# from MObaseline import MODBA, MODPSO  # 多目标基线
# from MObaseline.GFMOGWOpackage import GFMOGWO
# from baseline import CELF, degree, RANDOM, pagerank, ClosenessCentr, betweennesscentr, eigenvectorcentr  # 单目标基线
from visualizer import Visualizer

def extract_fitness(archive):
    """提取种群中所有解的Cost（目标值）"""
    return [cost for cost in archive]

def main():
    # === Step 1: Load Dataset ===
    # edges_file = 'datasets/fb-pages-food/community-info-food/mapped_nodes_edges.txt'
    # edges_file = 'datasets/soc-hamsterster/community-info-ham/mapped_nodes_edges.txt'
    edges_file = 'datasets/email/communityinfo-email/mapped_nodes_edges.txt'
    # edges_file = 'datasets/facebook/communityinfo-facebook/mapped_nodes_edges.txt'
    # edges_file = 'datasets/wiki/communityinfo-wiki/mapped_nodes_edges.txt'
    # edges_file = 'datasets/socfb-Rice31/community-info-rice/mapped_nodes_edges.txt'

    graph = nx.read_edgelist(edges_file, nodetype=int)

    # === Step 2: Algorithm Parameters ===
    budget_ratio = 0.01  # 1%
    budget = max(5, int(graph.number_of_nodes() * budget_ratio))
    print("Budget =", budget)
    pop_size = 30
    archive_size = 10
    max_iter = 10
    runs = 2

    # === Step 3: Precompute Structural Metrics ===
    metrics = StructureMetrics(graph)

    # === Step 4: Create Directory for Results ===
    save_dir_multi = 'results/MultiObjBaselines'  # 多目标算法保存
    save_dir_single = 'results/SingleObjBaselines'  # 单目标算法保存
    os.makedirs(save_dir_multi, exist_ok=True)
    os.makedirs(save_dir_single, exist_ok=True)

    # === Step 5: Initialize Results Containers ===
    all_runs_fmodgwo = []
    all_runs_StratifiedFMODGWO=[]
    all_runs_modba = []
    all_runs_modpso = []
    all_runs_gfmogwo=[]

    all_runs_celf = []
    all_runs_degree = []
    all_runs_random = []
    all_runs_pagerank = []
    all_runs_closeness = []
    all_runs_betweenness = []
    all_runs_eigenvector = []

    # === Step 6: Execute Multiple Runs ===
    for run_idx in range(runs):
        print(f'Run {run_idx + 1}/{runs}')


        optimizer = StratifiedFMODGWO(graph, metrics, budget, pop_size, archive_size)
        archive, archive_costs_history, hv_values, times = optimizer.optimize(max_iter)
        run_result_StratifiedFMODGWO = {
            "final_archive": archive,
            "archive_costs_history": archive_costs_history,
            "hv_values": hv_values,
            "times": times
        }
        all_runs_StratifiedFMODGWO.append(run_result_StratifiedFMODGWO)
        print("solution of StratifiedFMODGWO: (all_runs_StratifiedFMODGWO)", all_runs_StratifiedFMODGWO)
        # --- 多目标算法1: FMODGWO ---
        # optimizer = FMODGWO(graph, metrics, budget, pop_size, archive_size)
        # archive, archive_costs_history, hv_values, times = optimizer.optimize(max_iter)
        # run_result_fmodgwo = {
        #     "final_archive": archive,
        #     "archive_costs_history": archive_costs_history,
        #     "hv_values": hv_values,
        #     "times": times
        # }
        # all_runs_fmodgwo.append(run_result_fmodgwo)

        # # --- 多目标算法2: MODBA ---
        # optimizer_modba = MODBA(graph, metrics, budget, pop_size, archive_size)
        # archive, archive_costs_history, hv_values, times = optimizer_modba.optimize(max_iter)
        # run_result_modba = {
        #     "final_archive": archive,
        #     "archive_costs_history": archive_costs_history,
        #     "hv_values": hv_values,
        #     "times": times
        # }
        # all_runs_modba.append(run_result_modba)
        #
        # # --- 多目标算法3: MODPSO ---
        # optimizer_modpso = MODPSO(graph, metrics, budget, pop_size, archive_size)
        # archive, archive_costs_history, hv_values, times = optimizer_modpso.optimize(max_iter)
        # run_result_modpso = {
        #     "final_archive": archive,
        #     "archive_costs_history": archive_costs_history,
        #     "hv_values": hv_values,
        #     "times": times
        # }
        # all_runs_modpso.append(run_result_modpso)

        # # --- 单目标基线: CELF ---
        # celf_solutions = CELF.CELF_seed_selection(graph, budget)
        # all_runs_celf.append(celf_solutions)
        #
        # # --- 单目标基线: Degree ---
        # degree_solutions = degree.degree_seed_selection(graph, budget)
        # all_runs_degree.append(degree_solutions)
        #
        # # --- 单目标基线: Random ---
        # random_solutions = RANDOM.random_seed_selection(graph, budget)
        # all_runs_random.append(random_solutions)
        #
        # # --- 单目标基线: PageRank ---
        # pagerank_solutions = pagerank.pagerank_seed_selection(graph, budget)
        # all_runs_pagerank.append(pagerank_solutions)
        #
        # # --- 单目标基线: Closeness Centrality ---
        # closeness_solutions = ClosenessCentr.closeness_seed_selection(graph, budget)
        # all_runs_closeness.append(closeness_solutions)
        #
        # # --- 单目标基线: Betweenness Centrality ---
        # betweenness_solutions = betweennesscentr.betweenness_seed_selection(graph, budget)
        # all_runs_betweenness.append(betweenness_solutions)
        #
        # # --- 单目标基线: Eigenvector Centrality ---
        # eigenvector_solutions = eigenvectorcentr.eigenvector_seed_selection(graph, budget)
        # all_runs_eigenvector.append(eigenvector_solutions)

    # === Step 7: Save and Plot for Multi-objective Algorithms ===
    visualizer_multi = Visualizer(save_dir_multi)

    # 保存 FMODGWO 多目标结果
    visualizer_multi.save_and_plot_multiobj(all_runs_StratifiedFMODGWO, algo_name='StratifiedFMODGWO')

    # 保存 MODBA 多目标结果
    # visualizer_multi.save_and_plot_multiobj(all_runs_modba, algo_name='MODBA')

    # 保存 MODPSO 多目标结果
    # visualizer_multi.save_and_plot_multiobj(all_runs_modpso, algo_name='MODPSO')

    # 保存 GFMOGWO 多目标结果
    # visualizer_multi.save_and_plot_multiobj(all_runs_gfmogwo, algo_name='GFMOGWO')

    # === Step 8: Save and Plot for Single-objective Baselines (Optional，后续可以打开) ===
    # 单目标算法，只保存最终 seed selection 和 spread
    # visualizer_single = Visualizer(save_dir_single)

    # 保存 CELF baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_celf, algo_name='CELF')

    # 保存 Degree baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_degree, algo_name='Degree')

    # 保存 Random baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_random, algo_name='Random')

    # 保存 PageRank baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_pagerank, algo_name='PageRank')

    # 保存 Closeness Centrality baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_closeness, algo_name='Closeness')

    # 保存 Betweenness Centrality baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_betweenness, algo_name='Betweenness')

    # 保存 Eigenvector Centrality baseline 单目标结果
    # visualizer_single.save_and_plot_singleobj(all_runs_eigenvector, algo_name='Eigenvector')

    print("✅ All experiments finished and results saved.")

    # === Step 9: Draw IGD Plots ===
    # 注意！！下面正式生成 IGD 分析
    # all_algorithms_solutions = []
    #
    # # 收集所有算法最终的 Pareto 解（比如 fmodgwo，modba，modpso，gfmogwo）
    # for result in all_runs_fmodgwo:
    #     solutions = np.array([[sol[0], sol[1]] for sol in result["final_archive"]])
    #     all_algorithms_solutions.append(solutions)
    #
    # # 后续其他算法加进去
    # # for result in all_runs_modba: ...
    #
    # true_pareto = EvaluationMetricsPlot.generate_true_pareto_front(all_algorithms_solutions)
    #
    # # 计算每个算法每个run的IGD
    # igd_results = {}
    #
    # igd_per_run_fmodgwo = []
    # for result in all_runs_fmodgwo:
    #     solutions = np.array([[sol[0], sol[1]] for sol in result["final_archive"]])
    #     igd = EvaluationMetricsPlot.calculate_igd(solutions, true_pareto)
    #     igd_per_run_fmodgwo.append(igd)
    # igd_results['FMODGWO'] = igd_per_run_fmodgwo
    #
    # # 后续可以继续加
    # # igd_results['MODBA'] = ...
    # # igd_results['MODPSO'] = ...
    # # igd_results['GFMOGWO'] = ...
    #
    # # 绘制IGD图
    # EvaluationMetricsPlot.plot_igd_per_run(igd_results, save_dir_multi, 'igd_comparison_per_run')
    # EvaluationMetricsPlot.plot_igd_boxplot(igd_results, save_dir_multi, 'igd_comparison_boxplot')
    #
    # print("✅ All experiments finished and results saved.")


'''
多目标算法	HV per iteration, final Pareto solutions, final HV boxplot, HV mean±std 曲线, Pareto scatter图
单目标算法	保存每次 run 的 seed set, spread 值，运行时间（绘简单柱状图或者统计即可）
'''

if __name__ == '__main__':
    main()
