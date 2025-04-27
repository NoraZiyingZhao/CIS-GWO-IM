from GreyWolf import *
from ArchiveManager import *
from Evaluator import *
from PerturbationHandler import *
from StructureMetrics import *
from visualizer import *
from FMODGWO import *
import networkx as nx
# import Save_Visualize as sv
import pandas as pd
# from baseline import CELF
# from baseline import degree
# from baseline import RANDOM
# from baseline import pagerank
# from baseline import ClosenessCentr
# from baseline import betweennesscentr
# from baseline import eigenvectorcentr
# from MObaseline import MODBA
# from MObaseline import MODPSO
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
    pop_size = 100
    archive_size = 30
    max_iter = 100
    runs = 10

    # === Step 3: Precompute Metrics ===
    metrics = StructureMetrics(graph)  # 计算closeness, degree

    # === Step 4: Run Multiple Independent Runs ===
    all_runs_results1 = []
    all_runs_results2 = []
    all_solutions=[]

    for run_idx in range(runs):
        print(f'Run {run_idx + 1}/{runs}')

        # 初始化优化器
        optimizer = FMODGWO(graph, metrics, budget, pop_size, archive_size)

        # 执行优化
        archive, archive_costs_history, hv_values, times = optimizer.optimize(max_iter)

        # 提取每次运行的结果
        run_result1 = {
            "final_archive": archive,
            "archive_costs_history": archive_costs_history,
            "hv_values": hv_values,
            "times": times
        }
        all_runs_results1.extend(run_result1)

        # 初始化优化器
        modba = MODBA(graph, metrics, budget, pop_size, archive_size)

        # 执行优化
        archive, archive_costs_history, hv_values, times = modba.optimise(max_iter)

        # 提取每次运行的结果
        run_result2 = {
            "final_archive": archive,
            "archive_costs_history": archive_costs_history,
            "hv_values": hv_values,
            "times": times
        }
        all_runs_results2.extend(run_result2)

    # === Step 5: 后处理（可以保存/画图/输出Excel） ===
    visualizer = Visualizer('results/FMODGWO_results')
    visualizer.save_and_plot(all_solutions)

if __name__ == '__main__':
    main()