import os
import matplotlib.pyplot as plt
import pandas as pd

def save_single_objective_results_to_excel(results_dict, save_dir, network_name, filename_prefix='SingleObjResults'):
    """
    Save the seed set, fitness values, and time of single-objective algorithms to an Excel file.

    :param results_dict: Dictionary mapping algorithm names to list of run results.
                         Each run result is a list of tuples: (seed_set, (f1, f2), time)
    :param save_dir: Directory to save the Excel file.
    :param network_name: Network name to use in the filename.
    :param filename_prefix: Filename prefix (without extension).
    """
    rows = []
    for algo_name, runs in results_dict.items():
        for run in runs:
            for seed_set, fitness, elapsed_time in run:
                rows.append({
                    'Algorithm': algo_name,
                    'SeedSet': sorted(seed_set),  # convert to sorted list for readability
                    'Fitness1': fitness[0],
                    'Fitness2': fitness[1],
                    'Time': elapsed_time
                })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{filename_prefix}_{network_name}.xlsx"
    output_path = os.path.join(save_dir, filename)
    df.to_excel(output_path, index=False)

    print(f"✅ Saved results to Excel: {output_path}")

def plot_and_save_pf_single_objective(results_dict, save_dir, network_name, filename_prefix='PF_single_objective'):
    """
    Plot and save Pareto Front for single-objective algorithm results.

    :param results_dict: Dict of {algorithm_name: [runs]}, where each run is a list of tuples (seed_set, (f1, f2), time)
    :param save_dir: Directory where the figure will be saved
    :param network_name: Name of the dataset/network to include in the figure title and filename
    :param filename_prefix: Optional prefix for the saved file name
    """
    plt.figure(figsize=(10, 7))

    for name, runs in results_dict.items():
        for run in runs:
            for solution in run:
                f1, f2 = solution[1]
                plt.scatter(f1, f2, label=name)

    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Fitness Objective 1")
    plt.ylabel("Fitness Objective 2")
    plt.title(f"Pareto Front (Single-Objective): {network_name}")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{filename_prefix}_{network_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Saved PF figure to: {save_path}")
