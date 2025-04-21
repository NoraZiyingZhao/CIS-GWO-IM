import matplotlib.pyplot as plt
import csv

class Visualizer:
    def __init__(self):
        self.hv_curve = []

    def record_hv(self, archive):
        hv = sum(w.Cost[0] for w in archive)  # sum of spreads as proxy HV
        self.hv_curve.append(hv)

    def plot_hv(self):
        plt.plot(self.hv_curve)
        plt.xlabel("Iteration")
        plt.ylabel("Sum of Spread (Proxy HV)")
        plt.title("Convergence Curve")
        plt.grid(True)
        plt.show()

    def save_archive_csv(self, archive, filename="results.csv"):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["SeedSet", "Spread", "Cost", "Fairness"])
            for wolf in archive:
                writer.writerow([sorted(wolf.Position)] + wolf.Cost)
