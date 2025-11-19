import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ga = pd.read_csv("benchmark_quickie_results/ga_all_seeds.csv")
nsga = pd.read_csv("benchmark_quickie_results/nsga_all_seeds.csv")
pareto = []
with open("benchmark_quickie_results/pareto_nsga_seed_4.txt") as f:
    for line in f:
        pareto.append(eval(line.strip()))
pareto = np.array(pareto)

ga_grouped = ga.groupby("gen").agg({
    "best_fitness": "mean",
    "mean_fitness": "mean"
})

plt.plot(ga_grouped.index, ga_grouped["best_fitness"])
plt.title("GA Convergence Across Seeds")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()

nsga_grouped = nsga.groupby("gen").hv_proxy.mean()
plt.plot(nsga_grouped.index, nsga_grouped)
plt.title("NSGA-II: Hypervolume Proxy Over Generations")
plt.xlabel("Generation")
plt.ylabel("HV proxy")
plt.show()

plt.scatter(pareto[:,0], pareto[:,1])
plt.xlabel("Stimulus")
plt.ylabel("Fatigue")
plt.title("Pareto Front: Stimulus vs Fatigue")
plt.show()

# not presented i just thought it's cool :DD 

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto[:,0], pareto[:,1], pareto[:,2])
ax.set_xlabel("Stimulus")
ax.set_ylabel("Fatigue")
ax.set_zlabel("Time (min)")
plt.title("3D Pareto Front")
plt.show()
