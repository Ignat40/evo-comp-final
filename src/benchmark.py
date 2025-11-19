import numpy as np
import pandas as pd
import random, os, time
from deap import tools
from src.ga_run import make_toolbox as make_ga_toolbox
from src.nsga import make_toolbox as make_nsga_toolbox
from src.model import (
    load_exercise_db, evaluate_single_objective,
    summarize_plan, decode_to_plan, DAYS, MAX_BLOCKS_PER_DAY
)


# Baseline 

# SEEDS = [0, 1, 2, 3, 4]
# GENERATIONS = 80
# POP_SIZE = 120
# GA_MUTPB = 0.25
# GA_CXPB = 0.85

# NSGA_MUTPB = 0.25
# NSGA_CXPB = 0.9

# Quickie

# SEEDS = [0, 1, 2, 3, 4]
# GENERATIONS = 40
# POP_SIZE = 80
# GA_MUTPB = 0.15
# GA_CXPB = 0.70

# NSGA_MUTPB = 0.20
# NSGA_CXPB = 0.75

# Loaded

# HEAVY / LOADED SETTINGS
SEEDS = [0, 1, 2, 3, 4]

GENERATIONS = 150   
POP_SIZE = 200       

GA_MUTPB = 0.30        
GA_CXPB  = 0.90       

NSGA_MUTPB = 0.30      
NSGA_CXPB  = 0.95     


EXDB_PATH = "data/exercises.json"

SAVE_DIR = "benchmark_loaded_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def run_single_objective_ga(seed, exdb):
    toolbox = make_ga_toolbox(exdb)
    random.seed(seed); np.random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    rows = []

    for gen in range(GENERATIONS):
        start = time.time()

        offspring = toolbox.select(pop, POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # pairing
        for i in range(1, len(offspring), 2):
            if random.random() < GA_CXPB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # mutation
        for i in range(len(offspring)):
            if random.random() < GA_MUTPB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # evaluate invalid
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # survivor selection
        pop[:] = toolbox.select(pop + offspring, POP_SIZE)

        # logging
        fits = [ind.fitness.values[0] for ind in pop]
        best = max(fits)

        rows.append({
            "seed": seed,
            "gen": gen,
            "best_fitness": best,
            "mean_fitness": sum(fits) / len(fits),
            "runtime_sec": time.time() - start
        })

    return pop, pd.DataFrame(rows)


def run_nsga(seed, exdb):
    toolbox = make_nsga_toolbox(exdb)
    random.seed(seed); np.random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    pop = tools.selNSGA2(pop, POP_SIZE)

    rows = []

    for gen in range(GENERATIONS):
        start = time.time()

        offspring = tools.selTournamentDCD(pop, len(pop))  
        offspring = list(map(toolbox.clone, offspring))

        for i in range(1, len(offspring), 2):
            if random.random() < NSGA_CXPB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < NSGA_MUTPB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = tools.selNSGA2(pop + offspring, POP_SIZE)

        fitnesses = np.array([ind.fitness.values for ind in pop])
        fronts = tools.sortNondominated(pop, k=len(pop))

        rows.append({
            "seed": seed,
            "gen": gen,
            "nondominated_count": len(fronts[0]),
            "hv_proxy": np.sum(1 / (fitnesses[:,0] + 1e-9)), 
            "runtime_sec": time.time() - start
        })

    return pop, pd.DataFrame(rows)


def main():
    exdb = load_exercise_db(EXDB_PATH)

    GA_logs = []
    NSGA_logs = []

    for seed in SEEDS:
        print(f"=== Running GA for seed {seed} ===")
        final_pop_ga, df_ga = run_single_objective_ga(seed, exdb)
        df_ga.to_csv(f"{SAVE_DIR}/ga_seed_{seed}.csv", index=False)
        GA_logs.append(df_ga)

        print(f"=== Running NSGA-II for seed {seed} ===")
        final_pop_nsga, df_nsga = run_nsga(seed, exdb)
        df_nsga.to_csv(f"{SAVE_DIR}/nsga_seed_{seed}.csv", index=False)
        NSGA_logs.append(df_nsga)

        # Save best GA plan
        best_ga = max(final_pop_ga, key=lambda ind: ind.fitness.values[0])
        plan_ga = decode_to_plan(best_ga)
        with open(f"{SAVE_DIR}/best_ga_seed_{seed}.txt", "w") as f:
            f.write(summarize_plan(plan_ga))

        # Save Pareto Front (NSGA-II)
        pareto_front = tools.sortNondominated(final_pop_nsga, 1)[0]
        with open(f"{SAVE_DIR}/pareto_nsga_seed_{seed}.txt", "w") as f:
            for ind in pareto_front:
                f.write(str(ind.fitness.values) + "\n")

    # Combine logs
    pd.concat(GA_logs).to_csv(f"{SAVE_DIR}/ga_all_seeds.csv", index=False)
    pd.concat(NSGA_logs).to_csv(f"{SAVE_DIR}/nsga_all_seeds.csv", index=False)

    print("✓ Benchmark complete")


if __name__ == "__main__":
    main()
