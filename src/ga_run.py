import argparse, random, numpy as np, pandas as pd, os
from deap import base, creator, tools
from .model import (load_exercise_db, Plan, Session, Block,
                    evaluate_single_objective, summarize_plan,
                    DAYS, MAX_BLOCKS_PER_DAY)

def make_toolbox(exdb):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    ex_ids = list(exdb.keys())

    def random_block():
        ex = random.choice(ex_ids)
        sets = random.randint(2, 5); reps = random.randint(6, 15)
        intensity = round(random.uniform(0.60, 0.85), 2)
        rest = random.choice([60, 90, 120, 150, 180])
        return (ex, sets, reps, intensity, rest)

    def init_individual():
        genome = []
        for _ in range(DAYS * MAX_BLOCKS_PER_DAY):
            genome.append(random_block() if random.random() < 0.8 else None)
        return creator.Individual(genome)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def decode_to_plan(ind):
        days = []
        it = iter(ind)
        for _ in range(DAYS):
            blocks = []
            for _ in range(MAX_BLOCKS_PER_DAY):
                b = next(it)
                if b is None: continue
                ex, sets, reps, intensity, rest = b
                blocks.append(Block(ex, sets, reps, intensity, rest))
            days.append(Session(blocks))
        return Plan(days)

    def evaluate(ind):
        plan = decode_to_plan(ind)
        score, H, F, M = evaluate_single_objective(plan, exdb)
        ind._last_eval = (score, H, F, M)
        return (score,)

    def mate(ind1, ind2):
        point = random.randint(1, DAYS-1) * MAX_BLOCKS_PER_DAY
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        return ind1, ind2

    def mutate(ind, indpb=0.15):
        for i in range(len(ind)):
            if random.random() < indpb:
                b = ind[i]
                if b is None and random.random() < 0.5:
                    ind[i] = random_block()
                elif b is not None:
                    ex, sets, reps, intensity, rest = b
                    r = random.random()
                    if r < 0.2:
                        ex = random.choice(ex_ids)
                    elif r < 0.4:
                        sets = max(2, min(6, sets + random.choice([-1, 1])))
                    elif r < 0.6:
                        reps = max(6, min(20, reps + random.choice([-2, 2])))
                    elif r < 0.8:
                        intensity = round(max(0.55, min(0.9, intensity + random.choice([-0.05, 0.05]))), 2)
                    else:
                        rest = max(45, min(240, rest + random.choice([-15, 15])))
                    ind[i] = (ex, sets, reps, intensity, rest)
        return (ind,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--cxpb", type=float, default=0.8)
    ap.add_argument("--mutpb", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="results/ga_log.csv")
    ap.add_argument("--exdb", type=str, default="data/exercises.json")
    args = ap.parse_args()

    exdb = load_exercise_db(args.exdb)
    toolbox = make_toolbox(exdb)

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed)
        pop = toolbox.population(n=args.pop)

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        rows = []
        for g in range(args.gens):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for i in range(1, len(offspring), 2):
                if random.random() < args.cxpb:
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
            for i in range(len(offspring)):
                if random.random() < args.mutpb:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            pop[:] = tools.selBest(pop + offspring, args.pop)

            fits = [ind.fitness.values[0] for ind in pop]
            best = tools.selBest(pop, 1)[0]
            score, H, F, M = getattr(best, "_last_eval", (None,None,None,None))
            rows.append({"seed":seed,"gen":g,"best":max(fits),"mean":sum(fits)/len(fits),
                         "best_H":H,"best_F":F,"best_Min":M})

        df = pd.DataFrame(rows)
        header = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, mode="a", header=header, index=False)

        # Print best plan for this seed
        def decode_to_plan(ind):
            days=[]; it=iter(ind)
            for _ in range(DAYS):
                blocks=[]
                for _ in range(MAX_BLOCKS_PER_DAY):
                    b=next(it)
                    if b is None: continue
                    ex,sets,reps,intensity,rest=b
                    blocks.append(Block(ex,sets,reps,intensity,rest))
                days.append(Session(blocks))
            return Plan(days)
        elite = tools.selBest(pop,1)[0]
        print(f"Seed {seed} best plan:")
        print(summarize_plan(decode_to_plan(elite)))

if __name__ == "__main__":
    main()
