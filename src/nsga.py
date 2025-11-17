import argparse, random, numpy as np, pandas as pd, os
from typing import List
from deap import base, creator, tools
from .model import (
    load_exercise_db, Plan, Session, Block,
    evaluate_single_objective,  # we'll reuse to compute (H_sum, F_sum, Minutes)
    summarize_plan, DAYS, MAX_BLOCKS_PER_DAY
)

# If you added this earlier, import; otherwise it’s optional.
try:
    from .model import order_session_by_compound_first
    HAS_ORDERING = True
except Exception:
    HAS_ORDERING = False

# --- Optional: same split-focus you used for GA
SPLIT_FOCUS = {
    0: ["chest", "shoulders", "triceps"],    # Push
    1: ["back", "biceps"],                   # Pull
    2: ["quads", "hamstrings", "glutes"],    # Legs
    3: ["chest", "shoulders", "triceps"],    # Push
    4: ["back", "biceps"],                   # Pull
    5: ["quads", "glutes", "calves"],        # Legs
    6: []                                    # Rest
}

def make_toolbox(exdb):
    # -------- Fitness + Individual (Multi-Objective) --------
    # weights: (+H_sum, -Minutes, -F_sum)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(+1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    ex_ids = list(exdb.keys())

    # -------- Building blocks (same style as your GA) --------
    def random_block(ex_ids: List[str], exdb, day_focus=None) -> Block:
        if day_focus:
            focus_ex = [eid for eid in ex_ids if any(m in exdb[eid]["targets"] for m in day_focus)]
            ex = random.choice(focus_ex) if focus_ex else random.choice(ex_ids)
        else:
            ex = random.choice(ex_ids)

        info = exdb[ex]
        reps = random.randint(5, 10) if info["type"] == "compound" else random.randint(10, 15)
        rest = random.choice([120, 150, 180]) if info["type"] == "compound" else random.choice([60, 90, 120])
        sets = random.randint(3, 5)
        intensity = round(random.uniform(0.7, 0.85 if info["type"] == "compound" else 0.8), 2)
        return Block(ex, sets, reps, intensity, rest)

    def init_individual():
        genome = []
        for day in range(DAYS):
            focus = SPLIT_FOCUS.get(day, [])
            day_pool = [eid for eid in ex_ids if not focus or any(m in exdb[eid]["targets"] for m in focus)]
            used = set()
            for _ in range(MAX_BLOCKS_PER_DAY):
                if not focus:
                    genome.append(None)  # rest day
                    continue
                if random.random() >= 0.9:
                    genome.append(None)
                    continue
                choices = [e for e in day_pool if e not in used] or day_pool
                ex_block = random_block(choices, exdb, focus)
                genome.append(ex_block)
                used.add(ex_block.ex_id)
        return creator.Individual(genome)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # -------- Encoding/Decoding --------
    def decode_to_plan(ind):
        days = []
        it = iter(ind)
        for _ in range(DAYS):
            blocks = []
            for _ in range(MAX_BLOCKS_PER_DAY):
                b = next(it)
                if b is None: 
                    continue
                if isinstance(b, Block):
                    blocks.append(b)
                else:
                    ex, sets, reps, intensity, rest = b
                    blocks.append(Block(ex, sets, reps, intensity, rest))
            if HAS_ORDERING:
                # ensure compounds first
                blocks = order_session_by_compound_first(Session(blocks), exdb).blocks
            days.append(Session(blocks))
        return Plan(days)

    toolbox.register("decode_to_plan", decode_to_plan)

    # -------- Multi-objective evaluation --------
    def evaluate(ind):
        plan = decode_to_plan(ind)
        # reuse the single-objective to get (score, H_sum, F_sum, minutes)
        _, H_sum, F_sum, minutes = evaluate_single_objective(plan, exdb)
        # Return a 3-tuple matching FitnessMulti weights
        #   (maximize H_sum, minimize minutes, minimize F_sum)
        return (H_sum, minutes, F_sum)

    toolbox.register("evaluate", evaluate)

    # -------- Variation operators (same as GA) --------
    def mate(ind1, ind2):
        point = random.randint(1, DAYS-1) * MAX_BLOCKS_PER_DAY
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        return ind1, ind2

    def mutate(ind, indpb=0.15):
        for i in range(len(ind)):
            if random.random() < indpb:
                day = i // MAX_BLOCKS_PER_DAY
                focus = SPLIT_FOCUS.get(day, [])
                b = ind[i]

                if b is None and random.random() < 0.5:
                    ind[i] = random_block(ex_ids, exdb, focus)
                    continue

                if b is not None:
                    if isinstance(b, Block):
                        ex, sets, reps, intensity, rest = b.ex_id, b.sets, b.reps, b.intensity, b.rest_s
                    else:
                        ex, sets, reps, intensity, rest = b

                    r = random.random()
                    if r < 0.2:
                        if random.random() < 0.5:
                            ind[i] = random_block(ex_ids, exdb, focus)
                            continue
                        focus_pool = [eid for eid in ex_ids if not focus or any(m in exdb[eid]["targets"] for m in focus)]
                        ex = random.choice(focus_pool) if focus_pool else random.choice(ex_ids)
                    elif r < 0.4:
                        sets = max(2, min(6, sets + random.choice([-1, 1])))
                    elif r < 0.6:
                        reps = max(6, min(20, reps + random.choice([-2, 2])))
                    elif r < 0.8:
                        intensity = round(max(0.55, min(0.9, intensity + random.choice([-0.05, 0.05]))), 2)
                    else:
                        rest = max(45, min(240, rest + random.choice([-15, 15])))

                    ind[i] = Block(ex, sets, reps, intensity, rest)

        # keep representation consistent (Blocks or None); ordering happens in decode/eval
        return (ind,)

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.15)

    # -------- NSGA-II selection --------
    toolbox.register("select", tools.selNSGA2)           # elitist replacement
    toolbox.register("select_tournament", tools.selTournamentDCD)  # for parent selection (crowding tournament)

    return toolbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--cxpb", type=float, default=0.9)
    ap.add_argument("--mutpb", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="results/nsga2_log.csv")
    ap.add_argument("--exdb", type=str, default="data/exercises.json")
    args = ap.parse_args()

    exdb = load_exercise_db(args.exdb)
    toolbox = make_toolbox(exdb)

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed)

        pop = toolbox.population(n=args.pop)
        # NSGA-II wants the initial population to be sorted and assigned crowding distances
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop, len(pop))  # nondominated sort + crowding

        pareto = tools.ParetoFront()  # track the global Pareto set

        rows = []
        for g in range(args.gens):
            # --- Parent selection with crowding tournament ---
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # --- Variation ---
            for i in range(1, len(offspring), 2):
                if random.random() < args.cxpb:
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
            for i in range(len(offspring)):
                if random.random() < args.mutpb:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # --- Evaluate ---
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # --- Elitist replacement (NSGA-II) ---
            pop = toolbox.select(pop + offspring, args.pop)

            # --- Track Pareto front ---
            pareto.update(pop)

            # --- Simple logging (best medians etc.) ---
            H_vals = [i.fitness.values[0] for i in pop]
            Min_vals = [i.fitness.values[1] for i in pop]
            F_vals = [i.fitness.values[2] for i in pop]
            rows.append({
                "seed": seed, "gen": g,
                "H_mean": float(np.mean(H_vals)), "H_max": float(np.max(H_vals)),
                "Min_mean": float(np.mean(Min_vals)), "Min_min": float(np.min(Min_vals)),
                "F_mean": float(np.mean(F_vals)), "F_min": float(np.min(F_vals)),
                "pareto_size": len(pareto)
            })

        # --- Write log ---
        df = pd.DataFrame(rows)
        header = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, mode="a", header=header, index=False)

        # --- Show a few Pareto solutions (diverse trade-offs) ---
        print(f"\n=== Seed {seed} | Pareto size: {len(pareto)} ===")
        # pick up to 3 diverse solutions: best H, best time, best fatigue
        best_H = max(pareto, key=lambda ind: ind.fitness.values[0])
        best_Min = min(pareto, key=lambda ind: ind.fitness.values[1])
        best_F = min(pareto, key=lambda ind: ind.fitness.values[2])

        def show_ind(label, ind):
            plan = toolbox.decode_to_plan(ind)
            _, H_sum, F_sum, minutes = evaluate_single_objective(plan, exdb)
            print(f"\n[{label}]  H_sum={H_sum:.2f} | Minutes={minutes:.1f} | F_sum={F_sum:.2f}")
            print(summarize_plan(plan))

        show_ind("Max stimulus", best_H)
        show_ind("Shortest time", best_Min)
        show_ind("Lowest fatigue", best_F)


if __name__ == "__main__":
    main()
