import argparse, random, numpy as np, pandas as pd, os
from typing import List
from .model import order_session_by_compound_first
from deap import base, creator, tools
from .model import (load_exercise_db, Plan, Session, Block,
                    evaluate_single_objective, summarize_plan,
                    DAYS, MAX_BLOCKS_PER_DAY)

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
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    ex_ids = list(exdb.keys())

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
                if random.random() >= 0.9:  # keep a little sparsity
                    genome.append(None)
                    continue
                # prefer unused first; if exhausted, allow reuse
                choices = [e for e in day_pool if e not in used] or day_pool
                ex_block = random_block(choices, exdb, focus)
                genome.append(ex_block)
                used.add(ex_block.ex_id)
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
                if isinstance(b, Block):
                    blocks.append(b)
                else:
                    ex, sets, reps, intensity, rest = b
                    blocks.append(Block(ex, sets, reps, intensity, rest))
            days.append(Session(blocks))
        return Plan(days)
    toolbox.register("decode_to_plan", decode_to_plan)


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
                day = i // MAX_BLOCKS_PER_DAY
                focus = SPLIT_FOCUS.get(day, [])
                b = ind[i]

                if b is None and random.random() < 0.5:
                    ind[i] = random_block(ex_ids, exdb, focus)
                    continue

                if b is not None:
                    # Unpack safely
                    if isinstance(b, Block):
                        ex, sets, reps, intensity, rest = b.ex_id, b.sets, b.reps, b.intensity, b.rest_s
                    else:
                        ex, sets, reps, intensity, rest = b

                    r = random.random()
                    if r < 0.2:
                        # either redraw a brand-new focused block...
                        if random.random() < 0.5:
                            ind[i] = random_block(ex_ids, exdb, focus)
                            continue
                        # ...or just swap the exercise within focus
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
                    

        plan = toolbox.decode_to_plan(ind)
        for day in range(DAYS):
            sess = plan.sessions[day]
            plan.sessions[day] = order_session_by_compound_first(sess, exdb)

        ind[:] = [b for sess in plan.sessions for b in sess.blocks + [None]*(MAX_BLOCKS_PER_DAY - len(sess.blocks))]

                    
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
                    if isinstance(b, Block):
                        blocks.append(b)
                    else:
                        ex, sets, reps, intensity, rest = b
                        blocks.append(Block(ex, sets, reps, intensity, rest))                                      
                days.append(Session(blocks))
            return Plan(days)
        elite = tools.selBest(pop,1)[0]
        print(f"Seed {seed} best plan:")
        print(summarize_plan(decode_to_plan(elite)))

if __name__ == "__main__":
    main()
