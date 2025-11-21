# Hypertrophy Training Plan Optimization Using Genetic Algorithms & NSGA-II

This project applies Evolutionary Computation to automatically generate 7-day hypertrophy-optimized gym training plans using:

* A Single-Objective Genetic Algorithm (GA)
* A Multi-Objective NSGA-II Algorithm
* A custom Stimulus–Recovery–Adaptation (SRA) performance model
* A realistic exercise database (compounds, isolations, machines)
* Domain-informed constraints such as push/pull/legs weekly split, session length limits, compound-first ordering, and exercise diversity

---

The system outputs weekly workout schedules that balance:

- Muscle stimulus
- Fatigue
- Weekly training time
- Exercise diversity
- Realistic gym constraints

---

# Repository Structure

```
/src
  ├── model.py            # Core domain model, SRA, stimulus calculation, repair
  ├── ga_run.py           # Single-objective Genetic Algorithm
  ├── nsga.py             # Multi-objective NSGA-II
  ├── benchmark.py        # Automated benchmarking for both algorithms
/data
  ├── exercises.json      # Exercise database (type, targets, prox-to-failure)
/benchmark_results        # Results from benchmark.py (GA + NSGA-II)
/README.md                # The thing you are reading :D
```

---

# Core Idea

The goal is to automatically generate hypertrophy-focused workout plans using evolutionary optimization.
Each weekly plan is represented as:

* 7 sessions
* each session = up to 5 exercise “blocks”
* each block = `(exercise, sets, reps, intensity, rest)`

A SRA-inspired performance model converts workout structure into:

* H_sum = total hypertrophy stimulus
* F_sum = accumulated fatigue
* Minutes = total weekly time

These are used as fitness objectives.

---

# Algorithms Implementation

## 1. Single-Objective GA

Maximizes a custom fitness function:

```
score = H_sum
      - fatigue penalties
      - time penalties
      + diversity & coverage bonuses
      + compound preference
```

Used to evolve one “best” weekly plan.

---

## 2. Multi-Objective NSGA-II

Optimizes 3 conflicting goals:

* Maximize H_sum
* Minimize Minutes
* Minimize F_sum

Why NSGA-II?

* Handles trade-offs without arbitrary weights
* Produces a Pareto front of optimal training plans
* Provides multiple solutions (high-stimulus, low-fatigue, short-time)
* Built into DEAP and stable for 3-objective problems

---

# How It Works

### Encoding (Genome Representation)

Each individual is a flattened vector of length:

```
7 days × 5 blocks/day = 35 genes
```

Each gene is either:

* `None` (empty slot)
* a `Block` object containing exercise parameters

### Decoding

`decode_to_plan()` reshapes the genome → 7-day Plan.

---

### Initialization

`init_individual()` creates realistic sessions:

* uses push/pull/legs split
* picks exercises matching daily focus
* avoids duplicates
* keeps occasional sparsity
* compounds or isolations chosen based on exercise type

---

### Mutation

Each gene has a 15% chance (`indpb=0.15`) to mutate:

* new exercise
* tweak sets
* tweak reps
* tweak intensity (RIR proxy)
* tweak rest
* or redraw entire block

Mutated sessions get compound-first ordering.

---

### Crossover

1-point crossover aligned to day boundaries:

```
| Day1 | Day2 | Day3 | Day4 | ...
                      ⬑ crossover occurs here
```

This prevents breaking session structure.

---

### Repair Operators

Repair ensures:

* sessions ≤ 100 minutes
* non-rest days ≥ 40 minutes
* rest days remain empty
* excessive long sessions trimmed
* short sessions topped-up

Makes the plan gym-realistic.

---

# Benchmarking

`benchmark.py` runs:

* 5 seeds
* 80 generations
* full GA & NSGA-II comparison

Outputs:

* logs
* Pareto front plots
* best GA plans
* best NSGA-II plans
* aggregated stats

---

# Results Highlights

* GA converges to a balanced, realistic weekly plan
* NSGA-II provides a frontier of training options, such as:

  * Max Stimulus plan
  * Low Fatigue plan
  * Short-Time plan
* Session times remained 45–55 minutes on average
* Exercise variety increased with mutation tuning
* No overstuffed sessions due to repair constraints

---

# How to Run

### Run single-objective GA

```bash
python3 -m src.ga_run --seeds 0 1 2 --gens 80 --pop 120
```

### Run NSGA-II

```bash
python3 -m src.nsga --seeds 0 1 2 --gens 80 --pop 120
```

### Run full benchmark

```bash
python3 src/benchmark.py
```

---
