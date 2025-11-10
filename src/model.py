"""
This module provides:
1) A full 7-day workout plan representation
2) Effective Stimulus (ES) per set calculation
3) Stimulus-Recovery-Adaptation (SRA) model
4) Feasibility repair (≤90 min per session)
5) Single-objective evaluation for the GA
6) Random plan generator for baseline
7) Pretty-printing for presentations
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json, math, random, os

# ==========================================================
# 0. GLOBAL CONFIGURATION
# ==========================================================

MUSCLES = [
    "chest","back","quads","hamstrings","glutes",
    "shoulders","biceps","triceps","calves","core"
]

DAYS = 7
MAX_BLOCKS_PER_DAY = 5
SESSION_CAP_MIN = 100.0 # Max minutes per session
C1_FATIGUE = 0.30       # fatigue penalty 
C2_TIME = 0.01          # time penalty

# ==========================================================
# 1. LOADING EXERCISE METADATA
# ==========================================================

def load_exercise_db(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        db = json.load(f)

    for k, v in db.items():
        if "type" not in v:
            raise ValueError(f"Exercise '{k}' missing type.")
        if "targets" not in v:
            raise ValueError(f"Exercise '{k}' missing targets.")
        v["prox"] = max(0.5, min(1.2, float(v.get("prox", 0.8))))
    return db

# ==========================================================
# 2. DATA STRUCTURES
# ==========================================================

@dataclass
class Block:
    ex_id: str
    sets: int
    reps: int
    intensity: float   # fraction of 1RM (0.55–0.9 typical)
    rest_s: int        # rest between sets in seconds

@dataclass
class Session:
    blocks: List[Block]

@dataclass
class Plan:
    sessions: List[Session]  # 7 sessions (some may be empty)

# ==========================================================
# 3. TIME MODEL
# ==========================================================

def set_minutes(reps: int, rest_s: int) -> float:
    """
    Approximate minutes per set = time under tension + rest.
    3 seconds per rep is typical for hypertrophy tempo.
    """
    return (reps * 3.0) / 60.0 + rest_s / 60.0

def block_minutes(b: Block) -> float:
    return b.sets * set_minutes(b.reps, b.rest_s)

def session_minutes(s: Session) -> float:
    return sum(block_minutes(b) for b in s.blocks)

def plan_minutes(p: Plan) -> float:
    return sum(session_minutes(s) for s in p.sessions)

# ==========================================================
# 4. EFFECTIVE STIMULUS PER SET
# ==========================================================

def es_per_set(ex: dict, reps: int, intensity: float) -> float:
    """
    Effective stimulus = w_range * w_prox * f_rep * f_intensity

    - w_range: 1.1 for compound, 0.9 for isolation
    - w_prox: proximity to failure (0.6-1.2)
    - f_rep: saturates at 10 reps (8-12 reps ideal)
    - f_intensity: saturates at 75% 1RM
    """
    w_range = 1.1 if ex["type"] == "compound" else 0.9
    w_prox = max(0.6, min(1.2, float(ex.get("prox", 0.8))))
    f_rep = min(1.0, reps / 10.0)
    f_int = min(1.0, intensity / 0.75)
    return w_range * w_prox * f_rep * f_int

# ==========================================================
# 5. SRA MODEL (Stimulus–Recovery–Adaptation)
# ==========================================================

class SRAModel:
    """
    Sports-science-inspired fitness-fatigue model.

    A_t = alpha*SS_t + beta*A_{t-1}*decay
    F_t = gamma*SS_t + delta*F_{t-1}*decay
    H_m = A_7 - λ*F_7
    """

    def __init__(self, tau_hours: float = 48.0,
                 alpha: float = 1.0, beta: float = 0.85,
                 gamma: float = 0.8, delta: float = 0.85,
                 lam: float = 0.6):
        self.tau = tau_hours
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.lam = lam
        self.decay = math.exp(-24.0 / self.tau)

    def roll_week(self, daily_stimuli: List[Dict[str, List[float]]]) -> Tuple[Dict[str,float], Dict[str,float]]:
        A = {m: 0.0 for m in MUSCLES}
        F = {m: 0.0 for m in MUSCLES}
        for day in daily_stimuli:
            for m in MUSCLES:
                SS = sum(day.get(m, []))
                A[m] = self.alpha * SS + self.beta * A[m] * self.decay
                F[m] = self.gamma * SS + self.delta * F[m] * self.decay
        H = {m: A[m] - self.lam * F[m] for m in MUSCLES}
        return H, F

# ==========================================================
# 6. PLAN → DAILY STIMULI
# ==========================================================

def plan_to_daily_stimuli(plan: Plan, exdb: Dict[str, dict]) -> List[Dict[str, List[float]]]:
    days = []
    for sess in plan.sessions:
        bucket = {m: [] for m in MUSCLES}
        for b in sess.blocks:
            ex = exdb[b.ex_id]
            es = es_per_set(ex, b.reps, b.intensity)
            for m in ex["targets"]:
                bucket[m].extend([es] * b.sets)
        days.append(bucket)
    return days

# ==========================================================
# 7. REPAIR (FEASIBILITY)
# ==========================================================

def repair_plan(plan: Plan, session_cap_min: float = SESSION_CAP_MIN) -> Plan:
    """
    Keeps sessions under 90 minutes by:
      - Reducing sets >2
      - Reducing rest >60s
      - Dropping smallest block
    """
    new_sessions = []
    for sess in plan.sessions:
        blocks = list(sess.blocks)
        cur = session_minutes(sess)
        j = 0
        while cur > session_cap_min and blocks:
            i = j % len(blocks)
            b = blocks[i]
            if b.sets > 2:
                b.sets -= 1
            elif b.rest_s > 60:
                b.rest_s -= 15
            else:
                # remove smallest-time block
                idx = min(range(len(blocks)), key=lambda k: block_minutes(blocks[k]))
                cur -= block_minutes(blocks[idx])
                blocks.pop(idx)
                j += 1
                continue
            cur = session_minutes(Session(blocks))
            j += 1
        new_sessions.append(Session(blocks))
    return Plan(new_sessions)

# ==========================================================
# 8. EVALUATION (SINGLE-OBJECTIVE)
# ==========================================================

def evaluate_single_objective(plan: Plan, exdb: Dict[str, dict],
                              c1: float = C1_FATIGUE, c2: float = C2_TIME,
                              sra: Optional[SRAModel] = None) -> Tuple[float, float, float, float]:
    sra = sra or SRAModel()
    plan = repair_plan(plan, SESSION_CAP_MIN)
    days = plan_to_daily_stimuli(plan, exdb)
    H, F = sra.roll_week(days)
    H_sum, F_sum = sum(H.values()), sum(F.values())
    minutes = plan_minutes(plan)
    score = H_sum - c1 * F_sum - c2 * minutes
    return score, H_sum, F_sum, minutes

# ==========================================================
# 9. RANDOM PLAN GENERATOR
# ==========================================================

def random_block(ex_ids: List[str]) -> Block:
    ex = random.choice(ex_ids)
    sets = random.randint(2, 5)
    reps = random.randint(6, 15)
    intensity = round(random.uniform(0.6, 0.85), 2)
    rest = random.choice([60, 90, 120, 150, 180])
    return Block(ex, sets, reps, intensity, rest)

def random_plan(exdb: Dict[str, dict],
                max_blocks_per_day: int = MAX_BLOCKS_PER_DAY,
                p_empty: float = 0.2) -> Plan:
    ex_ids = list(exdb.keys())
    sessions = []
    for _ in range(DAYS):
        blocks = []
        for _ in range(max_blocks_per_day):
            if random.random() < p_empty:
                continue
            blocks.append(random_block(ex_ids))
        sessions.append(Session(blocks))
    return repair_plan(Plan(sessions))

# ==========================================================
# 10. PRETTY PRINTING
# ==========================================================

def summarize_plan(plan: Plan) -> str:
    lines = []
    for d, sess in enumerate(plan.sessions, 1):
        if not sess.blocks:
            lines.append(f"Day {d}: REST")
            continue
        lines.append(f"Day {d}:")
        for b in sess.blocks:
            lines.append(
                f"  - {b.ex_id:12s}  {b.sets}x{b.reps} @ {b.intensity:.2f}  rest {b.rest_s}s  (~{block_minutes(b):.1f} min)"
            )
        lines.append(f"    Session ≈ {session_minutes(sess):.1f} min")
    lines.append(f"Week total ≈ {plan_minutes(plan):.1f} min")
    return "\n".join(lines)

# ==========================================================
# 11. SELF-TEST
# ==========================================================

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    exdb_path = os.path.join(os.path.dirname(here), "data", "exercises.json")
    exdb = load_exercise_db(exdb_path)
    random.seed(0)

    plan = random_plan(exdb)
    print("RANDOM FEASIBLE PLAN")
    print(summarize_plan(plan))

    score, H, F, M = evaluate_single_objective(plan, exdb)
    print(f"\nScore={score:.3f} | H_sum={H:.3f} | F_sum={F:.3f} | Minutes={M:.1f}")
