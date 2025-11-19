

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json, math, random, os
import numpy as np



MUSCLES = [
    "chest","back","quads","hamstrings","glutes",
    "shoulders","biceps","triceps","calves","core"
]

DAYS = 7
MAX_BLOCKS_PER_DAY = 5
SESSION_CAP_MIN = 100.0 
C1_FATIGUE = 0.30       # fatigue penalty 
C2_TIME = 0.01          # time penalty


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



@dataclass
class Block:
    ex_id: str
    sets: int
    reps: int
    intensity: float   
    rest_s: int        

@dataclass
class Session:
    blocks: List[Block]

@dataclass
class Plan:
    sessions: List[Session]  # 7 sessions (some may be empty)


def set_minutes(reps: int, rest_s: int) -> float:
    return (reps * 3.0) / 60.0 + rest_s / 60.0

def block_minutes(b: Block) -> float:
    return b.sets * set_minutes(b.reps, b.rest_s)

def session_minutes(s: Session) -> float:
    return sum(block_minutes(b) for b in s.blocks)

def plan_minutes(p: Plan) -> float:
    return sum(session_minutes(s) for s in p.sessions)



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


def order_session_by_compound_first(sess: Session, exdb: Dict[str, dict]) -> Session:
    """Sort blocks so compound exercises come before isolation ones."""
    compounds = [b for b in sess.blocks if exdb[b.ex_id]["type"] == "compound"]
    isolations = [b for b in sess.blocks if exdb[b.ex_id]["type"] != "compound"]
    return Session(compounds + isolations)


def repair_plan(plan: Plan, session_cap_min: float = SESSION_CAP_MIN,
                min_session_min: float = 50.0, exdb=None) -> Plan:
    """Keeps sessions under cap and ensures ≥ min_session_min ONLY for non-empty sessions."""
    ex_ids = list(exdb.keys()) if exdb else []
    new_sessions = []

    for sess in plan.sessions:
        blocks = list(sess.blocks)
        was_empty = (len(blocks) == 0)   # if this is a rest day
        cur = session_minutes(sess)
        j = 0

        # Cap long sessions
        while cur > session_cap_min and blocks:
            i = j % len(blocks)
            b = blocks[i]
            if b.sets > 2:
                b.sets -= 1
            elif b.rest_s > 60:
                b.rest_s -= 15
            else:
                idx = min(range(len(blocks)), key=lambda k: block_minutes(blocks[k]))
                cur -= block_minutes(blocks[idx])
                blocks.pop(idx)
                j += 1
                continue
            cur = session_minutes(Session(blocks))
            j += 1

        if not was_empty and ex_ids:
            while cur < min_session_min and blocks:
                blocks.append(random_block(ex_ids, exdb))
                cur = session_minutes(Session(blocks))


        sess_ordered = order_session_by_compound_first(Session(blocks), exdb)
        new_sessions.append(sess_ordered)

    return Plan(new_sessions)

# ==========================================================
# 8. EVALUATION (SINGLE-OBJECTIVE)
# ==========================================================

# def evaluate_single_objective(plan: Plan, exdb: Dict[str, dict],
#                               c1: float = C1_FATIGUE, c2: float = C2_TIME,
#                               sra: Optional[SRAModel] = None) -> Tuple[float, float, float, float]:
#     sra = sra or SRAModel()
#     plan = repair_plan(plan, SESSION_CAP_MIN)
#     days = plan_to_daily_stimuli(plan, exdb)
#     H, F = sra.roll_week(days)
#     H_sum, F_sum = sum(H.values()), sum(F.values())
#     minutes = plan_minutes(plan)
#     score = H_sum - c1 * F_sum - c2 * minutes
#     return score, H_sum, F_sum, minutes

def evaluate_single_objective(plan, exdb, sra=None):
    import numpy as np
    sra = sra or SRAModel()
    plan = repair_plan(plan, SESSION_CAP_MIN, 40.0, exdb)
    days = plan_to_daily_stimuli(plan, exdb)
    H, F = sra.roll_week(days)
    H_sum, F_sum = sum(H.values()), sum(F.values())
    minutes = plan_minutes(plan)

    # ---- Diversity & balance ----
    all_ex = [b.ex_id for s in plan.sessions for b in s.blocks]
    dup_penalty = sum((len(exs) - len(set(exs))) / max(1, len(exs))
                      for exs in ([b.ex_id for b in s.blocks] for s in plan.sessions))
    stim = np.array(list(H.values()))
    imbalance = np.std(stim)

    # ---- Session durations ----
    daily_minutes = [session_minutes(s) for s in plan.sessions]
    short_penalty = sum(1 for m in daily_minutes if 0 < m < 45) * 5.0   
    long_bonus    = sum(1 for m in daily_minutes if m > 55) * 3.0       
    rest_bonus    = sum(1 for m in daily_minutes if m < 25) * 1.0       

    # ---- Compound preference ----
    compound_ratio = sum(1 for b in all_ex if exdb[b]["type"] == "compound") / max(1, len(all_ex))

    # ---- Coverage & alternation ----
    coverage = sum(1 for v in H.values() if v > 0.5) / len(MUSCLES)
    prev_targets = set(); penalty_repeat = 0
    for sess in plan.sessions:
        todays_targets = {m for b in sess.blocks for m in exdb[b.ex_id]["targets"]}
        overlap = len(todays_targets & prev_targets)
        penalty_repeat += overlap / max(1, len(todays_targets))
        prev_targets = todays_targets

    
    score = (
        1.2 * H_sum                      
        - 0.25 * F_sum
        - 0.015 * minutes
        - 20 * dup_penalty
        - 15 * imbalance
        + 10 * coverage
        + 5 * compound_ratio             
        + long_bonus
        + rest_bonus
        - short_penalty
        - 5 * penalty_repeat
    )
    return score, H_sum, F_sum, minutes


def random_block(ex_ids: List[str], exdb=None, day_focus=None) -> Block:
    """Generate a random exercise block, optionally using focus muscles."""
    if exdb and day_focus:
        focus_ex = [eid for eid in ex_ids if any(m in exdb[eid]["targets"] for m in day_focus)]
        ex = random.choice(focus_ex) if focus_ex else random.choice(ex_ids)
    else:
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
            blocks.append(random_block(ex_ids, exdb))
        s = Session(blocks)
        
        while session_minutes(s) < 40 and blocks:
            blocks.append(random_block(ex_ids, exdb))
            s = Session(blocks)
        sessions.append(s)
    return repair_plan(Plan(sessions))


def decode_to_plan(individual) -> Plan:
    """
    Converts a flat GA/NSGA-II genome of length DAYS * MAX_BLOCKS_PER_DAY
    into a Plan object containing 7 Sessions.
    Each slot may contain:
        - None
        - a Block instance
        - a tuple representation (ex_id, sets, reps, intensity, rest)
    """
    days = []
    it = iter(individual)

    for _ in range(DAYS):
        blocks = []
        for _ in range(MAX_BLOCKS_PER_DAY):
            gene = next(it)

            if gene is None:
                continue

            if isinstance(gene, Block):
                blocks.append(gene)
            else:
                # tuple-like representation
                ex_id, sets, reps, intensity, rest_s = gene
                blocks.append(Block(ex_id, sets, reps, intensity, rest_s))

        days.append(Session(blocks))

    return Plan(days)



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

# needs to be optimized in the future 
