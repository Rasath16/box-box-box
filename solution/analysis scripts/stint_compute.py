"""
Test stint-based computation vs lap-by-lap.
Hypothesis: the simulator computes total time per stint analytically,
then adds stints + pit penalties. This eliminates floating point
accumulation differences for symmetric strategies.

Per stint of N laps on compound c:
stint_time = N * base + N * cb[c] + rate[c] * ts[c] * sum_excess(N, cliff[c])

where sum_excess(N, C) = sum(max(0, age - C) for age in 1..N)
= (N-C)(N-C+1)/2 if N > C, else 0

total_time = sum(stint_times) + n_pits * pit_time
"""
import json
import math

tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

CB = {"SOFT": -0.9665103286569976, "MEDIUM": 0.0, "HARD": 0.755284994643082}
RATE = {"SOFT": 1.6213600572975244, "MEDIUM": 0.813268608577364, "HARD": 0.345981233247675}
CLIFF = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}
TC = {"SOFT": 0.025806274187704845, "MEDIUM": 0.02777171692356944, "HARD": 0.02401965544225936}
TREF = 27.96640138772966

def sum_excess(N, C):
    """Sum of max(0, age - C) for age in 1..N"""
    if N <= C:
        return 0.0
    k = N - C
    return k * (k + 1) / 2.0

def get_stints(strategy, total_laps):
    pit_laps = sorted([(ps["lap"], ps["to_tire"]) for ps in strategy["pit_stops"]])
    stints = []
    compound = strategy["starting_tire"]
    start = 1
    for plap, new_comp in pit_laps:
        n = plap - start + 1
        stints.append((compound, n))
        compound = new_comp
        start = plap + 1
    stints.append((compound, total_laps - start + 1))
    return stints

def compute_stint_based(strategy, config):
    """Compute total time using per-stint analytical formula."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    n_pits = len(strategy["pit_stops"])

    stints = get_stints(strategy, total_laps)
    total_time = 0.0

    for compound, n_laps in stints:
        ts = 1.0 + TC[compound] * (temp - TREF)
        se = sum_excess(n_laps, CLIFF[compound])
        stint_time = n_laps * base + n_laps * CB[compound] + RATE[compound] * ts * se
        total_time += stint_time

    total_time += n_pits * pit_time
    return total_time

def compute_lap_by_lap(strategy, config):
    """Original lap-by-lap computation."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]

    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strategy["pit_stops"]}
    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        ts = 1.0 + TC[compound] * (temp - TREF)
        deg = RATE[compound] * ts * max(0.0, tire_age - CLIFF[compound])
        total_time += base + CB[compound] + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

# Test stint-based vs lap-by-lap
correct_stint = 0
correct_lbl = 0
for inp, exp in tests:
    config = inp["race_config"]
    results_stint = []
    results_lbl = []

    for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
        s = inp["strategies"][pk]
        grid = int(pk[3:])
        t_stint = compute_stint_based(s, config)
        t_lbl = compute_lap_by_lap(s, config)
        results_stint.append((t_stint, grid, s["driver_id"]))
        results_lbl.append((t_lbl, grid, s["driver_id"]))

    results_stint.sort(key=lambda r: (r[0], r[1]))
    results_lbl.sort(key=lambda r: (r[0], r[1]))

    pred_stint = [d for _, _, d in results_stint]
    pred_lbl = [d for _, _, d in results_lbl]

    if pred_stint == exp:
        correct_stint += 1
    if pred_lbl == exp:
        correct_lbl += 1

print(f"Stint-based: {correct_stint}/100")
print(f"Lap-by-lap:  {correct_lbl}/100")

# Try different computation orders
# Order 1: accumulate base separately
def compute_separated(strategy, config):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    n_pits = len(strategy["pit_stops"])

    stints = get_stints(strategy, total_laps)

    # Base time - same for all drivers, but add it as total_laps * base
    base_total = total_laps * base
    pit_total = n_pits * pit_time

    # Strategy-dependent part
    strategy_time = 0.0
    for compound, n_laps in stints:
        ts = 1.0 + TC[compound] * (temp - TREF)
        se = sum_excess(n_laps, CLIFF[compound])
        strategy_time += n_laps * CB[compound] + RATE[compound] * ts * se

    return base_total + pit_total + strategy_time

correct_sep = 0
for inp, exp in tests:
    config = inp["race_config"]
    results = []
    for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
        s = inp["strategies"][pk]
        grid = int(pk[3:])
        t = compute_separated(s, config)
        results.append((t, grid, s["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    pred = [d for _, _, d in results]
    if pred == exp:
        correct_sep += 1

print(f"Separated:   {correct_sep}/100")

# Order 2: compute only strategy cost (skip base since it doesn't affect ranking)
def compute_cost_only(strategy, config):
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    n_pits = len(strategy["pit_stops"])

    stints = get_stints(strategy, total_laps)

    cost = n_pits * pit_time
    for compound, n_laps in stints:
        ts = 1.0 + TC[compound] * (temp - TREF)
        se = sum_excess(n_laps, CLIFF[compound])
        cost += n_laps * CB[compound] + RATE[compound] * ts * se

    return cost

correct_cost = 0
for inp, exp in tests:
    config = inp["race_config"]
    results = []
    for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
        s = inp["strategies"][pk]
        grid = int(pk[3:])
        t = compute_cost_only(s, config)
        results.append((t, grid, s["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    pred = [d for _, _, d in results]
    if pred == exp:
        correct_cost += 1

print(f"Cost-only:   {correct_cost}/100")
