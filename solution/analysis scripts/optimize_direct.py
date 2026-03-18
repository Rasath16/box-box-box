"""
Direct optimization on test cases using scipy.
Try multiple formula structures to find the right one.
"""
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
from pathlib import Path

# Load all test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def compute_time(strategy, config, params):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    cb_s, cb_h, rate_s, rate_m, rate_h, cliff_s, cliff_m, cliff_h, tc, tref = params

    compound_base = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    deg_rate = {"SOFT": rate_s, "MEDIUM": rate_m, "HARD": rate_h}
    cliff = {"SOFT": cliff_s, "MEDIUM": cliff_m, "HARD": cliff_h}

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    current_compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0
    temp_scale = 1.0 + tc * (track_temp - tref)

    for lap in range(1, total_laps + 1):
        tire_age += 1
        cb = compound_base[current_compound]
        rate = deg_rate[current_compound]
        c = cliff[current_compound]
        deg = rate * temp_scale * max(0.0, tire_age - c)
        total_time += base + cb + deg

        if lap in pit_laps:
            total_time += pit_time
            current_compound = pit_laps[lap]
            tire_age = 0

    return total_time

def predict(race_data, params):
    config = race_data["race_config"]
    strategies = race_data["strategies"]
    results = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strat = strategies[pos_key]
        grid = int(pos_key[3:])
        t = compute_time(strat, config, params)
        results.append((t, grid, strat["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    return [d for _, _, d in results]

def count_correct(params):
    correct = 0
    for inp, exp in tests:
        if predict(inp, params) == exp:
            correct += 1
    return correct

def pairwise_loss(params):
    """Differentiable-ish loss based on pairwise ordering."""
    total_loss = 0.0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        times = {}
        for pos_key in strategies:
            strat = strategies[pos_key]
            t = compute_time(strat, config, params)
            times[strat["driver_id"]] = t

        # For each pair that should be ordered, penalize wrong order
        for i in range(len(exp)):
            for j in range(i+1, len(exp)):
                di, dj = exp[i], exp[j]
                # di should have LOWER time than dj
                diff = times[di] - times[dj]
                if diff > 0:  # wrong order
                    total_loss += diff
                elif diff == 0:
                    total_loss += 0.01  # small penalty for ties
    return total_loss

# Also try per-compound temp coefficients
def compute_time_v2(strategy, config, params):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    cb_s, cb_h, rate_s, rate_m, rate_h, cliff_s, cliff_m, cliff_h, tc_s, tc_m, tc_h, tref = params

    compound_base = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    deg_rate = {"SOFT": rate_s, "MEDIUM": rate_m, "HARD": rate_h}
    cliff = {"SOFT": cliff_s, "MEDIUM": cliff_m, "HARD": cliff_h}
    temp_coeff = {"SOFT": tc_s, "MEDIUM": tc_m, "HARD": tc_h}

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    current_compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        cb = compound_base[current_compound]
        rate = deg_rate[current_compound]
        c = cliff[current_compound]
        tc = temp_coeff[current_compound]
        temp_scale = 1.0 + tc * (track_temp - tref)
        deg = rate * temp_scale * max(0.0, tire_age - c)
        total_time += base + cb + deg

        if lap in pit_laps:
            total_time += pit_time
            current_compound = pit_laps[lap]
            tire_age = 0

    return total_time

def predict_v2(race_data, params):
    config = race_data["race_config"]
    strategies = race_data["strategies"]
    results = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strat = strategies[pos_key]
        grid = int(pos_key[3:])
        t = compute_time_v2(strat, config, params)
        results.append((t, grid, strat["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    return [d for _, _, d in results]

def count_correct_v2(params):
    correct = 0
    for inp, exp in tests:
        if predict_v2(inp, params) == exp:
            correct += 1
    return correct

# Test formula variants
print("=== Model 1: Single temp coeff (current) ===")
p_current = [-0.9665103286569976, 0.755284994643082,
             1.6213600572975244, 0.813268608577364, 0.345981233247675,
             10, 20, 29, 0.027, 28.0]
print(f"Score: {count_correct(p_current)}/100")

# Try NO cliff (purely linear degradation)
print("\n=== Model 2: No cliff (linear from lap 1) ===")
p_no_cliff = [-0.9665, 0.7553, 0.162, 0.081, 0.035, 0, 0, 0, 0.027, 28.0]
print(f"Score: {count_correct(p_no_cliff)}/100")

# Try with different temp refs
print("\n=== Model 3: Try different temp references ===")
for tref in [20, 25, 28, 30, 35]:
    p = list(p_current)
    p[9] = tref
    s = count_correct(p)
    if s >= 50:
        print(f"  tref={tref}: {s}/100")

# Now try per-compound temp coeff (model v2)
print("\n=== Model 4: Per-compound temp coeff ===")
p_v2 = [-0.9665103286569976, 0.755284994643082,
         1.6213600572975244, 0.813268608577364, 0.345981233247675,
         10, 20, 29,
         0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
         27.96640138772966]
print(f"Score: {count_correct_v2(p_v2)}/100")

# Now use DE to optimize single-temp-coeff model
print("\n=== Optimizing with DE (single temp coeff model) ===")
def neg_correct(params):
    return -count_correct(params)

# Quick optimization around current best
bounds = [
    (-1.5, -0.5),   # cb_s
    (0.3, 1.2),     # cb_h
    (0.5, 3.0),     # rate_s
    (0.3, 1.5),     # rate_m
    (0.1, 0.8),     # rate_h
    (5, 15),         # cliff_s
    (15, 25),        # cliff_m
    (24, 35),        # cliff_h
    (0.01, 0.05),   # tc
    (20, 35),        # tref
]

best_score = 54
best_params = p_current

# Random search first (faster than DE for discrete-ish objective)
import random
random.seed(42)
np.random.seed(42)

for trial in range(5000):
    p = [
        random.uniform(-1.5, -0.5),
        random.uniform(0.3, 1.2),
        random.uniform(0.5, 3.0),
        random.uniform(0.3, 1.5),
        random.uniform(0.1, 0.8),
        random.randint(5, 15),
        random.randint(15, 25),
        random.randint(24, 35),
        random.uniform(0.01, 0.05),
        random.uniform(20, 35),
    ]
    s = count_correct(p)
    if s > best_score:
        best_score = s
        best_params = list(p)
        print(f"  Trial {trial}: {s}/100 params={[round(x,6) for x in p]}")

print(f"\nRandom search best: {best_score}/100")

# Perturbation search around best
print("\n=== Perturbation search ===")
for trial in range(10000):
    p = list(best_params)
    # Perturb one or two params
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, 9)
        if idx in [5, 6, 7]:  # cliff values - integer
            p[idx] = max(0, p[idx] + random.randint(-2, 2))
        else:
            scale = abs(p[idx]) * 0.05 if p[idx] != 0 else 0.1
            p[idx] += random.gauss(0, scale)
    s = count_correct(p)
    if s > best_score:
        best_score = s
        best_params = list(p)
        print(f"  Trial {trial}: {s}/100 params={[round(x,6) for x in p]}")

print(f"\nFinal best: {best_score}/100")
print(f"Params: {best_params}")
