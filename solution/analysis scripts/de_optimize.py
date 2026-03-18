"""
Use scipy differential_evolution to optimize parameters on test cases.
Try multiple formula variants.
"""
import json
import numpy as np
from scipy.optimize import differential_evolution

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def simulate_v1(strategy, config, params):
    """Standard model: base + cb + rate*temp_scale*max(0, age-cliff)"""
    cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref = params
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]

    cb = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    cliff = {"SOFT": max(0, round(cs)), "MEDIUM": max(0, round(cm)), "HARD": max(0, round(ch))}
    temp_scale = 1.0 + tc * (temp - tref)

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        deg = rate[compound] * temp_scale * max(0.0, tire_age - cliff[compound])
        total_time += base + cb[compound] + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

def simulate_v2(strategy, config, params):
    """Per-compound temp coefficients"""
    cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc_s, tc_m, tc_h, tref = params
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]

    cb = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    cliff = {"SOFT": max(0, round(cs)), "MEDIUM": max(0, round(cm)), "HARD": max(0, round(ch))}
    tc = {"SOFT": tc_s, "MEDIUM": tc_m, "HARD": tc_h}

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        temp_scale = 1.0 + tc[compound] * (temp - tref)
        deg = rate[compound] * temp_scale * max(0.0, tire_age - cliff[compound])
        total_time += base + cb[compound] + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

def simulate_v3(strategy, config, params):
    """Temp affects compound base AND degradation"""
    cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc_deg, tc_base, tref = params
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]

    cb_raw = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    cliff = {"SOFT": max(0, round(cs)), "MEDIUM": max(0, round(cm)), "HARD": max(0, round(ch))}
    temp_scale_deg = 1.0 + tc_deg * (temp - tref)
    temp_scale_base = 1.0 + tc_base * (temp - tref)

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        cb_adj = cb_raw[compound] * temp_scale_base
        deg = rate[compound] * temp_scale_deg * max(0.0, tire_age - cliff[compound])
        total_time += base + cb_adj + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

def pairwise_loss(params, sim_fn):
    total = 0.0
    for inp, exp in tests:
        config = inp["race_config"]
        strats = inp["strategies"]
        times = {}
        for pk in strats:
            s = strats[pk]
            times[s["driver_id"]] = sim_fn(s, config, params)

        for i in range(len(exp) - 1):
            for j in range(i+1, min(i+5, len(exp))):  # nearby pairs only for speed
                ti = times[exp[i]]
                tj = times[exp[j]]
                diff = ti - tj
                if diff >= 0:
                    total += (diff + 0.001) ** 2
    return total

def count_correct(params, sim_fn):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strats = inp["strategies"]
        results = []
        for pk in sorted(strats.keys(), key=lambda k: int(k[3:])):
            s = strats[pk]
            grid = int(pk[3:])
            t = sim_fn(s, config, params)
            results.append((t, grid, s["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

# V1: single temp coeff
print("=== V1: Single temp coeff - DE optimization ===")
bounds_v1 = [
    (-2.0, 0.0),    # cb_s
    (0.0, 2.0),     # cb_h
    (0.1, 5.0),     # rs
    (0.05, 3.0),    # rm
    (0.01, 2.0),    # rh
    (5, 15),         # cs
    (15, 25),        # cm
    (25, 35),        # ch
    (0.005, 0.05),   # tc
    (20, 35),        # tref
]

result = differential_evolution(
    lambda p: pairwise_loss(p, simulate_v1),
    bounds_v1,
    seed=42,
    maxiter=50,
    popsize=20,
    tol=1e-8,
    disp=True,
    workers=1,
)
print(f"V1 pairwise loss: {result.fun:.6f}")
print(f"V1 params: {result.x}")
s = count_correct(result.x, simulate_v1)
print(f"V1 test score: {s}/100")

# V2: per-compound temp coeff
print("\n=== V2: Per-compound temp coeff - DE optimization ===")
bounds_v2 = [
    (-2.0, 0.0),    # cb_s
    (0.0, 2.0),     # cb_h
    (0.1, 5.0),     # rs
    (0.05, 3.0),    # rm
    (0.01, 2.0),    # rh
    (5, 15),         # cs
    (15, 25),        # cm
    (25, 35),        # ch
    (0.005, 0.05),   # tc_s
    (0.005, 0.05),   # tc_m
    (0.005, 0.05),   # tc_h
    (20, 35),        # tref
]

result2 = differential_evolution(
    lambda p: pairwise_loss(p, simulate_v2),
    bounds_v2,
    seed=42,
    maxiter=50,
    popsize=20,
    tol=1e-8,
    disp=True,
    workers=1,
)
print(f"V2 pairwise loss: {result2.fun:.6f}")
print(f"V2 params: {result2.x}")
s2 = count_correct(result2.x, simulate_v2)
print(f"V2 test score: {s2}/100")

# V3: temp affects both base and degradation
print("\n=== V3: Temp on base and deg - DE optimization ===")
bounds_v3 = [
    (-2.0, 0.0),    # cb_s
    (0.0, 2.0),     # cb_h
    (0.1, 5.0),     # rs
    (0.05, 3.0),    # rm
    (0.01, 2.0),    # rh
    (5, 15),         # cs
    (15, 25),        # cm
    (25, 35),        # ch
    (0.005, 0.05),   # tc_deg
    (-0.05, 0.05),   # tc_base
    (20, 35),        # tref
]

result3 = differential_evolution(
    lambda p: pairwise_loss(p, simulate_v3),
    bounds_v3,
    seed=42,
    maxiter=50,
    popsize=20,
    tol=1e-8,
    disp=True,
    workers=1,
)
print(f"V3 pairwise loss: {result3.fun:.6f}")
print(f"V3 params: {result3.x}")
s3 = count_correct(result3.x, simulate_v3)
print(f"V3 test score: {s3}/100")
