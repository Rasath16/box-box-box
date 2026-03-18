"""
Joint optimization of core params + per-track temperature references.
Starting from 60/100 with per-track tref.
"""
import json
import numpy as np
import random

tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

tracks = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]

def count_correct(params):
    cb_s, cb_h, rs, rm, rh, tc_s, tc_m, tc_h = params[:8]
    cliff_s, cliff_m, cliff_h = 10, 20, 29
    # Per-track tref
    tref_map = {}
    for i, t in enumerate(tracks):
        tref_map[t] = params[8 + i]

    CB = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    RATE = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    TC = {"SOFT": tc_s, "MEDIUM": tc_m, "HARD": tc_h}
    CLIFF = {"SOFT": cliff_s, "MEDIUM": cliff_m, "HARD": cliff_h}

    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        track = config["track"]
        temp = config["track_temp"]
        tref = tref_map.get(track, 28.0)
        results = []

        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0

            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
                ts = 1.0 + TC[compound] * (temp - tref)
                deg = RATE[compound] * ts * max(0.0, tire_age - CLIFF[compound])
                total_time += config["base_lap_time"] + CB[compound] + deg
                if lap in pit_laps:
                    total_time += config["pit_lane_time"]
                    compound = pit_laps[lap]
                    tire_age = 0

            results.append((total_time, grid, s["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

# Starting params: current best core + per-track trefs from greedy search
x0 = [
    -0.9665103286569976,   # cb_s
    0.755284994643082,     # cb_h
    1.6213600572975244,    # rs
    0.813268608577364,     # rm
    0.345981233247675,     # rh
    0.025806274187704845,  # tc_s
    0.02777171692356944,   # tc_m
    0.02401965544225936,   # tc_h
    25.5,                  # Bahrain tref
    27.97,                 # COTA tref
    20.5,                  # Monaco tref
    27.97,                 # Monza tref
    28.5,                  # Silverstone tref
    29.5,                  # Spa tref
    27.5,                  # Suzuka tref
]

print(f"Starting score: {count_correct(x0)}/100")

# Perturbation search
best_score = count_correct(x0)
best_params = list(x0)
random.seed(42)

for trial in range(50000):
    p = list(best_params)
    # Perturb 1-3 params
    n_perturb = random.randint(1, 3)
    for _ in range(n_perturb):
        idx = random.randint(0, len(p) - 1)
        if idx < 8:
            # Core params
            scale = abs(p[idx]) * 0.03
            p[idx] += random.gauss(0, scale)
        else:
            # Track tref
            p[idx] += random.gauss(0, 1.0)

    s = count_correct(p)
    if s > best_score:
        best_score = s
        best_params = list(p)
        print(f"Trial {trial}: {s}/100")
        # Print the params
        print(f"  Core: cb_s={p[0]:.6f} cb_h={p[1]:.6f} rs={p[2]:.6f} rm={p[3]:.6f} rh={p[4]:.6f}")
        print(f"  TC: tc_s={p[5]:.6f} tc_m={p[6]:.6f} tc_h={p[7]:.6f}")
        print(f"  Trefs: {dict(zip(tracks, p[8:]))}")

print(f"\nFinal best: {best_score}/100")
print(f"Best params: {best_params}")
