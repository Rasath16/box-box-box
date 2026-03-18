"""
Refine around the 58/100 params:
cb_s=-1.0, cb_h=0.8, rs=1.5, rm=0.75, rh=0.4, cs=10, cm=20, ch=30, tc=0.029, tref=24.5
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

def count_correct(cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        temp = config["track_temp"]
        ts = 1.0 + tc * (temp - tref)
        results = []
        CB = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
        RATE = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
        CLIFF = {"SOFT": cs, "MEDIUM": cm, "HARD": ch}
        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0
            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
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

# Base: 58/100
base_params = [-1.0, 0.8, 1.5, 0.75, 0.4, 10, 20, 30, 0.029, 24.5]
print(f"Base: {count_correct(*base_params)}/100")

# Fine grid around each param
print("\n=== Fine grid search ===")
best = 58
best_params = list(base_params)

# Vary tc and tref more finely
print("tc/tref:")
for tc in np.arange(0.025, 0.035, 0.001):
    for tref in np.arange(22, 28, 0.25):
        s = count_correct(-1.0, 0.8, 1.5, 0.75, 0.4, 10, 20, 30, tc, tref)
        if s > best:
            best = s
            best_params = [-1.0, 0.8, 1.5, 0.75, 0.4, 10, 20, 30, tc, tref]
            print(f"  {s}/100: tc={tc:.3f} tref={tref:.2f}")

# Vary rates
print("rates:")
tc_b, tref_b = best_params[8], best_params[9]
for rs in np.arange(1.3, 1.8, 0.025):
    for rm in np.arange(0.6, 0.9, 0.025):
        for rh in np.arange(0.3, 0.5, 0.025):
            s = count_correct(-1.0, 0.8, rs, rm, rh, 10, 20, 30, tc_b, tref_b)
            if s > best:
                best = s
                best_params = [-1.0, 0.8, rs, rm, rh, 10, 20, 30, tc_b, tref_b]
                print(f"  {s}/100: rs={rs:.3f} rm={rm:.3f} rh={rh:.3f}")

# Vary compound bases
print("compound bases:")
rs_b, rm_b, rh_b = best_params[2], best_params[3], best_params[4]
for cb_s in np.arange(-1.3, -0.6, 0.05):
    for cb_h in np.arange(0.5, 1.1, 0.05):
        s = count_correct(cb_s, cb_h, rs_b, rm_b, rh_b, 10, 20, 30, tc_b, tref_b)
        if s > best:
            best = s
            best_params = [cb_s, cb_h, rs_b, rm_b, rh_b, 10, 20, 30, tc_b, tref_b]
            print(f"  {s}/100: cb_s={cb_s:.2f} cb_h={cb_h:.2f}")

# Vary cliffs
print("cliffs:")
cb_s_b, cb_h_b = best_params[0], best_params[1]
for cs in range(8, 14):
    for cm in range(17, 24):
        for ch in range(27, 34):
            s = count_correct(cb_s_b, cb_h_b, rs_b, rm_b, rh_b, cs, cm, ch, tc_b, tref_b)
            if s > best:
                best = s
                best_params = [cb_s_b, cb_h_b, rs_b, rm_b, rh_b, cs, cm, ch, tc_b, tref_b]
                print(f"  {s}/100: cliffs=({cs},{cm},{ch})")

print(f"\nFinal fine-grid best: {best}/100")
print(f"Params: {best_params}")

# Perturbation search from best
print("\n=== Perturbation search ===")
random.seed(42)
for trial in range(100000):
    p = list(best_params)
    n = random.randint(1, 3)
    for _ in range(n):
        idx = random.randint(0, 9)
        if idx in [5, 6, 7]:
            p[idx] = max(0, p[idx] + random.randint(-1, 1))
        elif idx == 8:  # tc
            p[idx] += random.gauss(0, 0.002)
        elif idx == 9:  # tref
            p[idx] += random.gauss(0, 0.5)
        else:
            scale = abs(p[idx]) * 0.02
            p[idx] += random.gauss(0, scale)

    s = count_correct(*p)
    if s > best:
        best = s
        best_params = list(p)
        print(f"Trial {trial}: {s}/100 params={[round(x,6) for x in p]}")

print(f"\nFinal best: {best}/100")
print(f"Params: {best_params}")
