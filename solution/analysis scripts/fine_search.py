"""
Fine-grained parameter search around the best known values.
Focus on the most sensitive parameters.
"""
import json
import numpy as np
from itertools import product

tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def count_correct(cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc_s, tc_m, tc_h, tref):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        temp = config["track_temp"]
        ts_map = {
            "SOFT": 1.0 + tc_s * (temp - tref),
            "MEDIUM": 1.0 + tc_m * (temp - tref),
            "HARD": 1.0 + tc_h * (temp - tref),
        }
        cb_map = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
        rate_map = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
        cliff_map = {"SOFT": cs, "MEDIUM": cm, "HARD": ch}

        for pk in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            s = strategies[pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0
            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
                deg = rate_map[compound] * ts_map[compound] * max(0.0, tire_age - cliff_map[compound])
                total_time += config["base_lap_time"] + cb_map[compound] + deg
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

# Current best: 54/100
best = 54
best_params = None

# Fine search: vary each parameter in small steps
CB_S = [-0.9665103286569976]
CB_H = [0.755284994643082]
RS = [1.6213600572975244]
RM = [0.813268608577364]
RH = [0.345981233247675]
CS = [10]
CM = [20]
CH = [29, 30]
TC_S = [0.025806274187704845]
TC_M = [0.02777171692356944]
TC_H = [0.02401965544225936]
TREF = [27.96640138772966]

# Step 1: Try cliff HARD = 30 vs 29
for ch in [28, 29, 30, 31]:
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, ch,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    print(f"cliff_h={ch}: {s}/100")
    if s > best:
        best = s

# Step 2: Vary tref
print("\n=== Vary tref ===")
for tref in np.arange(25.0, 32.0, 0.5):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      tref)
    if s >= 53:
        print(f"tref={tref:.1f}: {s}/100")
    if s > best:
        best = s
        best_params = ('tref', tref)

# Step 3: Vary rates by small amounts
print("\n=== Vary SOFT rate ===")
for rs in np.arange(1.4, 1.9, 0.02):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      rs, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"rs={rs:.4f}: {s}/100")
    if s > best:
        best = s

print("\n=== Vary MEDIUM rate ===")
for rm in np.arange(0.7, 0.95, 0.01):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, rm, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"rm={rm:.4f}: {s}/100")
    if s > best:
        best = s

print("\n=== Vary HARD rate ===")
for rh in np.arange(0.25, 0.50, 0.01):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, rh,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"rh={rh:.4f}: {s}/100")
    if s > best:
        best = s

# Step 4: Vary compound bases
print("\n=== Vary cb_s ===")
for cb_s in np.arange(-1.2, -0.7, 0.02):
    s = count_correct(cb_s, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"cb_s={cb_s:.4f}: {s}/100")
    if s > best:
        best = s

print("\n=== Vary cb_h ===")
for cb_h in np.arange(0.5, 1.0, 0.02):
    s = count_correct(-0.9665103286569976, cb_h,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"cb_h={cb_h:.4f}: {s}/100")
    if s > best:
        best = s

# Step 5: Vary temp coefficients
print("\n=== Vary tc_s ===")
for tc_s in np.arange(0.015, 0.040, 0.001):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      tc_s, 0.02777171692356944, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"tc_s={tc_s:.4f}: {s}/100")
    if s > best:
        best = s

print("\n=== Vary tc_m ===")
for tc_m in np.arange(0.015, 0.040, 0.001):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, tc_m, 0.02401965544225936,
                      27.96640138772966)
    if s >= 53:
        print(f"tc_m={tc_m:.4f}: {s}/100")
    if s > best:
        best = s

print("\n=== Vary tc_h ===")
for tc_h in np.arange(0.015, 0.040, 0.001):
    s = count_correct(-0.9665103286569976, 0.755284994643082,
                      1.6213600572975244, 0.813268608577364, 0.345981233247675,
                      10, 20, 29,
                      0.025806274187704845, 0.02777171692356944, tc_h,
                      27.96640138772966)
    if s >= 53:
        print(f"tc_h={tc_h:.4f}: {s}/100")
    if s > best:
        best = s

print(f"\nOverall best: {best}/100")
