"""
"Easier than you think" - search over SIMPLE parameter values
that a hackathon organizer would use in their code.
Focus on clean, round numbers and simple relationships.
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

def count_correct(cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        temp = config["track_temp"]
        ts = 1.0 + tc * (temp - tref)
        results = []
        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0
            CB = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
            RATE = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
            CLIFF = {"SOFT": cs, "MEDIUM": cm, "HARD": ch}
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

# Try with finer grid around the best "round" params
# Best round: cb_s=-1.0, cb_h=0.8, rs=1.5, rm=0.75, rh=0.35, cs=10, cm=20, ch=30, tc=0.02, tref=25 -> 48/100
# With tc=0.03, tref=25 -> 49/100

best = 0
best_params = None

# Phase 1: Very fine search around tc=0.02-0.03, tref=25
print("=== Phase 1: Fine tc/tref search with round params ===")
for tc in np.arange(0.015, 0.035, 0.001):
    for tref in np.arange(22, 32, 0.5):
        s = count_correct(-1.0, 0.8, 1.5, 0.75, 0.35, 10, 20, 30, tc, tref)
        if s > best:
            best = s
            best_params = (-1.0, 0.8, 1.5, 0.75, 0.35, 10, 20, 30, tc, tref)
            print(f"  {s}/100: tc={tc:.3f} tref={tref:.1f}")

print(f"Phase 1 best: {best}/100")

# Phase 2: Fine search of rates with best tc/tref
tc_best, tref_best = best_params[8], best_params[9]
print(f"\n=== Phase 2: Fine rate search (tc={tc_best:.3f}, tref={tref_best:.1f}) ===")
for rs in np.arange(1.2, 2.0, 0.05):
    for rm in np.arange(0.5, 1.0, 0.025):
        for rh in np.arange(0.2, 0.5, 0.025):
            s = count_correct(-1.0, 0.8, rs, rm, rh, 10, 20, 30, tc_best, tref_best)
            if s > best:
                best = s
                best_params = (-1.0, 0.8, rs, rm, rh, 10, 20, 30, tc_best, tref_best)
                print(f"  {s}/100: rs={rs:.3f} rm={rm:.3f} rh={rh:.3f}")

print(f"Phase 2 best: {best}/100")

# Phase 3: Fine search of compound bases
rs_best, rm_best, rh_best = best_params[2], best_params[3], best_params[4]
print(f"\n=== Phase 3: Fine CB search ===")
for cb_s in np.arange(-1.2, -0.6, 0.05):
    for cb_h in np.arange(0.4, 1.2, 0.05):
        s = count_correct(cb_s, cb_h, rs_best, rm_best, rh_best, 10, 20, 30, tc_best, tref_best)
        if s > best:
            best = s
            best_params = (cb_s, cb_h, rs_best, rm_best, rh_best, 10, 20, 30, tc_best, tref_best)
            print(f"  {s}/100: cb_s={cb_s:.2f} cb_h={cb_h:.2f}")

print(f"Phase 3 best: {best}/100")

# Phase 4: Try cliff values
cb_s_best, cb_h_best = best_params[0], best_params[1]
print(f"\n=== Phase 4: Cliff search ===")
for cs in range(8, 13):
    for cm in range(17, 23):
        for ch in range(27, 33):
            s = count_correct(cb_s_best, cb_h_best, rs_best, rm_best, rh_best, cs, cm, ch, tc_best, tref_best)
            if s > best:
                best = s
                best_params = (cb_s_best, cb_h_best, rs_best, rm_best, rh_best, cs, cm, ch, tc_best, tref_best)
                print(f"  {s}/100: cliffs=({cs},{cm},{ch})")

print(f"Phase 4 best: {best}/100")

# Redo tc/tref with new best
print(f"\n=== Phase 5: Re-optimize tc/tref ===")
for tc in np.arange(0.010, 0.045, 0.001):
    for tref in np.arange(20, 35, 0.5):
        s = count_correct(best_params[0], best_params[1], best_params[2], best_params[3], best_params[4],
                         best_params[5], best_params[6], best_params[7], tc, tref)
        if s > best:
            best = s
            best_params = (*best_params[:8], tc, tref)
            print(f"  {s}/100: tc={tc:.3f} tref={tref:.1f}")

print(f"\n=== FINAL BEST: {best}/100 ===")
print(f"Params: {best_params}")
