"""
Try per-track degradation multiplier.
The accuracy varies hugely by track (Monaco 31% vs Monza 74%).
The hint says "only variables in PROBLEM_STATEMENT.md affect outcomes"
and track IS listed.
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

CB = {"SOFT": -0.9665103286569976, "MEDIUM": 0.0, "HARD": 0.755284994643082}
RATE = {"SOFT": 1.6213600572975244, "MEDIUM": 0.813268608577364, "HARD": 0.345981233247675}
CLIFF = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}
TC = {"SOFT": 0.025806274187704845, "MEDIUM": 0.02777171692356944, "HARD": 0.02401965544225936}
TREF = 27.96640138772966

def count_correct(track_mult=None, cb_mult=None):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        track = config["track"]
        temp = config["track_temp"]
        results = []

        tm = track_mult.get(track, 1.0) if track_mult else 1.0
        cm = cb_mult.get(track, 1.0) if cb_mult else 1.0

        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0

            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
                ts = 1.0 + TC[compound] * (temp - TREF)
                deg = RATE[compound] * ts * tm * max(0.0, tire_age - CLIFF[compound])
                total_time += config["base_lap_time"] + CB[compound] * cm + deg
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

# Baseline
print(f"Baseline (no track effect): {count_correct()}/100")

# Try degradation multiplier per track
tracks = ["Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka"]

# Grid search for a single track multiplier applied to all tracks
print("\n=== Single track degradation multiplier ===")
for mult in np.arange(0.7, 1.5, 0.05):
    tm = {t: mult for t in tracks}
    s = count_correct(track_mult=tm)
    if s >= 53:
        print(f"  mult={mult:.2f}: {s}/100")

# Try different multiplier per track
# First, let's see what happens if we scale just one track
print("\n=== Per-track scaling (one at a time) ===")
for track in tracks:
    for mult in np.arange(0.5, 2.0, 0.1):
        tm = {track: mult}
        s = count_correct(track_mult=tm)
        if s > 54:
            print(f"  {track}={mult:.1f}: {s}/100")

# Try compound base multiplier per track
print("\n=== Per-track CB scaling (one at a time) ===")
for track in tracks:
    for mult in np.arange(0.5, 2.0, 0.1):
        cm = {track: mult}
        s = count_correct(cb_mult=cm)
        if s > 54:
            print(f"  {track} cb_mult={mult:.1f}: {s}/100")

# Try different formula: track affects the CLIFF value
print("\n=== Track-dependent cliff offset ===")
def count_correct_cliff_offset(cliff_offset):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        track = config["track"]
        temp = config["track_temp"]
        results = []

        co = cliff_offset.get(track, 0)

        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in s["pit_stops"]}
            compound = s["starting_tire"]
            tire_age = 0
            total_time = 0.0

            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
                ts = 1.0 + TC[compound] * (temp - TREF)
                cliff = CLIFF[compound] + co
                deg = RATE[compound] * ts * max(0.0, tire_age - cliff)
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

for track in tracks:
    for offset in range(-5, 6):
        s = count_correct_cliff_offset({track: offset})
        if s > 54:
            print(f"  {track} cliff_offset={offset}: {s}/100")

# What about a temperature reference that varies by track?
print("\n=== Per-track temperature reference ===")
def count_correct_tref(tref_map):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        track = config["track"]
        temp = config["track_temp"]
        tref = tref_map.get(track, TREF)
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

for track in tracks:
    for tref in np.arange(20, 36, 2):
        s = count_correct_tref({track: tref})
        if s > 54:
            print(f"  {track} tref={tref:.0f}: {s}/100")
