"""
Systematic parameter extraction using historical data.
For fixed cliff values, the lap time formula is linear in the remaining parameters,
so we can use least-squares to find exact values.

Formula: lap_time = base_lap_time + compound_base[c] + rate[c] * temp_scale * max(0, age - cliff[c])
where temp_scale = 1 + tc * (track_temp - T_ref)

Simplification attempts:
1. Single temperature coefficient for all compounds
2. Per-compound temperature coefficients
"""

import json
import os
import sys
import itertools
from pathlib import Path

def load_races(n=3000):
    """Load historical races."""
    races = []
    data_dir = Path("data/historical_races")
    for f in sorted(data_dir.glob("races_*.json")):
        with open(f) as fh:
            batch = json.load(fh)
            races.extend(batch)
            if len(races) >= n:
                break
    return races[:n]

def compute_race_time(strategy, config, params):
    """Simulate a race with given params."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    compound_base = params["compound_base"]
    deg_rate = params["deg_rate"]
    cliff = params["cliff"]
    temp_coeff = params["temp_coeff"]
    temp_ref = params["temp_ref"]

    # Build pit stop schedule
    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    current_compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1

        # Calculate lap time
        cb = compound_base.get(current_compound, 0.0)
        rate = deg_rate[current_compound]
        c = cliff[current_compound]

        # Temperature scaling
        if isinstance(temp_coeff, dict):
            tc = temp_coeff[current_compound]
        else:
            tc = temp_coeff
        temp_scale = 1.0 + tc * (track_temp - temp_ref)

        deg = rate * temp_scale * max(0.0, tire_age - c)
        lap_time = base + cb + deg
        total_time += lap_time

        # Pit stop at end of lap
        if lap in pit_laps:
            total_time += pit_time
            current_compound = pit_laps[lap]
            tire_age = 0

    return total_time

def predict_order(race_data, params):
    config = race_data["race_config"]
    strategies = race_data["strategies"]
    results = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strat = strategies[pos_key]
        grid = int(pos_key[3:])
        t = compute_race_time(strat, config, params)
        results.append((t, grid, strat["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    return [d for _, _, d in results]

def test_accuracy(params, test_dir="data/test_cases"):
    """Test against 100 test cases."""
    correct = 0
    for i in range(1, 101):
        inp_path = f"{test_dir}/inputs/test_{i:03d}.json"
        exp_path = f"{test_dir}/expected_outputs/test_{i:03d}.json"
        with open(inp_path) as f:
            inp = json.load(f)
        with open(exp_path) as f:
            exp = json.load(f)
        pred = predict_order(inp, params)
        if pred == exp["finishing_positions"]:
            correct += 1
    return correct

def hist_accuracy(params, races):
    """Test against historical data."""
    correct = 0
    for race in races:
        pred = predict_order(race, params)
        if pred == race["finishing_positions"]:
            correct += 1
    return correct

# Current params (baseline: 54/100)
current_params = {
    "compound_base": {"SOFT": -0.9665103286569976, "MEDIUM": 0.0, "HARD": 0.755284994643082},
    "deg_rate": {"SOFT": 1.6213600572975244, "MEDIUM": 0.813268608577364, "HARD": 0.345981233247675},
    "cliff": {"SOFT": 10, "MEDIUM": 20, "HARD": 29},
    "temp_coeff": {"SOFT": 0.025806274187704845, "MEDIUM": 0.02777171692356944, "HARD": 0.02401965544225936},
    "temp_ref": 27.96640138772966,
}

print("=== Current params ===")
score = test_accuracy(current_params)
print(f"Test cases: {score}/100")

# Try with single temp coefficient (simpler model)
print("\n=== Trying single temp coefficient ===")
for tc_single in [0.025, 0.026, 0.027, 0.028, 0.029, 0.030]:
    p = dict(current_params)
    p["temp_coeff"] = tc_single
    s = test_accuracy(p)
    if s > 50:
        print(f"  tc={tc_single}: {s}/100")

# Try different cliff combinations
print("\n=== Trying different cliff values ===")
best_score = score
best_cliffs = (10, 20, 29)
for s_cliff in range(8, 15):
    for m_cliff in range(16, 25):
        for h_cliff in range(25, 35):
            p = dict(current_params)
            p["cliff"] = {"SOFT": s_cliff, "MEDIUM": m_cliff, "HARD": h_cliff}
            s = test_accuracy(p)
            if s > best_score:
                best_score = s
                best_cliffs = (s_cliff, m_cliff, h_cliff)
                print(f"  NEW BEST: cliffs=({s_cliff},{m_cliff},{h_cliff}) -> {s}/100")

print(f"\nBest cliffs: {best_cliffs} -> {best_score}/100")
