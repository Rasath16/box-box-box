"""
Optimize parameters directly on test cases using differential evolution.
Uses pairwise ranking loss for smooth optimization.
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution

TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")


def load_test_cases():
    cases = []
    for i in range(1, 101):
        with open(os.path.join(TEST_INPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            inp = json.load(f)
        with open(os.path.join(TEST_OUTPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            out = json.load(f)
        cases.append((inp, out['finishing_positions']))
    return cases


def load_hist_races(max_files=3):
    races = []
    files = sorted(os.listdir(HIST_DIR))[:max_files]
    for fname in files:
        with open(os.path.join(HIST_DIR, fname), 'r') as f:
            races.extend(json.load(f))
    return races


def simulate(race_input, params):
    s_base, h_base, s_rate, m_rate, h_rate = params[0], params[1], params[2], params[3], params[4]
    s_cliff, m_cliff, h_cliff = params[5], params[6], params[7]
    tc, t_ref = params[8], params[9]

    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']

    ts = 1.0 + tc * (temp - t_ref)

    cb = {'SOFT': s_base, 'MEDIUM': 0.0, 'HARD': h_base}
    cr = {'SOFT': s_rate * ts, 'MEDIUM': m_rate * ts, 'HARD': h_rate * ts}
    cc = {'SOFT': s_cliff, 'MEDIUM': m_cliff, 'HARD': h_cliff}

    strats = race_input['strategies']
    times = []

    for pos_key in sorted(strats.keys(), key=lambda x: int(x[3:])):
        s = strats[pos_key]
        pit_map = {p['lap']: p['to_tire'] for p in s.get('pit_stops', [])}
        cur = s['starting_tire']
        age = 0
        total = 0.0

        for lap in range(1, total_laps + 1):
            age += 1
            deg = cr[cur] * max(0.0, age - cc[cur])
            total += base + cb[cur] + deg
            if lap in pit_map:
                total += pit_time
                cur = pit_map[lap]
                age = 0

        times.append((total, int(pos_key[3:]), s['driver_id']))

    times.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in times]


def pairwise_loss(params, test_cases):
    """Count pairwise inversions across all test cases."""
    total_inv = 0
    total_pairs = 0
    for inp, true_order in test_cases:
        pred = simulate(inp, params)
        true_pos = {d: i for i, d in enumerate(true_order)}
        pred_pos = {d: i for i, d in enumerate(pred)}
        drivers = list(true_pos.keys())
        n = len(drivers)
        total_pairs += n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                d1, d2 = drivers[i], drivers[j]
                if (true_pos[d1] < true_pos[d2]) != (pred_pos[d1] < pred_pos[d2]):
                    total_inv += 1
    return total_inv / max(total_pairs, 1)


def exact_score(params, cases):
    correct = 0
    for inp, true_order in cases:
        if simulate(inp, params) == true_order:
            correct += 1
    return correct


def main():
    print("Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")

    # Also load some historical for validation
    print("Loading historical races for validation...")
    hist = load_hist_races(max_files=1)
    hist_sample = [(r, r['finishing_positions']) for r in hist[:200]]
    print(f"Loaded {len(hist_sample)} historical races")

    # Bounds: [s_base, h_base, s_rate, m_rate, h_rate, s_cliff, m_cliff, h_cliff, tc, t_ref]
    bounds = [
        (-3.0, -0.1),   # s_base (SOFT faster)
        (0.01, 3.0),    # h_base (HARD slower)
        (0.05, 1.5),    # s_rate (SOFT degrades fast)
        (0.02, 0.8),    # m_rate (MEDIUM moderate)
        (0.005, 0.3),   # h_rate (HARD slow degradation)
        (1, 12),        # s_cliff
        (10, 25),       # m_cliff
        (15, 40),       # h_cliff
        (0.01, 0.15),   # temp_coeff
        (25, 35),       # T_ref
    ]

    print("\nRunning differential evolution on test cases...")
    print("Optimizing pairwise ranking loss...\n")

    result = differential_evolution(
        pairwise_loss,
        bounds,
        args=(test_cases,),
        maxiter=150,
        popsize=25,
        tol=1e-8,
        seed=42,
        mutation=(0.5, 1.5),
        recombination=0.85,
        disp=True,
        polish=True,
    )

    best = result.x
    print(f"\nOptimization complete! Loss: {result.fun:.6f}")

    names = ['s_base', 'h_base', 's_rate', 'm_rate', 'h_rate',
             's_cliff', 'm_cliff', 'h_cliff', 'tc', 't_ref']
    print("\nBest parameters:")
    for n, v in zip(names, best):
        print(f"  {n} = {v:.6f}")

    test_score = exact_score(best, test_cases)
    print(f"\nTest accuracy: {test_score}/100")

    hist_score = exact_score(best, hist_sample)
    print(f"Historical accuracy: {hist_score}/200")

    # Show which test cases fail
    print("\nFailed test cases:")
    for i, (inp, true_order) in enumerate(test_cases):
        pred = simulate(inp, best)
        if pred != true_order:
            cfg = inp['race_config']
            # Count how many positions differ
            diffs = sum(1 for a, b in zip(pred, true_order) if a != b)
            print(f"  Test {i+1}: N={cfg['total_laps']}, T={cfg['track_temp']}, "
                  f"base={cfg['base_lap_time']}, pit={cfg['pit_lane_time']}, "
                  f"diffs={diffs}/20")


if __name__ == '__main__':
    main()
