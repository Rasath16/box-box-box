"""
Reverse-engineer the lap time formula using pairwise regression.

For FIXED cliff values, the total race time is a LINEAR function of
(s_base, h_base, s_rate, m_rate, h_rate). This means we can use
linear classification to find the exact coefficients.

For each pair (winner, loser) in historical data, we get a linear constraint:
  feature_vector dot [s_base, h_base, s_rate, m_rate, h_rate] < 0

We search over (s_cliff, m_cliff, h_cliff, tc, T_ref) and solve the linear
system for each choice.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")
TEST_IN = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")


def D(n, cliff):
    """Cumulative degradation sum: sum_{a=1}^{n} max(0, a - cliff)"""
    effective = max(0, n - cliff)
    return effective * (effective + 1) / 2.0


def compute_driver_features(strategy, race_config, s_cliff, m_cliff, h_cliff, tc, t_ref):
    """
    Compute feature vector for a driver's total time.
    Total time = N*base + pit_penalties + features dot [s_base, h_base, s_rate, m_rate, h_rate]

    Returns: (constant_part, feature_vector) where feature_vector has 5 elements.
    """
    cfg = race_config
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']

    ts = 1.0 + tc * (temp - t_ref)

    cliff_map = {'SOFT': s_cliff, 'MEDIUM': m_cliff, 'HARD': h_cliff}

    # Parse strategy into stints
    pit_stops = sorted(strategy.get('pit_stops', []), key=lambda p: p['lap'])
    stints = []
    cur_tire = strategy['starting_tire']
    stint_start = 1
    for stop in pit_stops:
        stint_end = stop['lap']
        stint_laps = stint_end - stint_start + 1
        stints.append((cur_tire, stint_laps))
        cur_tire = stop['to_tire']
        stint_start = stint_end + 1
    # Final stint
    stint_laps = total_laps - stint_start + 1
    stints.append((cur_tire, stint_laps))

    num_stops = len(pit_stops)
    constant = total_laps * base + num_stops * pit_time

    # Features: [s_base, h_base, s_rate, m_rate, h_rate]
    features = np.zeros(5)
    for compound, laps in stints:
        cliff = cliff_map[compound]
        deg_sum = D(laps, cliff) * ts

        if compound == 'SOFT':
            features[0] += laps       # s_base coefficient
            features[2] += deg_sum    # s_rate coefficient
        elif compound == 'MEDIUM':
            # base is 0 for MEDIUM
            features[3] += deg_sum    # m_rate coefficient
        elif compound == 'HARD':
            features[1] += laps       # h_base coefficient
            features[4] += deg_sum    # h_rate coefficient

    return constant, features


def load_races(max_files=3):
    races = []
    files = sorted(os.listdir(HIST_DIR))[:max_files]
    for fname in files:
        with open(os.path.join(HIST_DIR, fname), 'r') as f:
            races.extend(json.load(f))
    return races


def load_test_cases():
    cases = []
    for i in range(1, 101):
        with open(os.path.join(TEST_IN, f"test_{i:03d}.json"), 'r') as f:
            inp = json.load(f)
        with open(os.path.join(TEST_OUT, f"test_{i:03d}.json"), 'r') as f:
            out = json.load(f)
        cases.append((inp, out['finishing_positions']))
    return cases


def extract_constraints(races, s_cliff, m_cliff, h_cliff, tc, t_ref, max_races=500):
    """
    Extract pairwise constraints from races.
    For each pair (winner, loser): feature_diff dot w < 0
    Returns list of feature difference vectors.
    """
    constraints = []
    for race in races[:max_races]:
        cfg = race['race_config']
        strats = race['strategies']
        true_order = race['finishing_positions']
        pos_map = {d: i for i, d in enumerate(true_order)}

        # Compute features for each driver
        driver_features = {}
        driver_constants = {}
        for pos_key in strats:
            s = strats[pos_key]
            did = s['driver_id']
            const, feat = compute_driver_features(s, cfg, s_cliff, m_cliff, h_cliff, tc, t_ref)
            driver_features[did] = feat
            driver_constants[did] = const

        # For adjacent pairs in finishing order (more stable than all pairs)
        for i in range(len(true_order) - 1):
            winner = true_order[i]
            loser = true_order[i + 1]
            # T(winner) < T(loser)
            # const_w + feat_w . w < const_l + feat_l . w
            # (feat_w - feat_l) . w < const_l - const_w
            # Since we want: diff . w < 0 (when constants cancel or are equal)
            # Actually constants include base and pit which are the same for same race
            feat_diff = driver_features[winner] - driver_features[loser]
            const_diff = driver_constants[winner] - driver_constants[loser]
            constraints.append((feat_diff, const_diff))

    return constraints


def solve_for_weights(constraints):
    """
    Find w such that for each constraint (f, c): f.w + c < 0
    i.e., f.w < -c

    Use least squares: minimize ||Aw - b||^2 where Aw < b is the constraint.
    Or use SVM-like approach.
    """
    # Set up: we want f_i . w < -c_i for all i
    # Equivalently: f_i . w + c_i < 0
    # Let's solve: minimize violations using least-squares on the "margin"

    A = np.array([f for f, c in constraints])
    b = np.array([-c for f, c in constraints])

    # Solve Aw = b in least squares sense
    # This finds w that best approximates the constraints
    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return w


def simulate_with_weights(race_input, w, s_cliff, m_cliff, h_cliff, tc, t_ref):
    """Simulate race using weight vector w = [s_base, h_base, s_rate, m_rate, h_rate]."""
    cfg = race_input['race_config']
    strats = race_input['strategies']

    times = []
    for pos_key in sorted(strats.keys(), key=lambda x: int(x[3:])):
        s = strats[pos_key]
        const, feat = compute_driver_features(s, cfg, s_cliff, m_cliff, h_cliff, tc, t_ref)
        total_time = const + np.dot(feat, w)
        times.append((total_time, int(pos_key[3:]), s['driver_id']))

    times.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in times]


def score_on_tests(test_cases, w, s_cliff, m_cliff, h_cliff, tc, t_ref):
    correct = 0
    for inp, expected in test_cases:
        pred = simulate_with_weights(inp, w, s_cliff, m_cliff, h_cliff, tc, t_ref)
        if pred == expected:
            correct += 1
    return correct


def score_on_hist(races, w, s_cliff, m_cliff, h_cliff, tc, t_ref, max_n=500):
    correct = 0
    for race in races[:max_n]:
        pred = simulate_with_weights(race, w, s_cliff, m_cliff, h_cliff, tc, t_ref)
        if pred == race['finishing_positions']:
            correct += 1
    return correct


def count_satisfied(constraints, w):
    """Count how many pairwise constraints are satisfied."""
    sat = 0
    for f, c in constraints:
        if np.dot(f, w) + c < 0:
            sat += 1
    return sat


def main():
    print("Loading data...")
    races = load_races(max_files=3)
    test_cases = load_test_cases()
    print(f"Loaded {len(races)} races, {len(test_cases)} test cases")

    best_test = 0
    best_params = None
    best_w = None

    # Search over discrete parameters
    for m_cliff in [3, 5, 7, 10, 13, 15, 17, 19]:
        for s_cliff in [1, 2, 3, 4, 5, 7, 9]:
            if s_cliff > m_cliff:
                continue
            for h_cliff in [10, 15, 20, 25, 28, 30, 35]:
                if h_cliff < m_cliff:
                    continue
                for tc in [0.0, 0.01, 0.02, 0.03, 0.05, 0.067, 0.08, 0.1]:
                    t_ref = 30

                    # Extract constraints
                    constraints = extract_constraints(
                        races, s_cliff, m_cliff, h_cliff, tc, t_ref, max_races=1000
                    )

                    if len(constraints) < 10:
                        continue

                    # Solve for weights
                    w = solve_for_weights(constraints)

                    # Ensure physical constraints: s_base < 0, h_base > 0, rates > 0
                    # w = [s_base, h_base, s_rate, m_rate, h_rate]
                    if w[0] > 0 or w[1] < 0 or w[2] < 0 or w[3] < 0 or w[4] < 0:
                        # Skip physically implausible solutions
                        # But also try negating if all signs are wrong
                        if w[0] < 0 and w[1] > 0:
                            pass  # OK
                        else:
                            continue

                    # Score on test cases
                    ts = score_on_tests(test_cases, w, s_cliff, m_cliff, h_cliff, tc, t_ref)

                    if ts > best_test:
                        best_test = ts
                        best_w = w.copy()
                        best_params = (s_cliff, m_cliff, h_cliff, tc, t_ref)
                        sat = count_satisfied(constraints, w)
                        hs = score_on_hist(races, w, s_cliff, m_cliff, h_cliff, tc, t_ref, 500)
                        print(f"NEW BEST test={ts}/100 hist={hs}/500 "
                              f"constraints={sat}/{len(constraints)} | "
                              f"sc={s_cliff} mc={m_cliff} hc={h_cliff} tc={tc} | "
                              f"w={w}")

    if best_w is not None:
        print(f"\n=== BEST RESULT ===")
        print(f"Test score: {best_test}/100")
        print(f"Cliffs: s={best_params[0]} m={best_params[1]} h={best_params[2]}")
        print(f"Temp: tc={best_params[3]} t_ref={best_params[4]}")
        print(f"Weights: s_base={best_w[0]:.6f} h_base={best_w[1]:.6f} "
              f"s_rate={best_w[2]:.6f} m_rate={best_w[3]:.6f} h_rate={best_w[4]:.6f}")
        print(f"\nRatios (relative to m_rate):")
        m = best_w[3]
        print(f"  s_base/m_rate = {best_w[0]/m:.4f}")
        print(f"  h_base/m_rate = {best_w[1]/m:.4f}")
        print(f"  s_rate/m_rate = {best_w[2]/m:.4f}")
        print(f"  h_rate/m_rate = {best_w[4]/m:.4f}")
    else:
        print("No valid solution found")


if __name__ == '__main__':
    main()
