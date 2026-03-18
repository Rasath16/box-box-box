"""
Extract parameters using linear algebra on historical data.

Key insight: For drivers with identical strategies in the same race,
they should have identical race times. But we can compare drivers
with DIFFERENT strategies to extract the compound/degradation parameters.

For two drivers A (compound X) and B (compound Y) who finish at positions
p_A and p_B, if p_A < p_B then time_A < time_B.

We can set up linear constraints and solve.

Better approach: for each race, compute the time difference formula
between all pairs of drivers and check if A beats B. This gives us
a system of linear inequalities that the parameters must satisfy.

Even better: with known finishing order, we can set up a regression
using the ORDER as the target and strategy features as predictors.
"""
import json
import os
import numpy as np
from scipy.optimize import minimize, differential_evolution

HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")
TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")


def load_hist(max_files=None):
    races = []
    files = sorted(os.listdir(HIST_DIR))
    if max_files:
        files = files[:max_files]
    for fname in files:
        with open(os.path.join(HIST_DIR, fname), 'r') as f:
            races.extend(json.load(f))
    return races


def load_test_cases():
    cases = []
    for i in range(1, 101):
        with open(os.path.join(TEST_INPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            inp = json.load(f)
        with open(os.path.join(TEST_OUTPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            out = json.load(f)
        cases.append((inp, out['finishing_positions']))
    return cases


def compute_features(strategy, total_laps, cliffs=(10, 20, 29)):
    """
    Compute features for the race time formula:
    time = total_laps * base + features @ [s_base, h_base, s_rate*ts, m_rate*ts, h_rate*ts] + n_stops * pit_time

    Returns: (n_laps_soft, n_laps_hard, soft_deg_sum, med_deg_sum, hard_deg_sum, n_stops)
    """
    s_cliff, m_cliff, h_cliff = cliffs

    pit_stops = sorted(strategy.get('pit_stops', []), key=lambda x: x['lap'])

    # Build stint list: (compound, start_lap, end_lap)
    stints = []
    cur = strategy['starting_tire']
    start = 1
    for stop in pit_stops:
        stints.append((cur, start, stop['lap']))
        cur = stop['to_tire']
        start = stop['lap'] + 1
    stints.append((cur, start, total_laps))

    n_soft = 0
    n_hard = 0
    soft_deg = 0.0
    med_deg = 0.0
    hard_deg = 0.0

    for compound, s_lap, e_lap in stints:
        n_laps = e_lap - s_lap + 1
        if compound == 'SOFT':
            n_soft += n_laps
            for age in range(1, n_laps + 1):
                soft_deg += max(0.0, age - s_cliff)
        elif compound == 'MEDIUM':
            for age in range(1, n_laps + 1):
                med_deg += max(0.0, age - m_cliff)
        elif compound == 'HARD':
            n_hard += n_laps
            for age in range(1, n_laps + 1):
                hard_deg += max(0.0, age - h_cliff)

    n_stops = len(pit_stops)
    return n_soft, n_hard, soft_deg, med_deg, hard_deg, n_stops


def time_from_features(feats, params, base_lap, pit_time, total_laps, temp):
    """
    Compute race time from features.
    params: [s_base, h_base, s_rate, m_rate, h_rate, tc_s, tc_m, tc_h, t_ref]
    """
    s_base, h_base = params[0], params[1]
    s_rate, m_rate, h_rate = params[2], params[3], params[4]
    tc_s, tc_m, tc_h = params[5], params[6], params[7]
    t_ref = params[8]

    dt = temp - t_ref
    n_soft, n_hard, soft_deg, med_deg, hard_deg, n_stops = feats

    time = total_laps * base_lap
    time += n_soft * s_base + n_hard * h_base
    time += soft_deg * s_rate * (1 + tc_s * dt)
    time += med_deg * m_rate * (1 + tc_m * dt)
    time += hard_deg * h_rate * (1 + tc_h * dt)
    time += n_stops * pit_time

    return time


def extract_from_pairwise_constraints(races, n_races=2000):
    """
    For each pair of drivers (A, B) where A finishes ahead of B,
    we know time_A < time_B, which means:
    features_A @ params < features_B @ params
    (features_B - features_A) @ params > 0

    We can use these constraints to find the parameters.
    """
    print(f"Building pairwise constraints from {n_races} races...")

    cliffs = (10, 20, 29)
    constraints = []  # (feature_diff, temp)

    for race in races[:n_races]:
        cfg = race['race_config']
        strats = race['strategies']
        fp = race['finishing_positions']

        # Compute features for all drivers
        driver_feats = {}
        for pk, s in strats.items():
            feats = compute_features(s, cfg['total_laps'], cliffs)
            driver_feats[s['driver_id']] = feats

        # For each pair where A finishes ahead of B (consecutive positions)
        for i in range(len(fp) - 1):
            a = fp[i]
            b = fp[i + 1]
            fa = driver_feats[a]
            fb = driver_feats[b]

            # time_B > time_A means features_B @ params_part > features_A @ params_part
            # diff = features_B - features_A
            diff = tuple(fb[k] - fa[k] for k in range(6))
            constraints.append((diff, cfg['track_temp'], cfg['base_lap_time'], cfg['pit_lane_time'], cfg['total_laps']))

    print(f"Built {len(constraints)} pairwise constraints")

    # Now optimize: maximize the number of satisfied constraints
    def loss(x):
        s_base, h_base, s_rate, m_rate, h_rate = x[:5]
        tc_s, tc_m, tc_h, t_ref = x[5:9]

        violations = 0
        for diff, temp, base_lap, pit_time, total_laps in constraints:
            dt = temp - t_ref
            # time_diff = diff_n_soft * s_base + diff_n_hard * h_base
            #           + diff_soft_deg * s_rate * (1+tc_s*dt)
            #           + diff_med_deg * m_rate * (1+tc_m*dt)
            #           + diff_hard_deg * h_rate * (1+tc_h*dt)
            #           + diff_n_stops * pit_time
            td = (diff[0] * s_base + diff[1] * h_base +
                  diff[2] * s_rate * (1 + tc_s * dt) +
                  diff[3] * m_rate * (1 + tc_m * dt) +
                  diff[4] * h_rate * (1 + tc_h * dt) +
                  diff[5] * pit_time)
            if td <= 0:  # Violated: B should be slower than A
                violations += 1
        return violations / len(constraints)

    bounds = [
        (-2.0, -0.3), (0.2, 1.5),
        (0.5, 3.0), (0.2, 1.5), (0.05, 0.8),
        (0.005, 0.08), (0.005, 0.08), (0.005, 0.08),
        (15, 40),
    ]

    print("Optimizing with DE on hist constraints...")
    result = differential_evolution(
        loss, bounds,
        maxiter=200, popsize=25,
        tol=1e-9, seed=42,
        mutation=(0.5, 1.5), recombination=0.9,
        disp=True, polish=True,
    )

    print(f"\nConstraint violation rate: {result.fun:.6f}")
    print(f"Params: {repr(list(result.x))}")
    return result.x


def main():
    print("Loading data...")
    races = load_hist(max_files=3)
    test_cases = load_test_cases()
    print(f"Loaded {len(races)} historical races, {len(test_cases)} test cases")

    params = extract_from_pairwise_constraints(races, n_races=3000)

    # Evaluate on test cases
    full_params = list(params)
    cliffs = (10, 20, 29)

    correct = 0
    for inp, true_order in test_cases:
        cfg = inp['race_config']
        strats = inp['strategies']

        times = []
        for pk in sorted(strats.keys(), key=lambda x: int(x[3:])):
            s = strats[pk]
            feats = compute_features(s, cfg['total_laps'], cliffs)
            t = time_from_features(feats, full_params, cfg['base_lap_time'],
                                   cfg['pit_lane_time'], cfg['total_laps'],
                                   cfg['track_temp'])
            times.append((t, int(pk[3:]), s['driver_id']))

        times.sort(key=lambda x: (x[0], x[1]))
        pred = [t[2] for t in times]
        if pred == true_order:
            correct += 1

    print(f"\nTest accuracy: {correct}/100")
    print(f"Full params: {list(params[:5])} + [10, 20, 29] + {list(params[5:])}")


if __name__ == '__main__':
    main()
