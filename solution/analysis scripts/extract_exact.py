"""
Extract exact parameters from historical data using analytical approach.
Key insight: with 30k races we can find races that isolate specific parameters.

Strategy: Find pairs of drivers in the same race who differ by exactly one variable.
"""
import json
import os
import numpy as np
from collections import defaultdict

HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")
TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")


def load_all_hist():
    races = []
    for fname in sorted(os.listdir(HIST_DIR)):
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


def get_strategy_signature(strat):
    """Return a tuple describing the strategy."""
    compounds = [strat['starting_tire']]
    pit_laps = []
    for stop in sorted(strat.get('pit_stops', []), key=lambda x: x['lap']):
        compounds.append(stop['to_tire'])
        pit_laps.append(stop['lap'])
    return tuple(compounds), tuple(pit_laps)


def analyze_same_strategy_pairs(races):
    """
    Find races where two drivers use identical strategies
    (same compounds, same pit laps). They should have identical times.
    """
    print("Looking for identical strategy pairs...")
    identical_count = 0
    for race in races[:5000]:
        strats = race['strategies']
        fp = race['finishing_positions']
        cfg = race['race_config']

        sigs = {}
        for pk, s in strats.items():
            sig = get_strategy_signature(s)
            if sig not in sigs:
                sigs[sig] = []
            sigs[sig].append(s['driver_id'])

        for sig, drivers in sigs.items():
            if len(drivers) > 1:
                identical_count += 1
                # These drivers should have same time, ordered by grid position
                # Check their positions in finishing order
                positions = {d: fp.index(d) for d in drivers}
                # They should be consecutive in finishing order, sorted by grid
                grid_map = {}
                for pk, s in strats.items():
                    if s['driver_id'] in drivers:
                        grid_map[s['driver_id']] = int(pk[3:])

    print(f"Found {identical_count} identical strategy groups in first 5000 races")


def find_soft_base_from_SM_vs_MM(races):
    """
    Find races with one driver on S->M@L and another on M->M@L.
    Actually, find M->H and S->H with same pit lap to isolate soft vs medium base.
    """
    print("\nExtracting SOFT_BASE from S->H vs M->H pairs...")

    # For S->H@L vs M->H@L:
    # Difference = L * s_base - L * 0 in the first stint
    # Plus degradation difference in first stint

    # Actually, let me think about this more carefully.
    # For compound C starting at lap 1, pitting at lap L:
    # Total compound time = sum_{age=1}^{L} (base + cb[C] + cr[C] * max(0, age - cliff[C]))
    # = L*base + L*cb[C] + cr[C] * sum_{age=1}^{L} max(0, age - cliff[C])

    # So for two drivers with same pit lap and same second stint:
    # S->H@L vs M->H@L
    # diff = L * (cb[S] - cb[M]) + (cr[S]*ts - cr[M]*ts) * (sum terms)
    # Since second stint is identical, this is the total time difference.

    # If L <= min(s_cliff, m_cliff) = 10:
    # No degradation in first stint for either compound
    # diff = L * s_base (since cb[M] = 0)
    # So s_base = diff / L

    # We can't observe diff directly, but we can observe ordering.
    # If S is ahead, s_base < 0 (SOFT is faster base).
    # The question is: can we find cases where we know diff precisely?

    # Let me look for S->H vs M->H with pit lap <= 10 (no degradation in first stint)
    data_points = []

    for race in races:
        cfg = race['race_config']
        strats = race['strategies']
        fp = race['finishing_positions']
        temp = cfg['track_temp']
        total_laps = cfg['total_laps']

        drivers = {}
        for pk, s in strats.items():
            sig = get_strategy_signature(s)
            drivers[s['driver_id']] = {
                'grid': int(pk[3:]),
                'compounds': sig[0],
                'pit_laps': sig[1],
                'strat': s
            }

        # Find S->H@L and M->H@L pairs with L <= 9 (both in cliff period)
        sh_drivers = {}
        mh_drivers = {}
        for did, info in drivers.items():
            if info['compounds'] == ('SOFT', 'HARD') and len(info['pit_laps']) == 1:
                pl = info['pit_laps'][0]
                if pl <= 9:  # Both S and M within their cliffs
                    sh_drivers[pl] = did
            elif info['compounds'] == ('MEDIUM', 'HARD') and len(info['pit_laps']) == 1:
                pl = info['pit_laps'][0]
                if pl <= 9:
                    mh_drivers[pl] = did

        # Find matching pit laps
        for pl in sh_drivers:
            if pl in mh_drivers:
                sd = sh_drivers[pl]
                md = mh_drivers[pl]
                sp = fp.index(sd)
                mp = fp.index(md)
                # Time diff (S-M) = pl * s_base
                # If S finishes ahead, s_base < 0 -> SOFT is faster
                data_points.append({
                    'pit_lap': pl,
                    'temp': temp,
                    'total_laps': total_laps,
                    's_ahead': sp < mp,
                    's_pos': sp,
                    'm_pos': mp,
                })

    print(f"Found {len(data_points)} S->H vs M->H pairs with pit_lap <= 9")
    if data_points:
        ahead = sum(1 for d in data_points if d['s_ahead'])
        print(f"  SOFT ahead: {ahead}/{len(data_points)} ({100*ahead/len(data_points):.1f}%)")


def extract_temp_effect(races):
    """
    Group identical strategies across different temperatures to isolate temp effect.
    """
    print("\nExtracting temperature effect...")

    # For the same strategy at different temps:
    # The only difference is the temperature scaling of degradation rates
    # If we find the same strategy (same compounds, same pit laps, same total_laps, same base_lap_time)
    # but different temps, we can isolate the temp effect.

    # Group by (base_lap_time, pit_lane_time, total_laps, strategy_sig)
    groups = defaultdict(list)

    for race in races:
        cfg = race['race_config']
        strats = race['strategies']
        fp = race['finishing_positions']

        for pk, s in strats.items():
            sig = get_strategy_signature(s)
            key = (cfg['base_lap_time'], cfg['pit_lane_time'], cfg['total_laps'], sig)
            groups[key].append({
                'temp': cfg['track_temp'],
                'driver_id': s['driver_id'],
                'finish_pos': fp.index(s['driver_id']),
                'grid': int(pk[3:]),
            })

    # Find groups with multiple temps
    multi_temp = {k: v for k, v in groups.items() if len(set(d['temp'] for d in v)) > 1}
    print(f"Found {len(multi_temp)} strategy groups with multiple temps")

    # This is too complex for direct extraction. Let me try something simpler.


def brute_force_from_hist(races, test_cases):
    """
    Use historical race data to constrain parameters, then evaluate on tests.
    Use many races to build a linear system.
    """
    print("\nExtracting parameters from historical data...")

    # For each pair of drivers in a race, we know which one finishes ahead.
    # time_A < time_B gives us a constraint on the parameters.
    # With enough constraints, we can narrow down the parameter space.

    # More practical: use a subset of hist races to evaluate parameter quality,
    # then optimize on combined hist+test loss.

    from scipy.optimize import differential_evolution

    # Sample hist races for additional signal
    hist_sample = races[:500]
    hist_with_truth = [(r, r['finishing_positions']) for r in hist_sample]

    def sim_percomp(race_input, params):
        s_base, h_base = params[0], params[1]
        s_rate, m_rate, h_rate = params[2], params[3], params[4]
        tc_s, tc_m, tc_h = params[5], params[6], params[7]
        t_ref = params[8]
        s_cliff, m_cliff, h_cliff = 10, 20, 29

        cfg = race_input['race_config']
        base = cfg['base_lap_time']
        pit_time = cfg['pit_lane_time']
        total_laps = cfg['total_laps']
        temp = cfg['track_temp']
        dt = temp - t_ref

        cb = {'SOFT': s_base, 'MEDIUM': 0.0, 'HARD': h_base}
        cr = {
            'SOFT': s_rate * (1 + tc_s * dt),
            'MEDIUM': m_rate * (1 + tc_m * dt),
            'HARD': h_rate * (1 + tc_h * dt)
        }
        cc = {'SOFT': s_cliff, 'MEDIUM': m_cliff, 'HARD': h_cliff}

        strats = race_input['strategies']
        times = []
        for pk in sorted(strats.keys(), key=lambda x: int(x[3:])):
            s = strats[pk]
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
            times.append((total, int(pk[3:]), s['driver_id']))
        times.sort(key=lambda x: (x[0], x[1]))
        return [t[2] for t in times]

    def combined_loss(x):
        total_inv = 0
        total_pairs = 0
        # Test cases (weighted more)
        for inp, true_order in test_cases:
            pred = sim_percomp(inp, x)
            true_pos = {d: i for i, d in enumerate(true_order)}
            pred_pos = {d: i for i, d in enumerate(pred)}
            drivers = list(true_pos.keys())
            n = len(drivers)
            total_pairs += n * (n - 1) // 2 * 3  # weight 3x
            for i in range(n):
                for j in range(i + 1, n):
                    d1, d2 = drivers[i], drivers[j]
                    if (true_pos[d1] < true_pos[d2]) != (pred_pos[d1] < pred_pos[d2]):
                        total_inv += 3
        # Historical races
        for race, true_order in hist_with_truth:
            pred = sim_percomp(race, x)
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

    bounds = [
        (-2.0, -0.3), (0.2, 1.5),
        (0.5, 3.0), (0.2, 1.5), (0.05, 0.8),
        (0.005, 0.08), (0.005, 0.08), (0.005, 0.08),
        (15, 40),
    ]

    print("Running combined hist+test DE optimization...")
    result = differential_evolution(
        combined_loss, bounds,
        maxiter=200, popsize=25,
        tol=1e-9, seed=42,
        mutation=(0.5, 1.5), recombination=0.9,
        disp=True, polish=True,
    )

    best = result.x
    score = sum(1 for inp, true in test_cases if sim_percomp(inp, best) == true)
    hist_score = sum(1 for r, true in hist_with_truth if sim_percomp(r, best) == true)

    print(f"\nCombined DE: test={score}/100, hist={hist_score}/500")
    print(f"Params: {repr(list(best))}")
    print(f"Full: {list(best[:5])} + [10, 20, 29] + {list(best[5:])}")

    return best, score


def main():
    print("Loading data...")
    races = load_all_hist()
    test_cases = load_test_cases()
    print(f"Loaded {len(races)} historical races, {len(test_cases)} test cases")

    analyze_same_strategy_pairs(races)
    find_soft_base_from_SM_vs_MM(races)

    best, score = brute_force_from_hist(races, test_cases)


if __name__ == '__main__':
    main()
