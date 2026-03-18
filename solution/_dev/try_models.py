"""
Try different model variants to see which scores best.
"""
import json
import os
import numpy as np
from scipy.optimize import differential_evolution, minimize

TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")


def load_test_cases():
    cases = []
    for i in range(1, 101):
        with open(os.path.join(TEST_INPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            inp = json.load(f)
        with open(os.path.join(TEST_OUTPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            out = json.load(f)
        cases.append((inp, out['finishing_positions']))
    return cases


def pairwise_loss(pred_order, true_order):
    true_pos = {d: i for i, d in enumerate(true_order)}
    pred_pos = {d: i for i, d in enumerate(pred_order)}
    drivers = list(true_pos.keys())
    n = len(drivers)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2 = drivers[i], drivers[j]
            if (true_pos[d1] < true_pos[d2]) != (pred_pos[d1] < pred_pos[d2]):
                inv += 1
    return inv


# ===================== MODEL VARIANTS =====================

def sim_model_A(race_input, params):
    """Model A: per-compound tc + temp-dependent base offsets."""
    s_base, h_base = params[0], params[1]
    s_rate, m_rate, h_rate = params[2], params[3], params[4]
    tc_s, tc_m, tc_h = params[5], params[6], params[7]
    t_ref = params[8]
    sb_tc, hb_tc = params[9], params[10]  # temp coeff for base offsets

    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']
    dt = temp - t_ref

    s_cliff, m_cliff, h_cliff = 10, 20, 29

    cb = {
        'SOFT': s_base * (1.0 + sb_tc * dt),
        'MEDIUM': 0.0,
        'HARD': h_base * (1.0 + hb_tc * dt)
    }
    cr = {
        'SOFT': s_rate * (1.0 + tc_s * dt),
        'MEDIUM': m_rate * (1.0 + tc_m * dt),
        'HARD': h_rate * (1.0 + tc_h * dt)
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


def sim_model_B(race_input, params):
    """Model B: per-compound tc + quadratic degradation term."""
    s_base, h_base = params[0], params[1]
    s_rate, m_rate, h_rate = params[2], params[3], params[4]
    tc_s, tc_m, tc_h = params[5], params[6], params[7]
    t_ref = params[8]
    quad = params[9]  # quadratic factor

    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']
    dt = temp - t_ref

    s_cliff, m_cliff, h_cliff = 10, 20, 29

    cb = {'SOFT': s_base, 'MEDIUM': 0.0, 'HARD': h_base}
    cr = {
        'SOFT': s_rate * (1.0 + tc_s * dt),
        'MEDIUM': m_rate * (1.0 + tc_m * dt),
        'HARD': h_rate * (1.0 + tc_h * dt)
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
            excess = max(0.0, age - cc[cur])
            deg = cr[cur] * (excess + quad * excess * excess)
            total += base + cb[cur] + deg
            if lap in pit_map:
                total += pit_time
                cur = pit_map[lap]
                age = 0
        times.append((total, int(pk[3:]), s['driver_id']))
    times.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in times]


def sim_model_C(race_input, params):
    """Model C: single tc + fractional cliffs."""
    s_base, h_base = params[0], params[1]
    s_rate, m_rate, h_rate = params[2], params[3], params[4]
    s_cliff, m_cliff, h_cliff = params[5], params[6], params[7]
    tc, t_ref = params[8], params[9]

    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']
    dt = temp - t_ref
    ts = 1.0 + tc * dt

    cb = {'SOFT': s_base, 'MEDIUM': 0.0, 'HARD': h_base}
    cr = {'SOFT': s_rate * ts, 'MEDIUM': m_rate * ts, 'HARD': h_rate * ts}
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


def optimize_model(name, sim_fn, bounds, test_cases, n_seeds=3):
    print(f"\n{'='*60}")
    print(f"Model {name}")
    print(f"{'='*60}")

    best_score = 0
    best_params = None

    for seed in [42, 123, 777]:
        def loss_fn(x):
            total_inv = 0
            total_pairs = 0
            for inp, true_order in test_cases:
                pred = sim_fn(inp, x)
                total_inv += pairwise_loss(pred, true_order)
                total_pairs += 190  # 20 choose 2
            return total_inv / total_pairs

        result = differential_evolution(
            loss_fn, bounds,
            maxiter=200, popsize=25,
            tol=1e-9, seed=seed,
            mutation=(0.5, 1.5), recombination=0.9,
            disp=True, polish=True,
        )

        params = result.x
        score = sum(1 for inp, true in test_cases if sim_fn(inp, params) == true)
        print(f"  Seed {seed}: score={score}/100, loss={result.fun:.8f}")
        print(f"  Params: {repr(list(params))}")

        if score > best_score:
            best_score = score
            best_params = list(params)

    print(f"\nBest for Model {name}: {best_score}/100")
    print(f"Params: {repr(best_params)}")
    return best_params, best_score


def main():
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")

    results = {}

    # Model A: per-compound tc + temp-dependent base offsets
    bounds_A = [
        (-2.0, -0.3), (0.2, 1.5),      # s_base, h_base
        (0.5, 3.0), (0.2, 1.5), (0.05, 0.8),  # rates
        (0.005, 0.08), (0.005, 0.08), (0.005, 0.08),  # tc_s, tc_m, tc_h
        (15, 40),           # t_ref
        (-0.05, 0.05), (-0.05, 0.05),  # sb_tc, hb_tc
    ]
    params_A, score_A = optimize_model("A (base+tc)", sim_model_A, bounds_A, test_cases)
    results['A'] = (params_A, score_A)

    # Model B: per-compound tc + quadratic degradation
    bounds_B = [
        (-2.0, -0.3), (0.2, 1.5),
        (0.5, 3.0), (0.2, 1.5), (0.05, 0.8),
        (0.005, 0.08), (0.005, 0.08), (0.005, 0.08),
        (15, 40),
        (-0.05, 0.1),   # quad factor
    ]
    params_B, score_B = optimize_model("B (quad deg)", sim_model_B, bounds_B, test_cases)
    results['B'] = (params_B, score_B)

    # Model C: fractional cliffs with single tc
    bounds_C = [
        (-2.0, -0.3), (0.2, 1.5),
        (0.5, 3.0), (0.2, 1.5), (0.05, 0.8),
        (5, 15), (12, 25), (20, 35),  # fractional cliffs
        (0.005, 0.1), (15, 40),
    ]
    params_C, score_C = optimize_model("C (frac cliffs)", sim_model_C, bounds_C, test_cases)
    results['C'] = (params_C, score_C)

    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, (params, score) in results.items():
        print(f"Model {name}: {score}/100")
        print(f"  Params: {repr(params)}")


if __name__ == '__main__':
    main()
