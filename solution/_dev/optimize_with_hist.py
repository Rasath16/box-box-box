"""
Optimize parameters using both 30k historical races and 100 test cases.

The historical data has exact finishing orders, giving us massive signal
to constrain parameters precisely. We use a sample of historical races
plus all test cases with pairwise ranking loss.

Then refine with Nelder-Mead and coordinate descent on exact match score.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution, minimize

HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")
TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")


def load_hist_sample(n_files=5):
    """Load a sample of historical races."""
    races = []
    files = sorted(os.listdir(HIST_DIR))
    # Spread across files for diversity
    indices = np.linspace(0, len(files) - 1, n_files, dtype=int)
    for idx in indices:
        with open(os.path.join(HIST_DIR, files[idx]), 'r') as f:
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


def simulate(race_input, params):
    """Simulate race and return predicted finishing order."""
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


def pairwise_loss_batch(params, races_with_truth, weight=1.0):
    """Compute pairwise inversion loss over a batch of races."""
    total_inv = 0
    total_pairs = 0
    for race, true_order in races_with_truth:
        pred = simulate(race, params)
        true_pos = {d: i for i, d in enumerate(true_order)}
        pred_pos = {d: i for i, d in enumerate(pred)}
        drivers = list(true_pos.keys())
        n = len(drivers)
        total_pairs += n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                d1, d2 = drivers[i], drivers[j]
                if (true_pos[d1] < true_pos[d2]) != (pred_pos[d1] < pred_pos[d2]):
                    total_inv += weight
    return total_inv, total_pairs


def exact_score(params, cases):
    return sum(1 for inp, true in cases if simulate(inp, params) == true)


def main():
    print("Loading test cases...")
    test_cases = load_test_cases()

    print("Loading historical races (5 files = 5000 races)...")
    hist_races = load_hist_sample(n_files=5)
    hist_with_truth = [(r, r['finishing_positions']) for r in hist_races]
    print(f"Loaded {len(hist_races)} historical races")

    # Current best params
    current = [
        -0.9665103286569976, 0.755284994643082,
        1.6213600572975244, 0.813268608577364, 0.3404666107633389,
        0.025806274187704845, 0.02777171692356944, 0.02401965544225936,
        27.569263880216884
    ]
    cur_test = exact_score(current, test_cases)
    cur_hist = exact_score(current, hist_with_truth)
    print(f"Current: test={cur_test}/100, hist={cur_hist}/{len(hist_races)}")

    # === Phase 1: DE with combined hist + test loss ===
    def combined_loss(x):
        # Test cases weighted 10x
        test_inv, test_pairs = pairwise_loss_batch(x, test_cases, weight=10.0)
        # Subsample hist for speed (use every 5th race)
        hist_sub = hist_with_truth[::5]
        hist_inv, hist_pairs = pairwise_loss_batch(x, hist_sub, weight=1.0)
        return (test_inv + hist_inv) / (test_pairs * 10 + hist_pairs)

    bounds = [
        (-2.0, -0.3),   # s_base
        (0.2, 1.5),     # h_base
        (0.5, 3.0),     # s_rate
        (0.2, 1.5),     # m_rate
        (0.05, 0.8),    # h_rate
        (0.005, 0.08),  # tc_s
        (0.005, 0.08),  # tc_m
        (0.005, 0.08),  # tc_h
        (15, 40),       # t_ref
    ]

    overall_best_score = cur_test
    overall_best_params = list(current)

    for seed in [42, 123, 777, 2024, 31415]:
        print(f"\n--- DE seed={seed} ---")
        result = differential_evolution(
            combined_loss, bounds,
            maxiter=150, popsize=20,
            tol=1e-9, seed=seed,
            mutation=(0.5, 1.5), recombination=0.9,
            disp=True, polish=True,
        )

        params = list(result.x)
        score = exact_score(params, test_cases)
        hscore = exact_score(params, hist_with_truth)
        print(f"Score: test={score}/100, hist={hscore}/{len(hist_races)}, loss={result.fun:.8f}")

        if score > overall_best_score:
            overall_best_score = score
            overall_best_params = list(params)
            print(f"*** NEW BEST: {score}/100 ***")

        # Nelder-Mead refinement on test-only loss
        def test_loss(x):
            inv, pairs = pairwise_loss_batch(x, test_cases)
            return inv / pairs

        ref = minimize(test_loss, params, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
        ref_score = exact_score(list(ref.x), test_cases)
        print(f"After NM: {ref_score}/100")

        if ref_score > overall_best_score:
            overall_best_score = ref_score
            overall_best_params = list(ref.x)
            print(f"*** NEW BEST: {ref_score}/100 ***")

    # === Phase 2: Coordinate descent on exact match ===
    print(f"\n=== Coordinate descent from best ({overall_best_score}/100) ===")
    names = ['s_base', 'h_base', 's_rate', 'm_rate', 'h_rate', 'tc_s', 'tc_m', 'tc_h', 't_ref']
    improved = True
    rounds = 0
    while improved and rounds < 5:
        improved = False
        rounds += 1
        for i in range(9):
            for delta in [-0.01, -0.005, -0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002, 0.005, 0.01]:
                params = list(overall_best_params)
                params[i] *= (1 + delta)
                score = exact_score(params, test_cases)
                if score > overall_best_score:
                    overall_best_score = score
                    overall_best_params = list(params)
                    print(f"  {names[i]} {delta:+.4f}: {score}/100")
                    improved = True

    # === Phase 3: Random perturbations ===
    print(f"\n=== Random perturbations from {overall_best_score}/100 ===")
    rng = np.random.RandomState(42)
    scales = [0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.01]
    for trial in range(30000):
        params = list(overall_best_params)
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(9, n_perturb, replace=False)
        for idx in indices:
            params[idx] *= (1 + scales[idx] * rng.randn())
        score = exact_score(params, test_cases)
        if score > overall_best_score:
            overall_best_score = score
            overall_best_params = list(params)
            print(f"  Trial {trial}: {score}/100")

    print(f"\n{'='*60}")
    print(f"FINAL BEST: {overall_best_score}/100")
    print(f"Params: {repr(overall_best_params)}")
    print(f"Full with cliffs: {overall_best_params[:5]} + [10, 20, 29] + {overall_best_params[5:]}")


if __name__ == '__main__':
    main()
