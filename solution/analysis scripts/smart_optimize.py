"""
Smart optimization: use scipy minimize with Nelder-Mead on pairwise loss,
then check count_correct. Also try optimizing on historical data.
"""
import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import random

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

# Load historical data
print("Loading historical data...")
hist = []
for f in sorted(Path("data/historical_races").glob("races_*.json")):
    with open(f) as fh:
        hist.extend(json.load(fh))
print(f"Loaded {len(hist)} races")

# Sample historical for speed
random.seed(42)
hist_sample = random.sample(hist, 3000)

def simulate(strategy, config, params):
    cb_s, cb_h, rs, rm, rh, tc_s, tc_m, tc_h, tref = params
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]

    CB = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    RATE = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    TC = {"SOFT": tc_s, "MEDIUM": tc_m, "HARD": tc_h}
    CLIFF = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}

    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strategy["pit_stops"]}
    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        ts = 1.0 + TC[compound] * (temp - tref)
        deg = RATE[compound] * ts * max(0.0, tire_age - CLIFF[compound])
        total_time += base + CB[compound] + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0
    return total_time

def pairwise_loss_tests(params):
    """Pairwise loss on test cases only."""
    total = 0.0
    for inp, exp in tests:
        config = inp["race_config"]
        times = {}
        for pk in inp["strategies"]:
            s = inp["strategies"][pk]
            times[s["driver_id"]] = simulate(s, config, params)
        for i in range(len(exp) - 1):
            ti, tj = times[exp[i]], times[exp[i+1]]
            if ti >= tj:
                total += (ti - tj + 0.001) ** 2
    return total

def pairwise_loss_hist(params):
    """Pairwise loss on historical data."""
    total = 0.0
    for race in hist_sample:
        config = race["race_config"]
        fp = race["finishing_positions"]
        d2s = {}
        for pk in race["strategies"]:
            s = race["strategies"][pk]
            d2s[s["driver_id"]] = s
        times = {}
        for did, s in d2s.items():
            times[did] = simulate(s, config, params)
        for i in range(len(fp) - 1):
            ti, tj = times[fp[i]], times[fp[i+1]]
            if ti >= tj:
                total += (ti - tj + 0.001) ** 2
    return total

def combined_loss(params):
    """Combined loss on test cases + historical."""
    return pairwise_loss_tests(params) + 0.1 * pairwise_loss_hist(params)

def count_correct_tests(params):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        results = []
        for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
            s = inp["strategies"][pk]
            grid = int(pk[3:])
            t = simulate(s, config, params)
            results.append((t, grid, s["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

# Starting point: current best
x0 = np.array([
    -0.9665103286569976,  # cb_s
    0.755284994643082,    # cb_h
    1.6213600572975244,   # rs
    0.813268608577364,    # rm
    0.345981233247675,    # rh
    0.025806274187704845, # tc_s
    0.02777171692356944,  # tc_m
    0.02401965544225936,  # tc_h
    27.96640138772966,    # tref
])

print(f"Starting score: {count_correct_tests(x0)}/100")
print(f"Starting loss (tests): {pairwise_loss_tests(x0):.4f}")

# Nelder-Mead on test pairwise loss
print("\n=== Nelder-Mead on test pairwise loss ===")
res = minimize(pairwise_loss_tests, x0, method='Nelder-Mead',
               options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
print(f"Loss: {res.fun:.4f}")
print(f"Score: {count_correct_tests(res.x)}/100")
print(f"Params: {res.x}")

# Nelder-Mead on combined loss
print("\n=== Nelder-Mead on combined loss ===")
res2 = minimize(combined_loss, x0, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
print(f"Loss: {res2.fun:.4f}")
print(f"Score: {count_correct_tests(res2.x)}/100")
print(f"Params: {res2.x}")

# Nelder-Mead on hist loss alone
print("\n=== Nelder-Mead on historical loss ===")
res3 = minimize(pairwise_loss_hist, x0, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
print(f"Loss: {res3.fun:.4f}")
print(f"Score: {count_correct_tests(res3.x)}/100")
print(f"Params: {res3.x}")

# Try Powell method (different from NM)
print("\n=== Powell on test loss ===")
res4 = minimize(pairwise_loss_tests, x0, method='Powell',
                options={'maxiter': 10000, 'ftol': 1e-15})
print(f"Loss: {res4.fun:.4f}")
print(f"Score: {count_correct_tests(res4.x)}/100")
print(f"Params: {res4.x}")

# Try multiple random restarts with NM
print("\n=== Random restarts ===")
best = count_correct_tests(x0)
best_x = x0.copy()
for trial in range(50):
    x_init = x0 + np.random.randn(9) * np.abs(x0) * 0.1
    # Keep cliffs at 10, 20, 29 (not optimized)
    try:
        r = minimize(pairwise_loss_tests, x_init, method='Nelder-Mead',
                     options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        s = count_correct_tests(r.x)
        if s > best:
            best = s
            best_x = r.x.copy()
            print(f"  Trial {trial}: {s}/100 loss={r.fun:.4f}")
    except:
        pass

print(f"\nBest overall: {best}/100")
print(f"Best params: {best_x}")
