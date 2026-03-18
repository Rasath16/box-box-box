"""
For each temperature value in the data, extract optimal rates
using just races at that temperature. Then plot how rates vary with temp.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# Load ALL races
print("Loading races...")
races = []
data_dir = Path("data/historical_races")
for f in sorted(data_dir.glob("races_*.json")):
    with open(f) as fh:
        races.extend(json.load(fh))
print(f"Loaded {len(races)} races")

# Group by temperature
temp_races = defaultdict(list)
for r in races:
    temp_races[r["race_config"]["track_temp"]].append(r)

print(f"Temperatures: {sorted(temp_races.keys())}")
print(f"Counts per temp: {[(t, len(temp_races[t])) for t in sorted(temp_races.keys())]}")

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

# For given cliff values, extract features
def extract_features(strategy, config, cliffs):
    total_laps = config["total_laps"]
    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0

    n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
    sum_excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    n_pits = len(strategy["pit_stops"])

    for lap in range(1, total_laps + 1):
        tire_age += 1
        n_laps[compound] += 1
        excess = max(0.0, tire_age - cliffs[compound])
        sum_excess[compound] += excess
        if lap in pit_laps:
            compound = pit_laps[lap]
            tire_age = 0

    return n_laps, sum_excess, n_pits

# For each temperature, use pairwise regression to find best params
cliffs = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}

print("\n=== Per-temperature parameter extraction ===")
temp_params = {}

for temp in sorted(temp_races.keys()):
    t_races = temp_races[temp]
    if len(t_races) < 50:
        continue

    A_rows = []
    b_rows = []

    for race in t_races[:500]:
        config = race["race_config"]
        fp = race["finishing_positions"]
        strategies = race["strategies"]

        d2s = {}
        for pk in strategies:
            s = strategies[pk]
            d2s[s["driver_id"]] = s

        # Consecutive pairs
        for i in range(len(fp) - 1):
            dw, dl = fp[i], fp[i+1]
            sw, sl = d2s[dw], d2s[dl]

            nw, ew, pw = extract_features(sw, config, cliffs)
            nl, el, pl = extract_features(sl, config, cliffs)

            # Features: [n_soft_w - n_soft_l, n_hard_w - n_hard_l,
            #            excess_soft_w - excess_soft_l, excess_med_w - excess_med_l, excess_hard_w - excess_hard_l]
            # Target: (pl - pw) * pit_time  (since pit_time contributes to total)
            delta_feat = [
                nw["SOFT"] - nl["SOFT"],
                nw["HARD"] - nl["HARD"],
                ew["SOFT"] - el["SOFT"],
                ew["MEDIUM"] - el["MEDIUM"],
                ew["HARD"] - el["HARD"],
            ]
            delta_const = (pl - pw) * config["pit_lane_time"]

            A_rows.append(delta_feat)
            b_rows.append(delta_const)

    A = np.array(A_rows)
    b = np.array(b_rows)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cb_s, cb_h, rs, rm, rh = params
    temp_params[temp] = (cb_s, cb_h, rs, rm, rh)
    print(f"  temp={temp}: cb_s={cb_s:.6f} cb_h={cb_h:.6f} rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")

# Now analyze how rates change with temperature
print("\n=== Rate vs Temperature ===")
temps = sorted(temp_params.keys())
print(f"{'Temp':>5} {'cb_s':>10} {'cb_h':>10} {'rs':>10} {'rm':>10} {'rh':>10}")
for t in temps:
    cb_s, cb_h, rs, rm, rh = temp_params[t]
    print(f"{t:>5} {cb_s:>10.6f} {cb_h:>10.6f} {rs:>10.6f} {rm:>10.6f} {rh:>10.6f}")

# Check if compound bases are constant (don't vary with temp)
print("\n=== Are compound bases constant across temperatures? ===")
cbs_list = [temp_params[t][0] for t in temps]
cbh_list = [temp_params[t][1] for t in temps]
print(f"cb_s: mean={np.mean(cbs_list):.6f} std={np.std(cbs_list):.6f}")
print(f"cb_h: mean={np.mean(cbh_list):.6f} std={np.std(cbh_list):.6f}")

# Check if rates scale linearly with temp
print("\n=== Linear fit of rates vs temperature ===")
for idx, name in [(2, "rs"), (3, "rm"), (4, "rh")]:
    rates = [temp_params[t][idx] for t in temps]
    # Fit: rate = a + b * temp
    coeffs = np.polyfit(temps, rates, 1)
    print(f"  {name} = {coeffs[1]:.6f} + {coeffs[0]:.6f} * temp")
    # Or: rate = r0 * (1 + tc * (temp - tref))
    # rate = r0 + r0*tc*temp - r0*tc*tref
    # So: b = r0*tc, a = r0 - r0*tc*tref = r0*(1-tc*tref)
    # r0 = (a*tref - a + b*tref ... hmm)
    # Let's just fit: rate = r0 * (1 + tc * (temp - tref))
    # For a given tref, r0 = rate_at_tref, tc = slope / r0
    for tref in [25, 28, 30]:
        r0 = coeffs[1] + coeffs[0] * tref
        if r0 != 0:
            tc = coeffs[0] / r0
            print(f"    tref={tref}: r0={r0:.6f} tc={tc:.6f}")

# Now check if all compounds have the SAME temperature coefficient
print("\n=== Check if tc is same across compounds ===")
for tref in [25, 28, 30]:
    tcs = []
    for idx, name in [(2, "rs"), (3, "rm"), (4, "rh")]:
        rates = [temp_params[t][idx] for t in temps]
        coeffs = np.polyfit(temps, rates, 1)
        r0 = coeffs[1] + coeffs[0] * tref
        if r0 != 0:
            tc = coeffs[0] / r0
            tcs.append((name, tc, r0))
    print(f"  tref={tref}: {[(n, f'{tc:.6f}', f'{r0:.6f}') for n, tc, r0 in tcs]}")
