"""
Better parameter extraction from historical data.
Use pairs of drivers within the same race who have identical strategies
(same compounds, same pit laps). These should have identical times.
Then look at near-identical strategies to extract degradation differences.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("Loading...")
races = []
for f in sorted(Path("data/historical_races").glob("races_*.json")):
    with open(f) as fh:
        races.extend(json.load(fh))

# Focus on single-compound-pair strategies: compound1 -> pit -> compound2
# These are the simplest to analyze

# Key: find races at moderate temp (28-30) where two drivers:
# 1. Use the same compound sequence (e.g., both SOFT->HARD)
# 2. Pit on the same lap
# But use different AMOUNTS of each compound via different pit laps

# For two drivers with same compounds in same order but different pit laps:
# time_A - time_B = cb_s * (n_s_A - n_s_B) + cb_h * (n_h_A - n_h_B)
#                 + rs * ts * (se_s_A - se_s_B) + rh * ts * (se_h_A - se_h_B)

# Since n_s + n_h = total_laps: n_h_A - n_h_B = -(n_s_A - n_s_B)
# So: time_diff = (cb_s - cb_h) * delta_n_s + rs*ts*delta_se_s + rh*ts*delta_se_h

def sum_excess(N, C):
    if N <= C:
        return 0.0
    k = N - C
    return k * (k + 1) / 2.0

# For each moderate-temp race, find same-sequence driver pairs
print("Finding pairs...")
pairs = []  # (delta_n_s, delta_se_s, delta_se_h, sign, config)
# sign: +1 if driver_a finishes before driver_b (lower time)

cliffs = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}

for race in races:
    config = race["race_config"]
    temp = config["track_temp"]
    if not (27 <= temp <= 31):
        continue

    fp = race["finishing_positions"]
    strategies = race["strategies"]

    # Group by compound sequence
    groups = defaultdict(list)
    for pk in strategies:
        s = strategies[pk]
        seq = [s["starting_tire"]]
        pit_details = []
        for ps in sorted(s["pit_stops"], key=lambda x: x["lap"]):
            seq.append(ps["to_tire"])
            pit_details.append(ps["lap"])
        key = tuple(seq)
        did = s["driver_id"]
        pos = fp.index(did)
        groups[key].append((did, pit_details, pos))

    for key, drivers in groups.items():
        if len(drivers) < 2 or len(key) != 2:  # only 1-pit strategies
            continue
        comp1, comp2 = key

        for i in range(len(drivers)):
            for j in range(i+1, len(drivers)):
                d1_id, d1_pits, d1_pos = drivers[i]
                d2_id, d2_pits, d2_pos = drivers[j]

                if d1_pos == d2_pos:
                    continue

                # Driver 1 pits at d1_pits[0], driver 2 at d2_pits[0]
                n1_c1 = d1_pits[0]  # laps on comp1
                n1_c2 = config["total_laps"] - d1_pits[0]  # laps on comp2
                n2_c1 = d2_pits[0]
                n2_c2 = config["total_laps"] - d2_pits[0]

                se1_c1 = sum_excess(n1_c1, cliffs[comp1])
                se1_c2 = sum_excess(n1_c2, cliffs[comp2])
                se2_c1 = sum_excess(n2_c1, cliffs[comp1])
                se2_c2 = sum_excess(n2_c2, cliffs[comp2])

                # Winner has lower position number
                if d1_pos < d2_pos:
                    # d1 is faster: time_d1 < time_d2
                    # time_d1 - time_d2 < 0
                    # Both have same number of pits (1)
                    # delta = features(d1) - features(d2)
                    delta_n_c1 = n1_c1 - n2_c1
                    delta_se_c1 = se1_c1 - se2_c1
                    delta_se_c2 = se1_c2 - se2_c2
                    # Constraint: (cb_c1 - cb_c2) * delta_n_c1 + rate_c1*ts*delta_se_c1 + rate_c2*ts*delta_se_c2 < 0
                    pairs.append({
                        "comp1": comp1,
                        "comp2": comp2,
                        "delta_n_c1": delta_n_c1,
                        "delta_se_c1": delta_se_c1,
                        "delta_se_c2": delta_se_c2,
                        "temp": temp,
                        "sign": -1,  # should be negative
                    })
                else:
                    delta_n_c1 = n1_c1 - n2_c1
                    delta_se_c1 = se1_c1 - se2_c1
                    delta_se_c2 = se1_c2 - se2_c2
                    pairs.append({
                        "comp1": comp1,
                        "comp2": comp2,
                        "delta_n_c1": delta_n_c1,
                        "delta_se_c1": delta_se_c1,
                        "delta_se_c2": delta_se_c2,
                        "temp": temp,
                        "sign": 1,  # should be positive
                    })

print(f"Found {len(pairs)} constrained pairs")

# Now let's check: for the 58/100 params, how many constraints are satisfied?
def check_constraints(cb_s, cb_h, rs, rm, rh, tc, tref):
    cb = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}

    satisfied = 0
    total = 0
    for p in pairs:
        c1, c2 = p["comp1"], p["comp2"]
        temp = p["temp"]
        ts = 1.0 + tc * (temp - tref)

        value = (cb[c1] - cb[c2]) * p["delta_n_c1"] + rate[c1]*ts*p["delta_se_c1"] + rate[c2]*ts*p["delta_se_c2"]

        total += 1
        if p["sign"] < 0 and value < 0:
            satisfied += 1
        elif p["sign"] > 0 and value > 0:
            satisfied += 1

    return satisfied, total

# Current per-compound params
s1, t1 = check_constraints(-0.9665, 0.7553, 1.6214, 0.8133, 0.3460, 0.027, 27.97)
print(f"Per-compound params: {s1}/{t1} ({s1/t1*100:.1f}%)")

# New single-tc params
s2, t2 = check_constraints(-1.0, 0.8, 1.5, 0.75, 0.4, 0.029, 24.5)
print(f"Simple params (58/100): {s2}/{t2} ({s2/t2*100:.1f}%)")

# Try a range of params and find the best constraint satisfaction
print("\nSearching for best constraint satisfaction...")
best_sat = 0
best_p = None
for cb_s in np.arange(-1.3, -0.6, 0.1):
    for cb_h in np.arange(0.4, 1.2, 0.1):
        for rs in np.arange(1.0, 2.5, 0.1):
            for rm in np.arange(0.4, 1.2, 0.1):
                for rh in np.arange(0.2, 0.7, 0.1):
                    for tc in [0.02, 0.025, 0.03]:
                        for tref in [24, 25, 28, 30]:
                            s, t = check_constraints(cb_s, cb_h, rs, rm, rh, tc, tref)
                            if s > best_sat:
                                best_sat = s
                                best_p = (cb_s, cb_h, rs, rm, rh, tc, tref)

print(f"Best: {best_sat}/{t1} ({best_sat/t1*100:.1f}%)")
print(f"Params: {best_p}")
