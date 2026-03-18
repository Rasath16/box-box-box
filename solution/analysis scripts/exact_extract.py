"""
Extract EXACT parameters from historical data by finding
carefully controlled pairs of drivers.

Key idea: Find two drivers in the same race where the ONLY
difference is one variable. Then solve for that variable's effect.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("Loading races...")
races = []
for f in sorted(Path("data/historical_races").glob("races_*.json")):
    with open(f) as fh:
        races.extend(json.load(fh))
print(f"Loaded {len(races)} races")

def get_stints(strategy, total_laps):
    """Get stints: list of (compound, laps_on_compound)"""
    pit_laps = sorted([(ps["lap"], ps["to_tire"]) for ps in strategy["pit_stops"]])
    stints = []
    compound = strategy["starting_tire"]
    start = 1
    for plap, new_comp in pit_laps:
        stints.append((compound, plap - start + 1))
        compound = new_comp
        start = plap + 1
    stints.append((compound, total_laps - start + 1))
    return stints

def get_strategy_key(strategy, total_laps):
    """A string that describes the strategy fully."""
    stints = get_stints(strategy, total_laps)
    return str(stints)

# For a single compound stint of N laps with cliff C:
# sum of degradation = rate * temp_scale * sum(max(0, age - C) for age in 1..N)
# = rate * temp_scale * sum(max(0, i - C) for i in 1..N)
# If N <= C: sum = 0
# If N > C: sum = sum(i - C for i in C+1..N) = sum(j for j in 1..N-C) = (N-C)(N-C+1)/2

def sum_excess(N, C):
    """Sum of max(0, age - C) for age in 1 to N."""
    if N <= C:
        return 0
    k = N - C
    return k * (k + 1) / 2

# Find pairs where both drivers use SAME compounds in SAME order,
# SAME number of laps per stint, but differ in pit timing
# This is hard because pit timing determines stint length.

# Instead, let's find races where specific compound pairs exist
# and use them to extract parameters.

# APPROACH 1: Extract compound_base from races at reference temperature
# Find races near temp=28, with two drivers whose strategies are
# identical except one uses more laps of SOFT and fewer of HARD (or vice versa)

# Actually, simpler: for two drivers with the same number of pit stops,
# same compound sequence, different pit timing, the time difference
# is purely a function of degradation rates.

# Let me find same-compound-sequence, different-timing pairs
print("\n=== Finding controlled pairs ===")

# Group drivers by (race_id, compound_sequence, n_pits)
pair_data = []
for race in races[:5000]:
    config = race["race_config"]
    fp = race["finishing_positions"]
    total_laps = config["total_laps"]
    temp = config["track_temp"]
    base = config["base_lap_time"]
    pit_time = config["pit_lane_time"]

    # Skip extreme temps for now - focus on moderate
    if abs(temp - 28) > 2:
        continue

    strategies = race["strategies"]
    # Group by compound sequence
    groups = defaultdict(list)
    for pk in strategies:
        s = strategies[pk]
        did = s["driver_id"]
        compounds = [s["starting_tire"]]
        pit_details = []
        for ps in sorted(s["pit_stops"], key=lambda x: x["lap"]):
            compounds.append(ps["to_tire"])
            pit_details.append(ps["lap"])
        key = tuple(compounds)
        # Find position in finishing order
        pos = fp.index(did) if did in fp else -1
        groups[key].append((did, pit_details, pos, s))

    # For groups with > 1 driver (same compound sequence)
    for key, drivers in groups.items():
        if len(drivers) < 2:
            continue
        # All these drivers use the same compounds in the same order
        # The only difference is pit timing
        for i in range(len(drivers)):
            for j in range(i+1, len(drivers)):
                d1_id, d1_pits, d1_pos, d1_strat = drivers[i]
                d2_id, d2_pits, d2_pos, d2_strat = drivers[j]
                if d1_pos == d2_pos:
                    continue
                pair_data.append({
                    "compounds": key,
                    "d1_pits": d1_pits,
                    "d2_pits": d2_pits,
                    "d1_pos": d1_pos,
                    "d2_pos": d2_pos,
                    "config": config,
                    "d1_strat": d1_strat,
                    "d2_strat": d2_strat,
                })

print(f"Found {len(pair_data)} controlled pairs")

# For these pairs, we know the ordering. Let's see what constraints
# they give us. The time difference is:
# time_d1 - time_d2 = sum of (degradation differences per lap)
# For same compound sequence, the compound bases cancel out per stint.

# Let me compute: for each pair, what are the degradation sums?
# time = n_laps * base + n_pits * pit_time + sum(cb[c] * n_laps_c) + sum(rate[c] * ts * sum_excess(N_c, cliff_c))

# Since both drivers have same compound sequence and same number of pits:
# time_d1 - time_d2 = sum over stints: rate[c] * ts * (sum_excess(N1_c, cliff_c) - sum_excess(N2_c, cliff_c))

# where N1_c and N2_c are the stint lengths for each driver.

# Wait, both drivers have DIFFERENT pit laps, so their stint lengths differ.
# But compound bases depend on stint length (number of laps on each compound).

# Actually no: cb[c] is per-lap. If d1 has M1 laps on SOFT and M2 on HARD,
# and d2 has M1' on SOFT and M2' on HARD, but M1+M2 = M1'+M2' = total_laps,
# then:
# sum(cb) for d1 = cb_s * M1 + cb_h * M2
# sum(cb) for d2 = cb_s * M1' + cb_h * M2'
# These DON'T cancel unless M1=M1' (which is only true if same pit timing)

# Hmm, so same-sequence pairs with different timing are NOT purely degradation
# because the compound base contributions differ.

# Let me instead look for pairs where drivers have DIFFERENT compound sequences
# but the comparison gives us information.

# Actually, the simplest case: find two drivers with IDENTICAL strategies.
# They should have IDENTICAL times and thus adjacent positions.
# The relative order between them is determined by grid position tiebreaker.

# Let me verify this...
identical_pairs = 0
identical_correct = 0
for race in races[:5000]:
    config = race["race_config"]
    fp = race["finishing_positions"]
    total_laps = config["total_laps"]
    strategies = race["strategies"]

    strat_groups = defaultdict(list)
    for pk in strategies:
        s = strategies[pk]
        key = get_strategy_key(s, total_laps)
        grid = int(pk[3:])
        strat_groups[key].append((s["driver_id"], grid))

    for key, drivers in strat_groups.items():
        if len(drivers) < 2:
            continue
        # These drivers should have identical times -> ordered by grid
        drivers_sorted_by_grid = sorted(drivers, key=lambda x: x[1])
        expected_order = [d for d, g in drivers_sorted_by_grid]

        # Check actual order
        actual_positions = {d: fp.index(d) for d, g in drivers}
        actual_order = sorted(drivers, key=lambda x: actual_positions[x[0]])
        actual_order = [d for d, g in actual_order]

        for i in range(len(expected_order)):
            identical_pairs += 1
            if expected_order[i] == actual_order[i]:
                identical_correct += 1

print(f"\n=== Identical strategy pairs ===")
print(f"Total: {identical_pairs}, Correct by grid tiebreak: {identical_correct}")
print(f"Accuracy: {identical_correct/identical_pairs*100:.1f}%")

# Now let me try to extract compound bases from specific controlled cases.
# Find race at temp~28 with two drivers:
# Driver A: all MEDIUM (or MEDIUM -> MEDIUM through pit)
# Driver B: some other strategy
# Then we can compute: time_A = total_laps * (base + cb_med + ...) + pits * pit_time
# etc.

# Better: find two single-stint-equivalent drivers (same total laps, no pits... wait, must use 2 compounds)

# Let me try extracting from the regression on MODERATE temp races (26-30)
# where we have the most data and the temp effect is minimal.
print("\n=== Regression on moderate-temp races (26-30) ===")

A_rows = []
b_rows = []

for race in races:
    config = race["race_config"]
    if not (26 <= config["track_temp"] <= 30):
        continue
    fp = race["finishing_positions"]
    strategies = race["strategies"]
    d2s = {}
    for pk in strategies:
        s = strategies[pk]
        d2s[s["driver_id"]] = s

    for idx in range(len(fp) - 1):
        dw, dl = fp[idx], fp[idx+1]
        sw, sl = d2s[dw], d2s[dl]

        stints_w = get_stints(sw, config["total_laps"])
        stints_l = get_stints(sl, config["total_laps"])

        # Features: n_laps_soft_w - n_laps_soft_l, n_laps_hard_w - n_laps_hard_l,
        # sum_excess_soft_w - ..., etc.
        n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
        excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}

        for comp, n in stints_w:
            n_laps[comp] += n
            excess[comp] += sum_excess(n, {"SOFT": 10, "MEDIUM": 20, "HARD": 29}[comp])
        fw = [n_laps["SOFT"], n_laps["HARD"], excess["SOFT"], excess["MEDIUM"], excess["HARD"]]

        n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
        excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        for comp, n in stints_l:
            n_laps[comp] += n
            excess[comp] += sum_excess(n, {"SOFT": 10, "MEDIUM": 20, "HARD": 29}[comp])
        fl = [n_laps["SOFT"], n_laps["HARD"], excess["SOFT"], excess["MEDIUM"], excess["HARD"]]

        delta = [fw[i] - fl[i] for i in range(5)]
        pit_diff = (len(sl["pit_stops"]) - len(sw["pit_stops"])) * config["pit_lane_time"]
        A_rows.append(delta)
        b_rows.append(pit_diff)

A = np.array(A_rows)
b = np.array(b_rows)
params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
cb_s, cb_h, rs, rm, rh = params
print(f"cb_s={cb_s:.10f} cb_h={cb_h:.10f}")
print(f"rs={rs:.10f} rm={rm:.10f} rh={rh:.10f}")

# Test these params (no temp effect) on test cases
tests_data = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests_data.append((inp, exp["finishing_positions"]))

correct = 0
for inp, exp in tests_data:
    config = inp["race_config"]
    results = []
    for pk in sorted(inp["strategies"].keys(), key=lambda k: int(k[3:])):
        s = inp["strategies"][pk]
        grid = int(pk[3:])
        stints = get_stints(s, config["total_laps"])
        total_time = config["total_laps"] * config["base_lap_time"] + len(s["pit_stops"]) * config["pit_lane_time"]
        for comp, n in stints:
            total_time += cb_s * n if comp == "SOFT" else (cb_h * n if comp == "HARD" else 0)
            cliff = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}[comp]
            rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}[comp]
            total_time += rate * sum_excess(n, cliff)
        results.append((total_time, grid, s["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    pred = [d for _, _, d in results]
    if pred == exp:
        correct += 1

print(f"Score with regression params (no temp): {correct}/100")

# Now do regression for EACH temperature to get rate vs temp
print("\n=== Per-temperature regression ===")
temp_rates = {}
for target_temp in sorted(set(r["race_config"]["track_temp"] for r in races)):
    A_rows = []
    b_rows = []
    for race in races:
        config = race["race_config"]
        if config["track_temp"] != target_temp:
            continue
        fp = race["finishing_positions"]
        strategies = race["strategies"]
        d2s = {}
        for pk in strategies:
            s = strategies[pk]
            d2s[s["driver_id"]] = s

        for idx in range(len(fp) - 1):
            dw, dl = fp[idx], fp[idx+1]
            sw, sl = d2s[dw], d2s[dl]
            stints_w = get_stints(sw, config["total_laps"])
            stints_l = get_stints(sl, config["total_laps"])

            n_laps_w = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
            excess_w = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            for comp, n in stints_w:
                n_laps_w[comp] += n
                excess_w[comp] += sum_excess(n, {"SOFT": 10, "MEDIUM": 20, "HARD": 29}[comp])

            n_laps_l = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
            excess_l = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            for comp, n in stints_l:
                n_laps_l[comp] += n
                excess_l[comp] += sum_excess(n, {"SOFT": 10, "MEDIUM": 20, "HARD": 29}[comp])

            delta = [
                n_laps_w["SOFT"] - n_laps_l["SOFT"],
                n_laps_w["HARD"] - n_laps_l["HARD"],
                excess_w["SOFT"] - excess_l["SOFT"],
                excess_w["MEDIUM"] - excess_l["MEDIUM"],
                excess_w["HARD"] - excess_l["HARD"],
            ]
            pit_diff = (len(sl["pit_stops"]) - len(sw["pit_stops"])) * config["pit_lane_time"]
            A_rows.append(delta)
            b_rows.append(pit_diff)

    if len(A_rows) < 100:
        continue
    A = np.array(A_rows)
    b = np.array(b_rows)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    temp_rates[target_temp] = params
    print(f"  temp={target_temp}: cb_s={params[0]:.8f} cb_h={params[1]:.8f} rs={params[2]:.8f} rm={params[3]:.8f} rh={params[4]:.8f}")

# Analyze: are compound bases constant? Do rates vary linearly with temp?
temps = sorted(temp_rates.keys())
print(f"\nTemps: {temps}")

print("\n=== Rate ratios across temperatures ===")
for t in temps:
    p = temp_rates[t]
    rs, rm, rh = p[2], p[3], p[4]
    if rm > 0:
        print(f"  temp={t}: rs/rm={rs/rm:.4f} rh/rm={rh/rm:.4f} rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")
