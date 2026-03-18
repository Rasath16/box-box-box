"""
Grid search over simple/round parameter values.
"Easier than you think" suggests the true parameters might be round numbers.
"""
import json
import itertools

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def compute_time(strategy, config, cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    compound_base = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
    deg_rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
    cliff = {"SOFT": cs, "MEDIUM": cm, "HARD": ch}

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0
    temp_scale = 1.0 + tc * (track_temp - tref)

    for lap in range(1, total_laps + 1):
        tire_age += 1
        cb = compound_base[compound]
        rate = deg_rate[compound]
        c = cliff[compound]
        deg = rate * temp_scale * max(0.0, tire_age - c)
        total_time += base + cb + deg
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

def count_correct(cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            strat = strategies[pos_key]
            grid = int(pos_key[3:])
            t = compute_time(strat, config, cb_s, cb_h, rs, rm, rh, cs, cm, ch, tc, tref)
            results.append((t, grid, strat["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

best = 0
# Search round values
# Focus on the most impactful: temp_coeff and temp_ref
# since that's where most error comes from
print("=== Grid search: round params ===")

for tref in [25, 28, 30]:
    for tc in [0.02, 0.025, 0.03]:
        for cb_s in [-1.0, -0.5, -0.8]:
            for cb_h in [0.5, 0.7, 0.8, 1.0]:
                for rs in [1.0, 1.5, 2.0]:
                    for rm in [0.5, 0.75, 1.0]:
                        for rh in [0.25, 0.35, 0.5]:
                            # Fixed cliffs
                            s = count_correct(cb_s, cb_h, rs, rm, rh, 10, 20, 30, tc, tref)
                            if s > best:
                                best = s
                                print(f"  {s}/100: cb_s={cb_s} cb_h={cb_h} rs={rs} rm={rm} rh={rh} cs=10 cm=20 ch=30 tc={tc} tref={tref}")

print(f"\nBest with round params: {best}/100")

# Now try per-compound temp coefficients with round values
print("\n=== Per-compound temp coeff with round values ===")
best2 = 0
for tref in [25, 28, 30]:
    for tc_s in [0.02, 0.025, 0.03]:
        for tc_m in [0.02, 0.025, 0.03]:
            for tc_h in [0.02, 0.025, 0.03]:
                # Use our best-ish base params
                for cb_s in [-1.0, -0.5]:
                    for cb_h in [0.5, 0.8]:
                        for rs in [1.0, 1.5, 2.0]:
                            for rm in [0.5, 0.75, 1.0]:
                                for rh in [0.25, 0.35, 0.5]:
                                    # Per-compound temp
                                    correct = 0
                                    for inp, exp in tests:
                                        config = inp["race_config"]
                                        strategies = inp["strategies"]
                                        results = []
                                        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
                                            strat = strategies[pos_key]
                                            grid = int(pos_key[3:])
                                            track_temp = config["track_temp"]

                                            pit_laps = {}
                                            for ps in strat["pit_stops"]:
                                                pit_laps[ps["lap"]] = ps["to_tire"]

                                            compound = strat["starting_tire"]
                                            tire_age = 0
                                            total_time = 0.0

                                            cb_map = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}
                                            rate_map = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
                                            tc_map = {"SOFT": tc_s, "MEDIUM": tc_m, "HARD": tc_h}
                                            cliff_map = {"SOFT": 10, "MEDIUM": 20, "HARD": 30}

                                            for lap in range(1, config["total_laps"] + 1):
                                                tire_age += 1
                                                ts = 1.0 + tc_map[compound] * (track_temp - tref)
                                                deg = rate_map[compound] * ts * max(0.0, tire_age - cliff_map[compound])
                                                total_time += config["base_lap_time"] + cb_map[compound] + deg
                                                if lap in pit_laps:
                                                    total_time += config["pit_lane_time"]
                                                    compound = pit_laps[lap]
                                                    tire_age = 0

                                            results.append((total_time, grid, strat["driver_id"]))
                                        results.sort(key=lambda r: (r[0], r[1]))
                                        pred = [d for _, _, d in results]
                                        if pred == exp:
                                            correct += 1

                                    if correct > best2:
                                        best2 = correct
                                        print(f"  {correct}/100: cb_s={cb_s} cb_h={cb_h} rs={rs} rm={rm} rh={rh} tc_s={tc_s} tc_m={tc_m} tc_h={tc_h} tref={tref}")

print(f"\nBest with per-compound round params: {best2}/100")
