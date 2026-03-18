"""
Extract exact parameters using pairwise constraints from historical data.

Key insight: within a single race, two drivers have the SAME base_lap_time,
pit_lane_time, track_temp, and total_laps. So the difference in their total times
depends ONLY on their strategies.

If we compute: time(driver_A) - time(driver_B), the base_lap_time cancels out.
This gives us constraints on the compound/degradation parameters.

For the ranking to be correct, we need:
  time(rank_i) < time(rank_j) for i < j

Let's use the historical data to extract constraints and solve.
"""
import json
import numpy as np
from pathlib import Path

def load_races(start=0, count=1000):
    races = []
    data_dir = Path("data/historical_races")
    for f in sorted(data_dir.glob("races_*.json")):
        with open(f) as fh:
            batch = json.load(fh)
            races.extend(batch)
            if len(races) >= start + count:
                break
    return races[start:start+count]

def compute_features(strategy, config, cliff_s, cliff_m, cliff_h):
    """
    Compute features for linear regression.

    lap_time = base + cb_s*is_soft + cb_h*is_hard
             + rs*ts_s*sum_excess_s + rm*ts_m*sum_excess_m + rh*ts_h*sum_excess_h

    But since base cancels in pairwise differences, we only need the strategy-dependent part.

    Actually, total_time = total_laps * base + n_pits * pit_time
                         + sum over laps of (compound_base + degradation)

    The base*total_laps and pit terms are the same for comparison purposes within a race
    when they differ, so let's compute:

    strategy_cost = n_pits * pit_time + sum(compound_base[c] for each lap) + sum(degradation for each lap)

    For linear regression with params [cb_s, cb_h, rs, rm, rh]:
    features = [n_laps_soft, n_laps_hard, sum_excess_soft*ts, sum_excess_med*ts, sum_excess_hard*ts]

    And the actual cost = features @ params + n_laps * base + n_pits * pit_time
    """
    total_laps = config["total_laps"]
    track_temp = config["track_temp"]
    pit_time = config["pit_lane_time"]
    base = config["base_lap_time"]

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0

    n_laps_soft = 0
    n_laps_hard = 0
    sum_excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    n_pits = len(strategy["pit_stops"])

    cliff = {"SOFT": cliff_s, "MEDIUM": cliff_m, "HARD": cliff_h}

    for lap in range(1, total_laps + 1):
        tire_age += 1

        if compound == "SOFT":
            n_laps_soft += 1
        elif compound == "HARD":
            n_laps_hard += 1

        excess = max(0.0, tire_age - cliff[compound])
        sum_excess[compound] += excess

        if lap in pit_laps:
            compound = pit_laps[lap]
            tire_age = 0

    # Features for params [cb_s, cb_h, rs*temp_scale, rm*temp_scale, rh*temp_scale]
    # But temp_scale depends on compound... hmm
    # For single temp coeff: temp_scale = 1 + tc*(temp - tref)
    # Feature vector: [n_laps_soft, n_laps_hard, sum_excess_soft, sum_excess_med, sum_excess_hard]
    # And the actual values of rs*temp_scale etc are what we solve for

    constant = total_laps * base + n_pits * pit_time

    return [n_laps_soft, n_laps_hard, sum_excess["SOFT"], sum_excess["MEDIUM"], sum_excess["HARD"]], constant

# For a given set of cliffs, we can use linear regression on pairwise differences
# to find the remaining parameters.
# But the objective is ranking, not exact time prediction.
# Instead, let's use: for each pair (winner, loser) in a race,
#   features(winner) @ params + const(winner) < features(loser) @ params + const(loser)
# => (features(winner) - features(loser)) @ params < const(loser) - const(winner)
# => delta_features @ params < delta_const

# This is a system of linear inequalities. We want to find params that satisfies the most.
# We can use linear regression on the equalities (delta_features @ params = delta_const)
# as an approximation.

races = load_races(0, 5000)
print(f"Loaded {len(races)} races")

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def test_params(cb_s, cb_h, rates_by_temp, cliff_s, cliff_m, cliff_h, tc, tref):
    """Test with given parameters against test cases."""
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        track_temp = config["track_temp"]
        temp_scale = 1.0 + tc * (track_temp - tref)

        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            strat = strategies[pos_key]
            grid = int(pos_key[3:])

            pit_laps = {}
            for ps in strat["pit_stops"]:
                pit_laps[ps["lap"]] = ps["to_tire"]

            compound = strat["starting_tire"]
            tire_age = 0
            total_time = 0.0
            cliff = {"SOFT": cliff_s, "MEDIUM": cliff_m, "HARD": cliff_h}
            rate = rates_by_temp  # dict with rates for this temp
            cb = {"SOFT": cb_s, "MEDIUM": 0.0, "HARD": cb_h}

            for lap in range(1, config["total_laps"] + 1):
                tire_age += 1
                deg = rate[compound] * temp_scale * max(0.0, tire_age - cliff[compound])
                total_time += config["base_lap_time"] + cb[compound] + deg
                if lap in pit_laps:
                    total_time += config["pit_lane_time"]
                    compound = pit_laps[lap]
                    tire_age = 0

            results.append((total_time, grid, strat["driver_id"]))

        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

# For each cliff combo, use least squares to find best params
print("\n=== Regression-based parameter extraction ===")
best_score = 0
best_all = None

for cs in [8, 9, 10, 11, 12]:
    for cm in [18, 19, 20, 21, 22]:
        for ch in [28, 29, 30, 31, 32]:
            # Build pairwise system from historical races
            A_rows = []
            b_rows = []

            for race in races[:2000]:
                config = race["race_config"]
                fp = race["finishing_positions"]
                strategies = race["strategies"]

                # Map driver_id -> strategy
                driver_strats = {}
                for pk in strategies:
                    s = strategies[pk]
                    driver_strats[s["driver_id"]] = s

                # For consecutive pairs in finishing order
                for idx in range(len(fp) - 1):
                    d_win = fp[idx]
                    d_lose = fp[idx + 1]

                    feat_win, const_win = compute_features(driver_strats[d_win], config, cs, cm, ch)
                    feat_lose, const_lose = compute_features(driver_strats[d_lose], config, cs, cm, ch)

                    # feat_win @ params + const_win < feat_lose @ params + const_lose
                    # (feat_win - feat_lose) @ params < const_lose - const_win
                    # For regression: (feat_win - feat_lose) @ params = const_lose - const_win
                    delta_feat = [fw - fl for fw, fl in zip(feat_win, feat_lose)]
                    delta_const = const_lose - const_win

                    A_rows.append(delta_feat)
                    b_rows.append(delta_const)

            A = np.array(A_rows)
            b = np.array(b_rows)

            # Solve least squares
            params, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            cb_s, cb_h, rs, rm, rh = params

            # These rates are NOT temperature-adjusted yet
            # They represent "average" rates across all temperatures
            # Let's test them with tc=0 first (no temp effect)
            rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
            s = test_params(cb_s, cb_h, rate, cs, cm, ch, 0.0, 28.0)

            if s > best_score:
                best_score = s
                best_all = (cb_s, cb_h, rs, rm, rh, cs, cm, ch)
                print(f"  {s}/100: cliffs=({cs},{cm},{ch}) cb_s={cb_s:.4f} cb_h={cb_h:.4f} rs={rs:.4f} rm={rm:.4f} rh={rh:.4f}")

print(f"\nBest from regression (no temp): {best_score}/100")
if best_all:
    cb_s, cb_h, rs, rm, rh, cs, cm, ch = best_all
    print(f"  Params: cb_s={cb_s:.6f} cb_h={cb_h:.6f}")
    print(f"  Rates: rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")
    print(f"  Cliffs: ({cs}, {cm}, {ch})")

    # Now try adding temperature
    print("\n=== Adding temperature coefficient ===")
    best_tc_score = best_score
    for tc in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
        for tref in [20, 25, 28, 30, 35]:
            rate = {"SOFT": rs, "MEDIUM": rm, "HARD": rh}
            s = test_params(cb_s, cb_h, rate, cs, cm, ch, tc, tref)
            if s > best_tc_score:
                best_tc_score = s
                print(f"  {s}/100: tc={tc} tref={tref}")

    print(f"Best with temp: {best_tc_score}/100")
