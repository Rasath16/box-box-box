"""
Check if parameters differ by track. The huge accuracy variation
(Monaco 31% vs Monza 74%) suggests track-specific effects.
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

def sum_excess(N, C):
    if N <= C:
        return 0
    k = N - C
    return k * (k + 1) / 2

# Group by track and temperature
track_temp_races = defaultdict(list)
for r in races:
    key = (r["race_config"]["track"], r["race_config"]["track_temp"])
    track_temp_races[key].append(r)

# Do regression per track (across all temps)
print("\n=== Per-track regression (all temps) ===")
CLIFFS = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}

for track in sorted(set(r["race_config"]["track"] for r in races)):
    track_races = [r for r in races if r["race_config"]["track"] == track]

    A_rows = []
    b_rows = []

    for race in track_races:
        config = race["race_config"]
        fp = race["finishing_positions"]
        strategies = race["strategies"]
        d2s = {strategies[pk]["driver_id"]: strategies[pk] for pk in strategies}

        for idx in range(len(fp) - 1):
            dw, dl = fp[idx], fp[idx+1]
            sw, sl = d2s[dw], d2s[dl]

            for driver, feat_dict in [(sw, {}), (sl, {})]:
                stints = get_stints(driver, config["total_laps"])
                n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
                excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
                for comp, n in stints:
                    n_laps[comp] += n
                    excess[comp] += sum_excess(n, CLIFFS[comp])
                feat_dict["n_s"] = n_laps["SOFT"]
                feat_dict["n_h"] = n_laps["HARD"]
                feat_dict["e_s"] = excess["SOFT"]
                feat_dict["e_m"] = excess["MEDIUM"]
                feat_dict["e_h"] = excess["HARD"]

            fw = feat_dict  # This is wrong, let me fix
            # Need to redo this properly
            stints_w = get_stints(sw, config["total_laps"])
            stints_l = get_stints(sl, config["total_laps"])

            def get_features(strategy, total_laps):
                stints = get_stints(strategy, total_laps)
                n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
                excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
                for comp, n in stints:
                    n_laps[comp] += n
                    excess[comp] += sum_excess(n, CLIFFS[comp])
                return [n_laps["SOFT"], n_laps["HARD"], excess["SOFT"], excess["MEDIUM"], excess["HARD"]]

            fw = get_features(sw, config["total_laps"])
            fl = get_features(sl, config["total_laps"])
            delta = [fw[i] - fl[i] for i in range(5)]
            pit_diff = (len(sl["pit_stops"]) - len(sw["pit_stops"])) * config["pit_lane_time"]
            A_rows.append(delta)
            b_rows.append(pit_diff)

    A = np.array(A_rows)
    b = np.array(b_rows)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cb_s, cb_h, rs, rm, rh = params
    print(f"  {track:12s}: cb_s={cb_s:.6f} cb_h={cb_h:.6f} rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")

# Do regression per track at FIXED temp (28) to remove temp variation
print("\n=== Per-track regression (temp=28 only) ===")
for track in sorted(set(r["race_config"]["track"] for r in races)):
    track_races = [r for r in races if r["race_config"]["track"] == track and r["race_config"]["track_temp"] == 28]
    if len(track_races) < 50:
        continue

    A_rows = []
    b_rows = []
    for race in track_races:
        config = race["race_config"]
        fp = race["finishing_positions"]
        strategies = race["strategies"]
        d2s = {strategies[pk]["driver_id"]: strategies[pk] for pk in strategies}

        def get_features(strategy, total_laps):
            stints = get_stints(strategy, total_laps)
            n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
            excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            for comp, n in stints:
                n_laps[comp] += n
                excess[comp] += sum_excess(n, CLIFFS[comp])
            return [n_laps["SOFT"], n_laps["HARD"], excess["SOFT"], excess["MEDIUM"], excess["HARD"]]

        for idx in range(len(fp) - 1):
            dw, dl = fp[idx], fp[idx+1]
            sw, sl = d2s[dw], d2s[dl]
            fw = get_features(sw, config["total_laps"])
            fl = get_features(sl, config["total_laps"])
            delta = [fw[i] - fl[i] for i in range(5)]
            pit_diff = (len(sl["pit_stops"]) - len(sw["pit_stops"])) * config["pit_lane_time"]
            A_rows.append(delta)
            b_rows.append(pit_diff)

    A = np.array(A_rows)
    b = np.array(b_rows)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cb_s, cb_h, rs, rm, rh = params
    print(f"  {track:12s} ({len(track_races)} races): cb_s={cb_s:.6f} cb_h={cb_h:.6f} rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")

# Check: does accuracy change if we use per-track params?
# First, extract per-track params at the REFERENCE temperature
print("\n=== Per-track params at temp=30 ===")
for track in sorted(set(r["race_config"]["track"] for r in races)):
    track_races = [r for r in races if r["race_config"]["track"] == track and r["race_config"]["track_temp"] == 30]
    if len(track_races) < 50:
        continue

    A_rows = []
    b_rows = []
    for race in track_races:
        config = race["race_config"]
        fp = race["finishing_positions"]
        strategies = race["strategies"]
        d2s = {strategies[pk]["driver_id"]: strategies[pk] for pk in strategies}

        def get_features(strategy, total_laps):
            stints = get_stints(strategy, total_laps)
            n_laps = {"SOFT": 0, "MEDIUM": 0, "HARD": 0}
            excess = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            for comp, n in stints:
                n_laps[comp] += n
                excess[comp] += sum_excess(n, CLIFFS[comp])
            return [n_laps["SOFT"], n_laps["HARD"], excess["SOFT"], excess["MEDIUM"], excess["HARD"]]

        for idx in range(len(fp) - 1):
            dw, dl = fp[idx], fp[idx+1]
            sw, sl = d2s[dw], d2s[dl]
            fw = get_features(sw, config["total_laps"])
            fl = get_features(sl, config["total_laps"])
            delta = [fw[i] - fl[i] for i in range(5)]
            pit_diff = (len(sl["pit_stops"]) - len(sw["pit_stops"])) * config["pit_lane_time"]
            A_rows.append(delta)
            b_rows.append(pit_diff)

    A = np.array(A_rows)
    b = np.array(b_rows)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cb_s, cb_h, rs, rm, rh = params
    print(f"  {track:12s} ({len(track_races)} races): cb_s={cb_s:.6f} cb_h={cb_h:.6f} rs={rs:.6f} rm={rm:.6f} rh={rh:.6f}")
