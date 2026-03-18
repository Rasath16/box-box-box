"""
Try radically different formula structures.
The organizers said "it's easier than you think".
"""
import json
import random
import math

# Load test cases
tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

def simulate(strategy, config, lap_time_fn):
    """Generic simulator - lap_time_fn(compound, tire_age, track_temp, base) -> lap_time"""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += lap_time_fn(compound, tire_age, track_temp, base)
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

    return total_time

def predict(race_data, lap_time_fn):
    config = race_data["race_config"]
    strategies = race_data["strategies"]
    results = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strat = strategies[pos_key]
        grid = int(pos_key[3:])
        t = simulate(strat, config, lap_time_fn)
        results.append((t, grid, strat["driver_id"]))
    results.sort(key=lambda r: (r[0], r[1]))
    return [d for _, _, d in results]

def test(lap_time_fn):
    correct = 0
    for inp, exp in tests:
        if predict(inp, lap_time_fn) == exp:
            correct += 1
    return correct

# Parameters
CB = {"SOFT": -0.9665, "MEDIUM": 0.0, "HARD": 0.7553}
RATE = {"SOFT": 1.6214, "MEDIUM": 0.8133, "HARD": 0.3460}
CLIFF = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}
TC = {"SOFT": 0.02581, "MEDIUM": 0.02777, "HARD": 0.02402}
TREF = 27.966

# Formula 1: Current (baseline)
def f1(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    return base + CB[c] + RATE[c] * ts * max(0.0, age - CLIFF[c])

print(f"F1 (current linear cliff): {test(f1)}/100")

# Formula 2: Quadratic degradation past cliff
def f2(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    excess = max(0.0, age - CLIFF[c])
    return base + CB[c] + RATE[c] * ts * excess * excess

print(f"F2 (quadratic past cliff): {test(f2)}/100")

# Formula 3: Square root degradation
def f3(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    excess = max(0.0, age - CLIFF[c])
    return base + CB[c] + RATE[c] * ts * math.sqrt(excess)

print(f"F3 (sqrt past cliff): {test(f3)}/100")

# Formula 4: No compound base - only degradation differs
RATE4 = {"SOFT": 0.5, "MEDIUM": 0.3, "HARD": 0.15}
CLIFF4 = {"SOFT": 5, "MEDIUM": 15, "HARD": 25}
def f4(c, age, temp, base):
    ts = 1.0 + 0.025 * (temp - 28)
    return base + RATE4[c] * ts * max(0.0, age - CLIFF4[c])

print(f"F4 (no compound base): {test(f4)}/100")

# Formula 5: Temperature affects compound base too
def f5(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    cb = CB[c] * ts  # temp scales compound base too!
    return base + cb + RATE[c] * ts * max(0.0, age - CLIFF[c])

print(f"F5 (temp scales compound base): {test(f5)}/100")

# Formula 6: Degradation from lap 1 but with different formula
# lap_time = base + cb + rate * age * temp_scale (linear from start, no cliff)
RATE6 = {"SOFT": 0.15, "MEDIUM": 0.05, "HARD": 0.02}
def f6(c, age, temp, base):
    ts = 1.0 + 0.025 * (temp - 28)
    return base + CB[c] + RATE6[c] * ts * age

print(f"F6 (linear from lap 1): {test(f6)}/100")

# Formula 7: Exponential degradation
def f7(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    excess = max(0, age - CLIFF[c])
    if excess > 0:
        return base + CB[c] + RATE[c] * ts * (math.exp(0.1 * excess) - 1)
    return base + CB[c]

print(f"F7 (exponential past cliff): {test(f7)}/100")

# Formula 8: Two-phase - constant degradation BEFORE cliff, accelerated AFTER
RATE8_BEFORE = {"SOFT": 0.05, "MEDIUM": 0.02, "HARD": 0.01}
def f8(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    if age <= CLIFF[c]:
        return base + CB[c] + RATE8_BEFORE[c] * ts * age
    else:
        before = RATE8_BEFORE[c] * ts * CLIFF[c]
        after = RATE[c] * ts * (age - CLIFF[c])
        return base + CB[c] + before + after

print(f"F8 (two-phase linear): {test(f8)}/100")

# Formula 9: Pit stop on the SAME lap affects that lap's time (pit before driving)
def simulate_pit_before(strategy, config, lap_time_fn):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        # Pit stop BEFORE driving the lap
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

        tire_age += 1
        total_time += lap_time_fn(compound, tire_age, track_temp, base)

    return total_time

def test_pit_before(lap_time_fn):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            strat = strategies[pos_key]
            grid = int(pos_key[3:])
            t = simulate_pit_before(strat, config, lap_time_fn)
            results.append((t, grid, strat["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

print(f"F9 (pit BEFORE lap, current formula): {test_pit_before(f1)}/100")

# Formula 10: What if "lap" in pit_stops means the NEW tires are used starting that lap?
def simulate_pit_newlap(strategy, config, lap_time_fn):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    # pit_stop lap = first lap on new tires
    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        # At the START of this lap, check if we pit
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = 0

        tire_age += 1
        total_time += lap_time_fn(compound, tire_age, track_temp, base)

    return total_time

def test_pit_newlap(lap_time_fn):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            strat = strategies[pos_key]
            grid = int(pos_key[3:])
            t = simulate_pit_newlap(strat, config, lap_time_fn)
            results.append((t, grid, strat["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

print(f"F10 (pit means new tires on that lap): {test_pit_newlap(f1)}/100")

# Formula 11: What if temp only affects degradation rate, not as a multiplier but as an addition?
def f11(c, age, temp, base):
    excess = max(0.0, age - CLIFF[c])
    deg = (RATE[c] + TC[c] * (temp - TREF)) * excess
    return base + CB[c] + deg

print(f"F11 (temp additive to rate): {test(f11)}/100")

# Formula 12: What if there's no temperature effect at all?
def f12(c, age, temp, base):
    return base + CB[c] + RATE[c] * max(0.0, age - CLIFF[c])

print(f"F12 (no temp effect): {test(f12)}/100")

# Formula 13: Fuel effect - car gets lighter each lap
# lap_time = base + cb + deg - fuel_factor * lap_number
def f13(c, age, temp, base):
    ts = 1.0 + TC[c] * (temp - TREF)
    return base + CB[c] + RATE[c] * ts * max(0.0, age - CLIFF[c])

# Can't do fuel in per-lap function easily, skip

# Formula 14: What if "age" starts at 1 (not 0) and first lap is age 1?
# This is already what we do... but what if first lap is age 0?
def simulate_age0(strategy, config, lap_time_fn):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]

    pit_laps = {}
    for ps in strategy["pit_stops"]:
        pit_laps[ps["lap"]] = ps["to_tire"]

    compound = strategy["starting_tire"]
    tire_age = -1  # so first lap is age 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age += 1
        total_time += lap_time_fn(compound, tire_age, track_temp, base)
        if lap in pit_laps:
            total_time += pit_time
            compound = pit_laps[lap]
            tire_age = -1

    return total_time

def test_age0(lap_time_fn):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            strat = strategies[pos_key]
            grid = int(pos_key[3:])
            t = simulate_age0(strat, config, lap_time_fn)
            results.append((t, grid, strat["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    return correct

print(f"F14 (age starts at 0): {test_age0(f1)}/100")
