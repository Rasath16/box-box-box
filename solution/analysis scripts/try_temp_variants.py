"""
Try different temperature application points.
Maybe temp scales base, or everything, not just degradation.
"""
import json

tests = []
for i in range(1, 101):
    with open(f"data/test_cases/inputs/test_{i:03d}.json") as f:
        inp = json.load(f)
    with open(f"data/test_cases/expected_outputs/test_{i:03d}.json") as f:
        exp = json.load(f)
    tests.append((inp, exp["finishing_positions"]))

# Best params
CB = {"SOFT": -0.9665103286569976, "MEDIUM": 0.0, "HARD": 0.755284994643082}
RATE = {"SOFT": 1.6213600572975244, "MEDIUM": 0.813268608577364, "HARD": 0.345981233247675}
CLIFF = {"SOFT": 10, "MEDIUM": 20, "HARD": 29}
TC = {"SOFT": 0.025806274187704845, "MEDIUM": 0.02777171692356944, "HARD": 0.02401965544225936}
TREF = 27.96640138772966

def test_model(sim_fn, name):
    correct = 0
    for inp, exp in tests:
        config = inp["race_config"]
        strategies = inp["strategies"]
        results = []
        for pk in sorted(strategies.keys(), key=lambda k: int(k[3:])):
            s = strategies[pk]
            grid = int(pk[3:])
            t = sim_fn(s, config)
            results.append((t, grid, s["driver_id"]))
        results.sort(key=lambda r: (r[0], r[1]))
        pred = [d for _, _, d in results]
        if pred == exp:
            correct += 1
    print(f"{name}: {correct}/100")
    return correct

# Current model (baseline)
def model_current(strat, config):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
    compound = strat["starting_tire"]
    tire_age = 0
    total = 0.0
    for lap in range(1, total_laps + 1):
        tire_age += 1
        ts = 1.0 + TC[compound] * (temp - TREF)
        deg = RATE[compound] * ts * max(0.0, tire_age - CLIFF[compound])
        total += base + CB[compound] + deg
        if lap in pit_laps:
            total += pit_time
            compound = pit_laps[lap]
            tire_age = 0
    return total

# Model A: temp scales base_lap_time
def model_temp_base(strat, config):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
    compound = strat["starting_tire"]
    tire_age = 0
    total = 0.0
    # Try various tc and tref for base scaling
    tc_base = 0.001  # small effect on base
    for lap in range(1, total_laps + 1):
        tire_age += 1
        base_adj = base * (1.0 + tc_base * (temp - 28))
        deg = RATE[compound] * max(0.0, tire_age - CLIFF[compound])
        total += base_adj + CB[compound] + deg
        if lap in pit_laps:
            total += pit_time
            compound = pit_laps[lap]
            tire_age = 0
    return total

# Model B: temp scales everything (base + cb + deg)
def model_temp_all(strat, config):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
    compound = strat["starting_tire"]
    tire_age = 0
    total = 0.0
    for lap in range(1, total_laps + 1):
        tire_age += 1
        raw_time = base + CB[compound] + RATE[compound] * max(0.0, tire_age - CLIFF[compound])
        ts = 1.0 + 0.001 * (temp - 28)
        total += raw_time * ts
        if lap in pit_laps:
            total += pit_time
            compound = pit_laps[lap]
            tire_age = 0
    return total

# Model C: temp scales compound_base and degradation but NOT base
def model_temp_strategy(strat, config):
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    temp = config["track_temp"]
    pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
    compound = strat["starting_tire"]
    tire_age = 0
    total = 0.0
    for lap in range(1, total_laps + 1):
        tire_age += 1
        ts = 1.0 + TC[compound] * (temp - TREF)
        # Both compound base AND degradation scaled by temp
        strategy_effect = (CB[compound] + RATE[compound] * max(0.0, tire_age - CLIFF[compound])) * ts
        total += base + strategy_effect
        if lap in pit_laps:
            total += pit_time
            compound = pit_laps[lap]
            tire_age = 0
    return total

test_model(model_current, "Current (temp on deg only)")
test_model(model_temp_strategy, "Temp on CB+deg")

# Model D: What if base_lap_time is NOT in the per-lap formula but as a fixed addition?
# total = total_laps * base + strategy_cost?
# This is mathematically equivalent... doesn't change ranking

# Model E: temp scales compound base separately from degradation
for tc_cb in [0.001, 0.005, 0.01, 0.02, 0.03]:
    def make_model(tc_cb_val):
        def model(strat, config):
            base = config["base_lap_time"]
            total_laps = config["total_laps"]
            pit_time = config["pit_lane_time"]
            temp = config["track_temp"]
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
            compound = strat["starting_tire"]
            tire_age = 0
            total = 0.0
            for lap in range(1, total_laps + 1):
                tire_age += 1
                ts_deg = 1.0 + TC[compound] * (temp - TREF)
                ts_cb = 1.0 + tc_cb_val * (temp - TREF)
                cb = CB[compound] * ts_cb
                deg = RATE[compound] * ts_deg * max(0.0, tire_age - CLIFF[compound])
                total += base + cb + deg
                if lap in pit_laps:
                    total += pit_time
                    compound = pit_laps[lap]
                    tire_age = 0
            return total
        return model
    test_model(make_model(tc_cb), f"Temp on CB (tc_cb={tc_cb}) + deg")

# Model F: What if CLIFF varies with temperature?
# cliff = base_cliff + cliff_tc * (temp - tref)
print("\n=== Temperature-dependent cliff ===")
for cliff_tc in [-0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5]:
    def make_cliff_model(ctc):
        def model(strat, config):
            base = config["base_lap_time"]
            total_laps = config["total_laps"]
            pit_time = config["pit_lane_time"]
            temp = config["track_temp"]
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
            compound = strat["starting_tire"]
            tire_age = 0
            total = 0.0
            for lap in range(1, total_laps + 1):
                tire_age += 1
                ts = 1.0 + TC[compound] * (temp - TREF)
                cliff_adj = CLIFF[compound] + ctc * (temp - TREF)
                deg = RATE[compound] * ts * max(0.0, tire_age - cliff_adj)
                total += base + CB[compound] + deg
                if lap in pit_laps:
                    total += pit_time
                    compound = pit_laps[lap]
                    tire_age = 0
            return total
        return model
    test_model(make_cliff_model(cliff_tc), f"Cliff tc={cliff_tc}")

# Model G: What if the formula uses fuel load?
# Each lap the car burns fuel and gets lighter
# lap_time = base + cb + deg - fuel_factor * (lap_number - 1)
# This doesn't affect ranking between drivers since all drive same number of laps!
# Unless... fuel weight affects tire degradation?

# Model H: What if there's a constant warm-up penalty for new tires?
print("\n=== Tire warm-up penalty ===")
for warmup in [0.1, 0.2, 0.5, 1.0]:
    def make_warmup_model(wu):
        def model(strat, config):
            base = config["base_lap_time"]
            total_laps = config["total_laps"]
            pit_time = config["pit_lane_time"]
            temp = config["track_temp"]
            pit_laps = {ps["lap"]: ps["to_tire"] for ps in strat["pit_stops"]}
            compound = strat["starting_tire"]
            tire_age = 0
            total = 0.0
            for lap in range(1, total_laps + 1):
                tire_age += 1
                ts = 1.0 + TC[compound] * (temp - TREF)
                deg = RATE[compound] * ts * max(0.0, tire_age - CLIFF[compound])
                warmup_pen = wu if tire_age == 1 else 0.0
                total += base + CB[compound] + deg + warmup_pen
                if lap in pit_laps:
                    total += pit_time
                    compound = pit_laps[lap]
                    tire_age = 0
            return total
        return model
    test_model(make_warmup_model(warmup), f"Warmup={warmup}")
