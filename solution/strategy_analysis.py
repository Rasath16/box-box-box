"""
Strategy analysis helpers for understanding F1 pit strategies.

This module provides utilities for:
- Parsing driver strategies from race input data
- Computing total race times using the tire degradation model
- Validating strategies against regulation requirements

Used during development to explore historical races and tune the model.
"""

from tire_model import (
    COMPOUND_BASE,
    CLIFF,
    calculate_lap_time,
    get_effective_degradation_rate,
)


def parse_pit_schedule(strategy):
    """
    Convert a strategy's pit stops into a lap-indexed lookup.

    Returns a dict mapping pit lap numbers to the new tire compound,
    e.g. {12: "HARD", 35: "SOFT"} for a two-stop strategy.
    """
    schedule = {}
    for stop in strategy.get("pit_stops", []):
        schedule[stop["lap"]] = stop["to_tire"]
    return schedule


def describe_strategy(strategy):
    """
    Create a human-readable description of a driver's strategy.
    e.g. "SOFT -> HARD@12" or "MEDIUM -> HARD@18 -> SOFT@35"
    """
    parts = [strategy["starting_tire"][0]]  # first letter
    for stop in sorted(strategy.get("pit_stops", []), key=lambda s: s["lap"]):
        parts.append(f"{stop['to_tire'][0]}@{stop['lap']}")
    return " -> ".join(parts)


def compute_race_time(strategy, race_config):
    """
    Simulate a full race for one driver and return total time.

    The simulation proceeds lap by lap:
    1. Increment tire age (starts at 0, so first lap is age 1)
    2. Calculate lap time based on current compound, age, and conditions
    3. If this lap has a pit stop, add pit lane penalty and switch tires

    Parameters:
        strategy: dict with 'starting_tire' and 'pit_stops'
        race_config: dict with 'base_lap_time', 'pit_lane_time',
                     'total_laps', 'track_temp'

    Returns:
        total race time in seconds (float)
    """
    base_lap_time = race_config["base_lap_time"]
    pit_lane_time = race_config["pit_lane_time"]
    total_laps = race_config["total_laps"]
    track_temp = race_config["track_temp"]

    pit_schedule = parse_pit_schedule(strategy)

    current_compound = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        # tires age before each lap (first lap on fresh tires = age 1)
        tire_age += 1

        # compute and accumulate this lap's time
        lap_time = calculate_lap_time(
            base_lap_time, current_compound, tire_age, track_temp
        )
        total_time += lap_time

        # pit stop at end of lap (if scheduled)
        if lap in pit_schedule:
            total_time += pit_lane_time
            current_compound = pit_schedule[lap]
            tire_age = 0  # fresh tires

    return total_time


def get_stint_breakdown(strategy, total_laps):
    """
    Break a strategy into individual stints for analysis.

    Returns a list of tuples: (compound, start_lap, end_lap, stint_length)
    """
    stops = sorted(strategy.get("pit_stops", []), key=lambda s: s["lap"])
    stints = []

    current = strategy["starting_tire"]
    start = 1

    for stop in stops:
        stints.append((current, start, stop["lap"], stop["lap"] - start + 1))
        current = stop["to_tire"]
        start = stop["lap"] + 1

    stints.append((current, start, total_laps, total_laps - start + 1))
    return stints


def validate_strategy(strategy, total_laps):
    """
    Check if a strategy is valid according to regulations.
    Must use at least 2 different compounds during the race.

    Returns (is_valid, reason)
    """
    compounds_used = {strategy["starting_tire"]}
    for stop in strategy.get("pit_stops", []):
        compounds_used.add(stop["to_tire"])

    if len(compounds_used) < 2:
        return False, "Must use at least 2 different tire compounds"

    # check pit laps are within race bounds
    for stop in strategy.get("pit_stops", []):
        if stop["lap"] < 1 or stop["lap"] >= total_laps:
            return False, f"Pit lap {stop['lap']} out of range [1, {total_laps-1}]"

    return True, "OK"
