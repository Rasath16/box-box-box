"""
F1 Race Simulator — Box Box Box Challenge

Predicts finishing positions for an F1 race based on each driver's
tire strategy (starting compound, pit stop timing, compound changes).

The core insight is that each car races independently (no interactions),
so finishing order is purely determined by total race time. Total time
depends on:
    1. Base lap time (track characteristic, same for everyone)
    2. Tire compound effect (SOFT is fastest, HARD is slowest)
    3. Tire degradation (increases after a compound-specific "cliff" age)
    4. Temperature scaling (hotter conditions amplify degradation)
    5. Pit stop penalties (fixed time cost per stop)

Usage:
    Reads race JSON from stdin, writes prediction JSON to stdout.
    python solution/race_simulator.py < input.json

Author's notes:
    Parameters were reverse-engineered from 30,000 historical races
    using a combination of data analysis and numerical optimisation.
    The key patterns found:
    - SOFT cliff ~10 laps, MEDIUM ~20, HARD ~29
    - Degradation is linear past the cliff
    - Temperature effect scales degradation rate per compound
"""

import json
import sys

# import the tire model and strategy helpers
from strategy_analysis import compute_race_time


def predict_finishing_order(race_data):
    """
    Predict the finishing order for a race.

    For each driver, simulate their race lap-by-lap to get their total time.
    Then sort by total time (ascending). If two drivers have the exact same
    total time, the one with the lower grid position (closer to front) wins.

    Parameters:
        race_data: dict containing 'race_config' and 'strategies'

    Returns:
        list of driver IDs in finishing order (1st to 20th)
    """
    config = race_data["race_config"]
    strategies = race_data["strategies"]

    results = []

    for position_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strategy = strategies[position_key]
        driver_id = strategy["driver_id"]
        grid_pos = int(position_key[3:])

        total_time = compute_race_time(strategy, config)
        results.append((total_time, grid_pos, driver_id))

    # primary sort by time, secondary by grid position for tiebreaking
    results.sort(key=lambda r: (r[0], r[1]))

    return [driver_id for _, _, driver_id in results]


def main():
    """Entry point — read race data from stdin, output prediction to stdout."""
    race_data = json.loads(sys.stdin.read())

    finishing_order = predict_finishing_order(race_data)

    output = {
        "race_id": race_data["race_id"],
        "finishing_positions": finishing_order,
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
