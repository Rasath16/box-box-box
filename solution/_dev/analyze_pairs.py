"""
Extract constraints from historical races by analyzing pairs of drivers
with the same compound strategy but different pit laps.
"""

import json
import os
import sys
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")


def get_strategy_type(strategy):
    """Get a string describing the compound sequence."""
    tires = [strategy['starting_tire']]
    for ps in strategy.get('pit_stops', []):
        tires.append(ps['to_tire'])
    return '->'.join(tires)


def load_races(max_files=5):
    races = []
    files = sorted(os.listdir(DATA_DIR))[:max_files]
    for fname in files:
        with open(os.path.join(DATA_DIR, fname), 'r') as f:
            races.extend(json.load(f))
    return races


def main():
    print("Loading races...")
    races = load_races(max_files=5)
    print(f"Loaded {len(races)} races")

    # For each race, group drivers by strategy type
    # Find races where many drivers share the same strategy type
    # but have different pit laps
    # This tells us: which pit lap is optimal

    # Collect: for MEDIUM->HARD with specific N, T, the ordering of pit laps
    mh_data = []  # (N, T, base, pit, [(pit_lap, finishing_pos)])

    for race in races:
        cfg = race['race_config']
        N = cfg['total_laps']
        T = cfg['track_temp']
        base = cfg['base_lap_time']
        pit = cfg['pit_lane_time']
        true_order = race['finishing_positions']

        # Map driver_id -> finishing position (0-indexed)
        finish_pos = {d: i for i, d in enumerate(true_order)}

        # Group by strategy type
        groups = defaultdict(list)
        strats = race['strategies']
        for pos_key in strats:
            s = strats[pos_key]
            stype = get_strategy_type(s)
            driver_id = s['driver_id']
            grid = int(pos_key[3:])
            pit_laps = [p['lap'] for p in s.get('pit_stops', [])]
            groups[stype].append({
                'driver_id': driver_id,
                'grid': grid,
                'finish': finish_pos[driver_id],
                'pit_laps': pit_laps,
                'strategy': s
            })

        # For MEDIUM->HARD groups with multiple drivers
        for stype, drivers in groups.items():
            if stype == 'MEDIUM->HARD' and len(drivers) >= 5:
                pit_finish = []
                for d in drivers:
                    if len(d['pit_laps']) == 1:
                        pit_finish.append((d['pit_laps'][0], d['finish'], d['grid']))
                if len(pit_finish) >= 5:
                    pit_finish.sort(key=lambda x: x[1])  # sort by finish pos
                    mh_data.append((N, T, base, pit, pit_finish))

    # Analyze: for each race, which pit lap wins?
    print(f"\nFound {len(mh_data)} races with 5+ MEDIUM->HARD drivers")
    print("\nSample results (sorted by finish position):")
    print("N    T    base   pit    Best_pit  Worst_pit  Pit_laps_by_finish")

    # Group by (N, T) to find patterns
    by_NT = defaultdict(list)
    for N, T, base, pit, pf in mh_data:
        by_NT[(N, T)].append((base, pit, pf))
        best_pit = pf[0][0]  # pit lap of P1
        worst_pit = pf[-1][0]  # pit lap of last
        pit_order = [p[0] for p in pf[:6]]
        if len(mh_data) <= 50 or N in [40, 45, 50, 55, 60]:
            pass  # will print below

    # Print grouped by N
    by_N = defaultdict(list)
    for N, T, base, pit, pf in mh_data:
        best_pit = pf[0][0]
        by_N[N].append((T, best_pit, pf))

    print("\n=== Optimal pit lap by race length (MEDIUM->HARD) ===")
    for N in sorted(by_N.keys()):
        entries = by_N[N]
        pit_laps = [e[1] for e in entries]
        temps = [e[0] for e in entries]
        avg_pit = sum(pit_laps) / len(pit_laps)
        print(f"N={N:3d}: avg_best_pit={avg_pit:.1f} (n={len(entries)}) "
              f"T_range=[{min(temps)},{max(temps)}] "
              f"pits={sorted(set(pit_laps))}")

    # Also check SOFT->HARD and SOFT->MEDIUM
    print("\n=== Checking other strategy types ===")
    for target_type in ['SOFT->HARD', 'SOFT->MEDIUM', 'HARD->SOFT', 'MEDIUM->SOFT']:
        sh_data = []
        for race in races:
            cfg = race['race_config']
            true_order = race['finishing_positions']
            finish_pos = {d: i for i, d in enumerate(true_order)}
            groups = defaultdict(list)
            strats = race['strategies']
            for pos_key in strats:
                s = strats[pos_key]
                stype = get_strategy_type(s)
                driver_id = s['driver_id']
                pit_laps = [p['lap'] for p in s.get('pit_stops', [])]
                groups[stype].append({
                    'driver_id': driver_id,
                    'pit_laps': pit_laps,
                    'finish': finish_pos[driver_id]
                })
            for stype, drivers in groups.items():
                if stype == target_type and len(drivers) >= 3:
                    pit_finish = [(d['pit_laps'][0], d['finish']) for d in drivers if len(d['pit_laps']) == 1]
                    if len(pit_finish) >= 3:
                        pit_finish.sort(key=lambda x: x[1])
                        sh_data.append((cfg['total_laps'], cfg['track_temp'], pit_finish))

        if sh_data:
            by_N2 = defaultdict(list)
            for N, T, pf in sh_data:
                by_N2[N].append((T, pf[0][0], pf))
            print(f"\n{target_type} ({len(sh_data)} races):")
            for N in sorted(by_N2.keys())[:10]:
                entries = by_N2[N]
                pits = [e[1] for e in entries]
                temps = [e[0] for e in entries]
                print(f"  N={N:3d}: avg_best_pit={sum(pits)/len(pits):.1f} "
                      f"T=[{min(temps)},{max(temps)}] n={len(entries)}")

    # Deep dive: for N=40, T=28, show exact ordering
    print("\n=== Detailed: MEDIUM->HARD ordering patterns ===")
    count = 0
    for N, T, base, pit, pf in mh_data:
        if count >= 10:
            break
        if len(pf) >= 8:
            print(f"\nN={N} T={T} base={base} pit={pit}")
            print(f"  Finish order (pit_lap, finish_pos, grid): {pf}")
            count += 1


if __name__ == '__main__':
    main()
