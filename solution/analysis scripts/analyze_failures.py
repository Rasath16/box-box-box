"""
Analyze failed test cases to find patterns and possible model improvements.
"""
import json
import os
import sys

TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")

# Current best params
PARAMS = {
    's_base': -0.9665103286569976,
    'h_base': 0.755284994643082,
    's_rate': 1.6213600572975244,
    'm_rate': 0.813268608577364,
    'h_rate': 0.3404666107633389,
    's_cliff': 10, 'm_cliff': 20, 'h_cliff': 29,
    'tc_s': 0.025806274187704845,
    'tc_m': 0.02777171692356944,
    'tc_h': 0.02401965544225936,
    't_ref': 27.569263880216884,
}


def simulate(race_input):
    p = PARAMS
    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']

    dt = temp - p['t_ref']
    cb = {'SOFT': p['s_base'], 'MEDIUM': 0.0, 'HARD': p['h_base']}
    cr = {
        'SOFT': p['s_rate'] * (1 + p['tc_s'] * dt),
        'MEDIUM': p['m_rate'] * (1 + p['tc_m'] * dt),
        'HARD': p['h_rate'] * (1 + p['tc_h'] * dt)
    }
    cc = {'SOFT': p['s_cliff'], 'MEDIUM': p['m_cliff'], 'HARD': p['h_cliff']}

    strats = race_input['strategies']
    results = []

    for pos_key in sorted(strats.keys(), key=lambda x: int(x[3:])):
        s = strats[pos_key]
        pit_map = {pp['lap']: pp['to_tire'] for pp in s.get('pit_stops', [])}
        cur = s['starting_tire']
        age = 0
        total = 0.0

        for lap in range(1, total_laps + 1):
            age += 1
            deg = cr[cur] * max(0.0, age - cc[cur])
            total += base + cb[cur] + deg
            if lap in pit_map:
                total += pit_time
                cur = pit_map[lap]
                age = 0

        strat_desc = s['starting_tire'][0]
        for stop in sorted(s.get('pit_stops', []), key=lambda x: x['lap']):
            strat_desc += f"->{stop['to_tire'][0]}@{stop['lap']}"

        results.append({
            'driver_id': s['driver_id'],
            'grid': int(pos_key[3:]),
            'time': total,
            'strategy': strat_desc,
        })

    results.sort(key=lambda x: (x['time'], x['grid']))
    return results


for i in range(1, 101):
    with open(os.path.join(TEST_INPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
        inp = json.load(f)
    with open(os.path.join(TEST_OUTPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
        out = json.load(f)

    true_order = out['finishing_positions']
    results = simulate(inp)
    pred_order = [r['driver_id'] for r in results]

    if pred_order != true_order:
        cfg = inp['race_config']
        diffs = sum(1 for a, b in zip(pred_order, true_order) if a != b)

        # Find the swapped pairs
        true_pos = {d: i for i, d in enumerate(true_order)}
        pred_pos = {d: i for i, d in enumerate(pred_order)}

        swaps = []
        for r in results:
            d = r['driver_id']
            if true_pos[d] != pred_pos[d]:
                swaps.append(f"  {d}: pred={pred_pos[d]+1} true={true_pos[d]+1} time={r['time']:.4f} strat={r['strategy']}")

        if diffs <= 4:  # Only show small failures
            print(f"\nTest {i}: N={cfg['total_laps']}, T={cfg['track_temp']:.1f}, "
                  f"base={cfg['base_lap_time']}, pit={cfg['pit_lane_time']}, diffs={diffs}")
            for s in swaps:
                print(s)
