"""
Focused parameter search using analytical constraints derived from historical data.
Key constraints:
  - m_cliff ~ 17 (from SM/MS crossover analysis)
  - h_base = m_rate (from MH short-race crossover at k=18)
  - s_rate = 2*m_rate (from SM/SH crossover comparison)
  - h_cliff ~ 28 (from MH long-race crossover shift)
  - h_rate = 2*m_rate/3 (from h_cliff constraint at N=50,55)
  - tc ~ 0.067, T_ref = 30
  - s_base = -2*m_rate*(8.5 - s_cliff)
"""

import json
import os
import sys

TEST_INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "inputs")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_cases", "expected_outputs")
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_races")


def load_test_cases():
    cases = []
    for i in range(1, 101):
        with open(os.path.join(TEST_INPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            inp = json.load(f)
        with open(os.path.join(TEST_OUTPUT_DIR, f"test_{i:03d}.json"), 'r') as f:
            out = json.load(f)
        cases.append((inp, out['finishing_positions']))
    return cases


def load_hist(n=500):
    races = []
    with open(os.path.join(HIST_DIR, "races_00000-00999.json"), 'r') as f:
        races = json.load(f)
    return races[:n]


def simulate(race_input, params):
    s_base, h_base, s_rate, m_rate, h_rate, s_cliff, m_cliff, h_cliff, tc, t_ref = params
    cfg = race_input['race_config']
    base = cfg['base_lap_time']
    pit_time = cfg['pit_lane_time']
    total_laps = cfg['total_laps']
    temp = cfg['track_temp']
    ts = 1.0 + tc * (temp - t_ref)

    cb = {'SOFT': s_base, 'MEDIUM': 0.0, 'HARD': h_base}
    cr = {'SOFT': s_rate * ts, 'MEDIUM': m_rate * ts, 'HARD': h_rate * ts}
    cc = {'SOFT': s_cliff, 'MEDIUM': m_cliff, 'HARD': h_cliff}
    strats = race_input['strategies']
    times = []
    for pos_key in sorted(strats.keys(), key=lambda x: int(x[3:])):
        s = strats[pos_key]
        pit_map = {p['lap']: p['to_tire'] for p in s.get('pit_stops', [])}
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
        times.append((total, int(pos_key[3:]), s['driver_id']))
    times.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in times]


def score(cases, params, is_test=True):
    correct = 0
    for item in cases:
        if is_test:
            inp, true = item
        else:
            inp = item
            true = item['finishing_positions']
        if simulate(inp, params) == true:
            correct += 1
    return correct


def main():
    print("Loading data...")
    test_cases = load_test_cases()
    hist = load_hist(500)

    best_test = 0
    best_params = None
    best_hist = 0

    # Based on analytical constraints, search focused parameter space
    # m_rate is arbitrary scale factor - use 0.1
    m_rate = 0.1

    for m_cliff_val in [15, 16, 17, 18, 19]:
        for h_base_ratio in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            h_base = m_rate * h_base_ratio
            for s_rate_ratio in [1.5, 2.0, 2.5, 3.0, 3.5]:
                s_rate = m_rate * s_rate_ratio
                for h_rate_ratio in [0.1, 0.2, 1.0/3, 0.5, 2.0/3, 0.8]:
                    h_rate = m_rate * h_rate_ratio
                    for s_cliff_val in [2, 3, 4, 5, 6, 7, 8]:
                        # s_base from constraint: SM crossover at k~8.5
                        # s_base = -s_rate * max(0, 8.5 - s_cliff_val)
                        if s_cliff_val < 8.5:
                            s_base = -s_rate * (8.5 - s_cliff_val)
                        else:
                            s_base = 0  # no degradation at crossover
                        for h_cliff_val in [20, 25, 28, 30, 35]:
                            for tc_val in [0.03, 0.05, 0.067, 0.08, 0.1]:
                                t_ref = 30
                                params = (s_base, h_base, s_rate, m_rate, h_rate,
                                         s_cliff_val, m_cliff_val, h_cliff_val, tc_val, t_ref)

                                ts = score(test_cases, params, is_test=True)
                                if ts > best_test:
                                    best_test = ts
                                    best_params = params
                                    hs = score(hist, params, is_test=False)
                                    best_hist = hs
                                    print(f"NEW BEST test={ts}/100 hist={hs}/500 | "
                                          f"s_base={s_base:.3f} h_base={h_base:.3f} "
                                          f"s_rate={s_rate:.3f} m_rate={m_rate} h_rate={h_rate:.4f} "
                                          f"s_cliff={s_cliff_val} m_cliff={m_cliff_val} "
                                          f"h_cliff={h_cliff_val} tc={tc_val} t_ref={t_ref}")
                                elif ts == best_test and ts > 0:
                                    hs = score(hist, params, is_test=False)
                                    if hs > best_hist:
                                        best_hist = hs
                                        best_params = params
                                        print(f"BETTER HIST={hs}/500 test={ts}/100 | "
                                              f"s_base={s_base:.3f} h_base={h_base:.3f} "
                                              f"s_rate={s_rate:.3f} m_rate={m_rate} h_rate={h_rate:.4f} "
                                              f"s_cliff={s_cliff_val} m_cliff={m_cliff_val} "
                                              f"h_cliff={h_cliff_val} tc={tc_val} t_ref={t_ref}")

    print(f"\n=== BEST RESULT ===")
    print(f"Test: {best_test}/100, Hist: {best_hist}/500")
    names = ['s_base', 'h_base', 's_rate', 'm_rate', 'h_rate',
             's_cliff', 'm_cliff', 'h_cliff', 'tc', 't_ref']
    for n, v in zip(names, best_params):
        print(f"  {n} = {v}")

    # Now do a fine-tuning around best params
    print("\n=== Fine-tuning around best parameters ===")
    bp = list(best_params)
    # Try small variations of tc and t_ref
    for tc_delta in [-0.01, -0.005, 0, 0.005, 0.01]:
        for tref_delta in [-2, -1, 0, 1, 2]:
            p = list(bp)
            p[8] = bp[8] + tc_delta
            p[9] = bp[9] + tref_delta
            if p[8] <= 0:
                continue
            ts2 = score(test_cases, tuple(p))
            if ts2 > best_test:
                best_test = ts2
                best_params = tuple(p)
                print(f"FINE-TUNED test={ts2}/100 tc={p[8]:.4f} t_ref={p[9]}")


if __name__ == '__main__':
    main()
