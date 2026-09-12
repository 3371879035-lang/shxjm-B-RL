"""三项因果拆分：复用Oracle Ladder的50个场景（Q3/Q4各50）。

A: 固定S3/S25覆盖顺序 + 普通measure，到首次发现全部真实源为止。
B: A状态 + 真值源位置 + 最优圆盘清扫（Oracle；使用真值直接标记空频道）。
C: 真值仅用于选择合法覆盖顺序/求解顺序；动作仍是measure/clear/双边求解。
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources
from brl.causal_policy import (run_discovery_only, run_discovery_optimal_cleanup,
                               run_route_oracle, run_route_oracle_best)

def load_ladder():
    p = ROOT / 'results' / 'extreme_plan' / 'oracle_ladder_50scenes.csv'
    rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    out = {}
    for r in rows:
        out[(int(r['mode']), int(r['seed']), r['method'])] = r
    return out

def make_env(mode, seed, sources):
    e = RadioEnv(mode=mode, n_sources=len(sources), seed=seed, step_limit=20000)
    e.reset(seed=seed, n_sources=len(sources), sources=sources)
    return e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--start', type=int, default=160000)
    args = ap.parse_args()
    ladder = load_ladder()
    rows = []
    for mode in (3, 4):
        for i in range(args.n):
            seed = args.start + i
            kind = 'edge' if i % 3 == 0 else 'uniform'
            src = generate_sources(seed, mode, kind)
            n = max(len(src), 1)
            a_env = make_env(mode, seed, src); a = run_discovery_only(a_env)
            b_env = make_env(mode, seed, src); b = run_discovery_optimal_cleanup(b_env)
            c_env = make_env(mode, seed, src); c = run_route_oracle_best(c_env)
            cn_env = make_env(mode, seed, src); cn = run_route_oracle(cn_env)
            base = ladder[(mode, seed, 'G25OR')]
            o3 = ladder[(mode, seed, 'O3')]
            o0 = ladder[(mode, seed, 'O0')]
            bv = float(base['virtual_time_s']); o3v = float(o3['virtual_time_s']); o0v = float(o0['virtual_time_s'])
            cv = float(c['virtual_time_s']); bvv = float(b['virtual_time_s']); av = float(a['virtual_time_s'])
            row = {
                'mode': mode, 'seed': seed, 'sources': len(src),
                'baseline_v': bv, 'baseline_per_source': bv / n,
                'O3_v': o3v, 'O3_per_source': o3v / n,
                'O0_v': o0v, 'O0_per_source': o0v / n,
                'A_v': av, 'A_per_source': av / n,
                'A_dist': float(a['discovery_distance_m']), 'A_dist_per_source': float(a['discovery_distance_m']) / n,
                'A_measures': int(a['discovery_measures']), 'A_measures_per_source': float(a['discovery_measures']) / n,
                'A_all_found': bool(a['all_found']),
                'B_v': bvv, 'B_per_source': bvv / n,
                'B_cleanup_v': float(b.get('cleanup_time_s', 0.0)), 'B_cleanup_per_source': float(b.get('cleanup_time_s', 0.0)) / n,
                'B_cleanup_dist': float(b.get('cleanup_distance_m', 0.0)), 'B_cleanup_dist_per_source': float(b.get('cleanup_distance_m', 0.0)) / n,
                'B_all_clear': bool(b.get('cleared', 0) == len(src)), 'B_success': bool(b.get('success', False)),
                'C_v': cv, 'C_per_source': cv / n,
                'C_naive_v': float(cn['virtual_time_s']), 'C_naive_per_source': float(cn['virtual_time_s']) / n,
                'C_dist': float(c.get('distance_m', 0.0)), 'C_measures': int(c.get('measure_calls', 0)),
                'C_candidate': c.get('candidate', ''), 'C_resolve_key': c.get('resolve_key', ''),
                'C_all_clear': bool(c.get('cleared', 0) == len(src)), 'C_success': bool(c.get('success', False)),
                'order_gain_per_source': (bv - cv) / n,
                'info_cost_per_source': (cv - bvv) / n,
                'integration_gap_per_source': (bvv - o0v) / n,
                'total_gap_per_source': (bv - o0v) / n,
            }
            assert abs(row['order_gain_per_source'] + row['info_cost_per_source'] +
                       row['integration_gap_per_source'] - row['total_gap_per_source']) < 1e-7
            rows.append(row)
        print('[causal] mode', mode, 'done', flush=True)
    out_csv = ROOT / 'results' / 'extreme_plan' / 'causal_decomposition_50scenes.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {}
    for mode in (3, 4):
        sub = [r for r in rows if r['mode'] == mode]
        def mean(k):
            return float(np.mean([r[k] for r in sub]))
        s = {
            'n_scenes': len(sub), 'sources_total': int(sum(r['sources'] for r in sub)),
            'all_A_found': bool(all(r['A_all_found'] for r in sub)),
            'all_B_clear': bool(all(r['B_all_clear'] for r in sub)),
            'all_B_success': bool(all(r['B_success'] for r in sub)),
            'all_C_clear': bool(all(r['C_all_clear'] for r in sub)),
            'all_C_success': bool(all(r['C_success'] for r in sub)),
            'baseline_s_per_source': mean('baseline_per_source'),
            'O3_s_per_source': mean('O3_per_source'),
            'O0_s_per_source': mean('O0_per_source'),
            'A_discovery_s_per_source': mean('A_per_source'),
            'A_discovery_distance_per_source': mean('A_dist_per_source'),
            'A_discovery_measures_per_source': mean('A_measures_per_source'),
            'B_total_s_per_source': mean('B_per_source'),
            'B_cleanup_s_per_source': mean('B_cleanup_per_source'),
            'B_cleanup_distance_per_source': mean('B_cleanup_dist_per_source'),
            'C_total_s_per_source': mean('C_per_source'),
            'C_naive_total_s_per_source': mean('C_naive_per_source'),
            'C_measures_per_source': mean('C_measures') / max(mean('sources'), 1) if False else float(np.mean([r['C_measures'] / max(r['sources'], 1) for r in sub])),
        }
        s['gaps'] = {
            'order_gain_G_minus_C': mean('order_gain_per_source'),
            'info_cost_C_minus_B': mean('info_cost_per_source'),
            'integration_gap_B_minus_O0': mean('integration_gap_per_source'),
            'total_gap_G_minus_O0': mean('total_gap_per_source'),
            'identity_check': (mean('order_gain_per_source') + mean('info_cost_per_source') +
                               mean('integration_gap_per_source') - mean('total_gap_per_source')),
            'baseline_minus_A': s['baseline_s_per_source'] - s['A_discovery_s_per_source'],
            'B_minus_A': s['B_total_s_per_source'] - s['A_discovery_s_per_source'],
            'G_minus_O3': s['baseline_s_per_source'] - s['O3_s_per_source'],
            'O3_minus_O0': s['O3_s_per_source'] - s['O0_s_per_source'],
        }
        choices = {}
        gains = []
        for r in sub:
            choices[r['C_candidate']] = choices.get(r['C_candidate'], 0) + 1
            gains.append(r['order_gain_per_source'])
        s['C_candidate_counts'] = choices
        s['C_scenes_with_positive_order_gain'] = int(sum(1 for g in gains if g > 1e-6))
        s['C_max_order_gain_s_per_source'] = float(max(gains)) if gains else 0.0
        summary[f'problem{mode}'] = s
    out = {'n': args.n, 'start': args.start, 'summary': summary}
    out_json = ROOT / 'results' / 'extreme_plan' / 'causal_decomposition_50scenes_summary.json'
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:8000])

if __name__ == '__main__':
    main()