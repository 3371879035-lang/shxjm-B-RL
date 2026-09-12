"""G21A vs G25OR-S21 本地严格配对。"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.g25o import G25OPolicy
from brl.g21a import G21APolicy
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources

def run_method(mode,seed,kind,method):
    src=generate_sources(seed,mode,kind); n=len(src)
    env=RadioEnv(mode=mode,n_sources=n,seed=seed,step_limit=20000); env.reset(seed=seed,n_sources=n,sources=src)
    t=time.time()
    if method=='G25OR-S21': out=G25OPolicy('G25OR',coverage='S21').run(env)
    elif method=='G21A': out=G21APolicy(mode=mode).run(env)
    else: raise ValueError(method)
    return {'kind':kind,'mode':mode,'seed':seed,'method':method,'sources':n,
            'success':bool(out['success']),'cleared':int(out['cleared']),
            'virtual_time_s':float(out['virtual_time_s']),'distance_m':float(out['distance_m']),
            'measure_calls':int(out['measure_calls']),'switches':int(out['switches']),
            'failed_clear':int(out['failed_clear']),'solver_failures':int(out['solver_failures']),
            'optical_fallbacks':int(out['optical_fallbacks']),
            'opportunistic_measure_count':int(out.get('opportunistic_measure_count',0)),
            'opportunistic_direction_count':int(out.get('opportunistic_direction_count',0)),
            'opportunistic_no_signal_count':int(out.get('opportunistic_no_signal_count',0)),
            'legs_selected':int(out.get('legs_selected',0)),'clear_ready_count':int(out.get('clear_ready_count',0)),
            'wall_s':time.time()-t}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=20); ap.add_argument('--start',type=int,default=97000)
    ap.add_argument('--out',default=str(ROOT/'results'/'extreme_plan'/'g21a_smoke.csv'))
    args=ap.parse_args(); rows=[]
    for kind in ('uniform','edge'):
        for seed in range(args.start,args.start+args.n):
            for m in ('G25OR-S21','G21A'):
                rows.append(run_method(4,seed,kind,m))
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for kind in ('uniform','edge'):
        summary[kind]={}
        for m in ('G25OR-S21','G21A'):
            sub=[r for r in rows if r['kind']==kind and r['method']==m]
            summary[kind][m]={'runs':len(sub),'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in sub)),
                'mean_v_s':float(np.mean([r['virtual_time_s'] for r in sub])),
                'mean_distance_m':float(np.mean([r['distance_m'] for r in sub])),
                'mean_measures':float(np.mean([r['measure_calls'] for r in sub])),
                'mean_opp':float(np.mean([r['opportunistic_measure_count'] for r in sub])),
                'mean_legs':float(np.mean([r['legs_selected'] for r in sub])),
                'mean_clear_ready':float(np.mean([r['clear_ready_count'] for r in sub])),
                'solver_failures':int(sum(r['solver_failures'] for r in sub)),
                'fallbacks':int(sum(r['optical_fallbacks'] for r in sub))}
    out={'n':args.n,'start':args.start,'summary':summary}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    (ROOT/'results'/'extreme_plan'/'g21a_smoke_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()