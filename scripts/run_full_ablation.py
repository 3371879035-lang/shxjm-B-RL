"""Full滚动版消融：G25OR vs 修复后Full(无Q4动态证书) vs Full+Q4证书。"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.g25o import G25OPolicy
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources

METHODS=[('G25OR',dict(variant='G25OR',coverage='S25')),('Full_noQ4',dict(variant='G25ORFull',coverage='S25',enable_q4_cert=False)),('Full_Q4',dict(variant='G25ORFull',coverage='S25',enable_q4_cert=True))]

def run(mode,seed,kind,coverage='S25'):
    for label,kw in METHODS:
        src=generate_sources(seed,mode,kind)
        env=RadioEnv(mode=mode,n_sources=len(src),seed=seed,step_limit=20000)
        env.reset(seed=seed,n_sources=len(src),sources=src)
        pol=G25OPolicy(**kw); t=time.time(); out=pol.run(env)
        yield {'kind':kind,'mode':mode,'seed':seed,'method':label,'sources':len(src),
               'success':bool(out['success']),'cleared':int(out['cleared']),
               'virtual_time_s':float(out['virtual_time_s']),'distance_m':float(out['distance_m']),
               'measure_calls':int(out['measure_calls']),'switches':int(out['switches']),
               'failed_clear':int(out['failed_clear']),'optical_fallbacks':int(out['optical_fallbacks']),
               'solver_failures':int(out['solver_failures']),'extra_measurements':int(out['extra_measurements']),
               'dynamic_absent_channels':int(out.get('dynamic_absent_channels',0)),
               'dynamic_absent_cells':int(out.get('dynamic_absent_cells',0)),
               'wall_s':time.time()-t}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=15); ap.add_argument('--start',type=int,default=96000)
    args=ap.parse_args(); rows=[]
    for kind in ('uniform','edge'):
        for mode in (3,4):
            for seed in range(args.start,args.start+args.n):
                rows.extend(run(mode,seed,kind))
    p=ROOT/'results'/'extreme_plan'/'full_ablation.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for kind in ('uniform','edge'):
        for mode in (3,4):
            sub=[r for r in rows if r['kind']==kind and r['mode']==mode]
            summary[f'{kind}_p{mode}']={}
            for label,_ in METHODS:
                s=[r for r in sub if r['method']==label]
                summary[f'{kind}_p{mode}'][label]={'runs':len(s),'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in s)),
                    'mean_v_s':float(np.mean([r['virtual_time_s'] for r in s])),
                    'mean_distance_m':float(np.mean([r['distance_m'] for r in s])),
                    'dynamic_absent_channels':int(sum(r['dynamic_absent_channels'] for r in s)),
                    'dynamic_absent_cells':int(sum(r['dynamic_absent_cells'] for r in s))}
    outmeta={'n':args.n,'start':args.start,'summary':summary}
    (ROOT/'results'/'extreme_plan'/'full_ablation_summary.json').write_text(json.dumps(outmeta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(outmeta,ensure_ascii=False,indent=2))
if __name__=='__main__': main()