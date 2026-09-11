"""离线 Oracle 差距评估：同一自建场景上运行 G25OR，并计算知情邻域上下界。
真实位置仅用于离线 oracle 统计，不进入策略决策。
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.g25o import G25OPolicy
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources
from scripts.oracle_bounds import oracle_bounds

def run_one(mode, seed, kind, coverage='S25', variant='G25OR'):
    src=generate_sources(seed,mode,kind)
    env=RadioEnv(mode=mode,n_sources=len(src),seed=seed,step_limit=20000)
    env.reset(seed=seed,n_sources=len(src),sources=src)
    pol=G25OPolicy(variant,coverage=coverage)
    out=pol.run(env)
    ob=oracle_bounds([s.position for s in src])
    actual=float(out['virtual_time_s']); n=len(src)
    gap_low=actual-ob['oracle_time_lower_bound_s']; gap_high=actual-ob['oracle_feasible_time_upper_bound_s']
    return {'mode':mode,'seed':seed,'kind':kind,'coverage':coverage,'sources':n,
            'actual_s':actual,'actual_s_per_source':actual/max(n,1),
            'oracle_lower_s':ob['oracle_time_lower_bound_s'],'oracle_upper_s':ob['oracle_feasible_time_upper_bound_s'],
            'oracle_lower_s_per_source':ob['oracle_lower_s_per_source'],'oracle_upper_s_per_source':ob['oracle_upper_s_per_source'],
            'gap_to_lower_s':gap_low,'gap_to_upper_s':gap_high,'cleared':out['cleared'],'success':out['success']}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=10); ap.add_argument('--start',type=int,default=94000)
    ap.add_argument('--coverage',default='S25'); ap.add_argument('--out',default=str(ROOT/'results'/'extreme_plan'/'oracle_gap.csv'))
    args=ap.parse_args(); rows=[]
    for kind in ('uniform','edge'):
        for mode in (3,4):
            for seed in range(args.start,args.start+args.n):
                rows.append(run_one(mode,seed,kind,args.coverage))
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for mode in (3,4):
        sub=[r for r in rows if r['mode']==mode]
        summary[f'problem{mode}']={'runs':len(sub),
            'actual_s_per_source':float(np.mean([r['actual_s_per_source'] for r in sub])),
            'oracle_lower_s_per_source':float(np.mean([r['oracle_lower_s_per_source'] for r in sub])),
            'oracle_upper_s_per_source':float(np.mean([r['oracle_upper_s_per_source'] for r in sub])),
            'gap_to_lower_s_per_source':float(np.mean([r['gap_to_lower_s']/r['sources'] for r in sub])),
            'gap_to_upper_s_per_source':float(np.mean([r['gap_to_upper_s']/r['sources'] for r in sub])),
            'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in sub))}
    meta={'n':args.n,'start':args.start,'coverage':args.coverage,'summary':summary}
    (ROOT/'results'/'extreme_plan'/'oracle_gap_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__': main()