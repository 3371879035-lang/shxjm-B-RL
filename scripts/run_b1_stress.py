"""B1压力测试：聚集、边缘、最小接收半径和朝外定向源。"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.coverage import s25_points, s21_points
from brl.g25o import G25OPolicy
from brl.local_env import RadioEnv, Source

def make_sources(seed, mode, scenario):
    rng=np.random.default_rng(seed)
    channels=rng.choice(np.arange(1,21),int(rng.integers(10,17)),replace=False)
    src=[]
    cluster_center=rng.normal(0,400,2); cluster_center=cluster_center/max(np.linalg.norm(cluster_center),1e-9)*min(np.linalg.norm(cluster_center),1200)
    for ch in channels:
        if scenario=='cluster':
            pos=cluster_center+rng.normal(0,260,2)
            if np.linalg.norm(pos)>1800: pos=pos/np.linalg.norm(pos)*1790
        elif scenario=='edge':
            r=1800*rng.uniform(0.92,1.0); th=rng.uniform(0,2*np.pi); pos=np.array([r*np.cos(th),r*np.sin(th)])
        else:
            r=1800*np.sqrt(rng.random()); th=rng.uniform(0,2*np.pi); pos=np.array([r*np.cos(th),r*np.sin(th)])
        radius=1000.0 if scenario in ('edge','minradius') else float(rng.uniform(1000,1500))
        kind='omni'; direction=0.0
        if mode==4:
            outward=float(np.degrees(np.arctan2(pos[1],pos[0])))
            if scenario in ('edge','outward'):
                kind='directional'; direction=outward+rng.normal(0,10)
            elif rng.random()<0.5:
                kind='directional'; direction=float(rng.uniform(0,360))
        src.append(Source(int(ch),pos,radius,kind,direction))
    return src

def run(mode,seed,scenario,coverage):
    sources=make_sources(seed,mode,scenario)
    env=RadioEnv(mode=mode,n_sources=len(sources),seed=seed,step_limit=20000)
    env.reset(seed=seed,n_sources=len(sources),sources=sources)
    out=G25OPolicy('G25OR',coverage=coverage).run(env)
    return {'scenario':scenario,'mode':mode,'seed':seed,'coverage':coverage,'sources':len(sources),
            'success':bool(out['success']),'cleared':int(out['cleared']),
            'virtual_time_s':float(out['virtual_time_s']),'distance_m':float(out['distance_m']),
            'solver_failures':int(out['solver_failures']),'optical_fallbacks':int(out['optical_fallbacks'])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=20); ap.add_argument('--start',type=int,default=98000)
    args=ap.parse_args(); rows=[]
    for scenario in ('cluster','edge','minradius','outward'):
        for mode in (3,4):
            for seed in range(args.start,args.start+args.n):
                for coverage in ('S25','S21'):
                    rows.append(run(mode,seed,scenario,coverage))
    p=ROOT/'results'/'extreme_plan'/'b1_stress.csv'; p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for scenario in ('cluster','edge','minradius','outward'):
        summary[scenario]={}
        for mode in (3,4):
            for coverage in ('S25','S21'):
                sub=[r for r in rows if r['scenario']==scenario and r['mode']==mode and r['coverage']==coverage]
                summary[scenario][f'p{mode}_{coverage}']={'runs':len(sub),
                    'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in sub)),
                    'mean_v_s':float(np.mean([r['virtual_time_s'] for r in sub])),
                    'mean_distance_m':float(np.mean([r['distance_m'] for r in sub]))}
    out={'n':args.n,'start':args.start,'summary':summary}
    (ROOT/'results'/'extreme_plan'/'b1_stress_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()