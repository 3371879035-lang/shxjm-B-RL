"""G21A 六类场景组、N=10..16 的严格配对矩阵实验。"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import stats
from brl.g21a import G21APolicy
from brl.g25o import G25OPolicy
from brl.local_env import RadioEnv, Source

FAMILIES=['uniform','edge','cluster','minradius','outward','mixed']

def make_sources(seed, n_sources, family):
    rng=np.random.default_rng(seed)
    channels=rng.choice(np.arange(1,21),int(n_sources),replace=False)
    center=rng.normal(0,420,2); center=center/max(np.linalg.norm(center),1e-9)*min(np.linalg.norm(center),1200)
    out=[]
    for ch in channels:
        if family=='cluster':
            pos=center+rng.normal(0,260,2)
            if np.linalg.norm(pos)>1800: pos=pos/np.linalg.norm(pos)*1795
        elif family in ('edge','outward'):
            r=1800*rng.uniform(0.92,1.0); th=rng.uniform(0,2*np.pi); pos=np.array([r*np.cos(th),r*np.sin(th)])
        else:
            r=1800*np.sqrt(rng.random()); th=rng.uniform(0,2*np.pi); pos=np.array([r*np.cos(th),r*np.sin(th)])
        if family in ('minradius','edge'):
            radius=1000.0
        elif family=='outward':
            radius=float(rng.uniform(1000,1150))
        else:
            radius=float(rng.uniform(1000,1500))
        kind='omni'; direction=0.0
        if family in ('outward',):
            kind='directional'; direction=float(np.degrees(np.arctan2(pos[1],pos[0]))+rng.normal(0,8))
        elif family in ('mixed',):
            if rng.random()<0.6:
                kind='directional'; direction=float(rng.uniform(0,360))
        out.append(Source(int(ch),pos,float(radius),kind,direction))
    return out

def run_one(method, mode, seed, sources, family, n_sources):
    env=RadioEnv(mode=mode,n_sources=n_sources,seed=seed,step_limit=20000)
    env.reset(seed=seed,n_sources=n_sources,sources=sources)
    if method=='G25OR-S21':
        out=G25OPolicy('G25OR',coverage='S21').run(env)
    else:
        out=G21APolicy(mode=mode).run(env)
    row={'family':family,'n_sources':n_sources,'seed':seed,'method':method,
         'success':bool(out['success']),'cleared':int(out['cleared']),
         'virtual_time_s':float(out['virtual_time_s']),'distance_m':float(out['distance_m']),
         'measure_calls':int(out['measure_calls']),'switches':int(out['switches']),
         'failed_clear':int(out['failed_clear']),'solver_failures':int(out['solver_failures']),
         'optical_fallbacks':int(out['optical_fallbacks'])}
    for k in ['opportunistic_measure_count','opportunistic_direction_count','opportunistic_no_signal_count',
              'legs_selected','became_clear_ready_count','zero_detour_clear_count','clear_ready_count',
              'coverage_move_m','opportunity_move_m','resolver_move_m','clear_move_m',
              'coverage_measure_count','resolver_measure_count','clear_calls']:
        row[k]=out.get(k,0)
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=15); ap.add_argument('--start',type=int,default=123000)
    ap.add_argument('--out',default=str(ROOT/'results'/'extreme_plan'/'g21a_matrix.csv'))
    args=ap.parse_args(); rows=[]; t0=time.time()
    seed=args.start
    for family in FAMILIES:
        for n in range(10,17):
            for r in range(args.reps):
                src=make_sources(seed,n,family)
                for method in ('G25OR-S21','G21A'):
                    rows.append(run_one(method,4,seed,src,family,n))
                seed+=1
        print('[matrix] family',family,'done',flush=True)
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for family in FAMILIES:
        summary[family]={}
        for method in ('G25OR-S21','G21A'):
            sub=[r for r in rows if r['family']==family and r['method']==method]
            per=np.array([r['virtual_time_s']/max(r['cleared'],1) for r in sub])
            summary[family][method]={'runs':len(sub),'all_clear':bool(all(r['success'] and r['cleared']==r['n_sources'] for r in sub)),
                'mean_v_per_source':float(per.mean()),'median_v_per_source':float(np.median(per)),
                'p95_v_per_source':float(np.percentile(per,95)),
                'mean_distance_m':float(np.mean([r['distance_m'] for r in sub])),
                'mean_measures':float(np.mean([r['measure_calls'] for r in sub])),
                'mean_resolver_move_m':float(np.mean([r['resolver_move_m'] for r in sub])) if method=='G21A' else None}
        a=np.array([r['virtual_time_s']/max(r['cleared'],1) for r in rows if r['family']==family and r['method']=='G25OR-S21'])
        b=np.array([r['virtual_time_s']/max(r['cleared'],1) for r in rows if r['family']==family and r['method']=='G21A'])
        summary[family]['comparison']={'mean_diff_s21g21a_minus_base':float(b.mean()-a.mean()),
            'pct_change':float((b.mean()-a.mean())/a.mean()*100),
            'median_pct_change':float((np.median(b)-np.median(a))/np.median(a)*100),
            'paired_ttest_p':float(stats.ttest_rel(a,b).pvalue),
            'wilcoxon_p':float(stats.wilcoxon(a,b).pvalue)}
    # overall
    a=np.array([r['virtual_time_s']/max(r['cleared'],1) for r in rows if r['method']=='G25OR-S21'])
    b=np.array([r['virtual_time_s']/max(r['cleared'],1) for r in rows if r['method']=='G21A'])
    overall={'runs_per_method':len(a),'all_clear_base':bool(all(r['success'] and r['cleared']==r['n_sources'] for r in rows if r['method']=='G25OR-S21')),
             'all_clear_g21a':bool(all(r['success'] and r['cleared']==r['n_sources'] for r in rows if r['method']=='G21A')),
             'base_mean_v_per_source':float(a.mean()),'g21a_mean_v_per_source':float(b.mean()),
             'overall_pct_change':float((b.mean()-a.mean())/a.mean()*100),
             'overall_paired_ttest_p':float(stats.ttest_rel(a,b).pvalue),
             'wall_s':time.time()-t0}
    out={'reps_per_family_n':args.reps,'scale':len(rows),'overall':overall,'families':summary}
    (ROOT/'results'/'extreme_plan'/'g21a_matrix_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)[:6000])
if __name__=='__main__': main()