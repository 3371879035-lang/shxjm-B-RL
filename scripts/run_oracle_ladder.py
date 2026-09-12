"""Oracle Ladder：Q3/Q4各若干场景严格配对运行 G25OR/O1/O2/O3/O0。"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from brl.local_env import RadioEnv, Source
from brl.g25o import G25OPolicy
from brl.oracle_policy import run_o1, run_o2, run_o3, run_o0
from scripts.run_g25o_compare import generate_sources

METHODS=['G25OR','O1','O2','O3','O0']

def run_method(method, mode, seed, sources):
    env=RadioEnv(mode=mode,n_sources=len(sources),seed=seed,step_limit=20000)
    env.reset(seed=seed,n_sources=len(sources),sources=sources)
    if method=='G25OR':
        out=G25OPolicy('G25OR',coverage='S25').run(env)
    elif method=='O1': out=run_o1(env)
    elif method=='O2': out=run_o2(env)
    elif method=='O3': out=run_o3(env)
    elif method=='O0': out=run_o0(env)
    else: raise ValueError(method)
    return {'mode':mode,'seed':seed,'method':method,'sources':len(sources),
            'success':bool(out['success']),'cleared':int(out['cleared']),
            'virtual_time_s':float(out['virtual_time_s']),'distance_m':float(out['distance_m']),
            'measure_calls':int(out['measure_calls']),'clear_calls':int(out['clear_calls']),
            'failed_clear':int(out['failed_clear'])}

def sanity_checks():
    # N=1 O0公式
    checks=[]
    for pos in ([100.,0.],[10.,0.],[0.,0.]):
        src=[Source(1,np.array(pos,float),1200.0,'omni',0.0)]
        env=RadioEnv(mode=4,n_sources=1,seed=1); env.reset(seed=1,n_sources=1,sources=src)
        out=run_o0(env); exp=max(0,float(np.linalg.norm(pos))-20)/5+5
        checks.append({'pos':pos,'o0_time':float(out['virtual_time_s']),'expected':float(exp),'ok':abs(out['virtual_time_s']-exp)<1e-6})
    # omni vs directional相同位置
    pos=[np.array([100.,0.]),np.array([300.,200.]),np.array([-500.,400.])]
    ch=[1,3,5]
    omni=[Source(c,p,1200.0,'omni',0.0) for c,p in zip(ch,pos)]
    dirs=[Source(c,p,1200.0,'directional',float((i*77)%360)) for i,(c,p) in enumerate(zip(ch,pos))]
    def o0(srcs):
        e=RadioEnv(mode=4,n_sources=len(srcs),seed=2); e.reset(seed=2,n_sources=len(srcs),sources=srcs); return run_o0(e)
    a=o0(omni); b=o0(dirs)
    checks.append({'omni_time':a['virtual_time_s'],'directional_time':b['virtual_time_s'],
                   'omni_dist':a['distance_m'],'directional_dist':b['distance_m'],
                   'ok':abs(a['virtual_time_s']-b['virtual_time_s'])<1e-6 and abs(a['distance_m']-b['distance_m'])<1e-6})
    return {'all_ok':all(c['ok'] for c in checks),'checks':checks}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=50); ap.add_argument('--start',type=int,default=160000)
    ap.add_argument('--out',default=str(ROOT/'results'/'extreme_plan'/'oracle_ladder.csv'))
    ap.add_argument('--summary-out',default=str(ROOT/'results'/'extreme_plan'/'oracle_ladder_summary.json'))
    args=ap.parse_args(); rows=[]; t0=time.time()
    for mode in (3,4):
        for i in range(args.n):
            seed=args.start+i
            kind='edge' if i%3==0 else 'uniform'
            src=generate_sources(seed,mode,kind)
            for method in METHODS:
                rows.append(run_method(method,mode,seed,src))
        print('[oracle] mode',mode,'done',flush=True)
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary={}
    for mode in (3,4):
        summary[f'problem{mode}']={}
        for method in METHODS:
            sub=[r for r in rows if r['mode']==mode and r['method']==method]
            per=np.array([r['virtual_time_s']/max(r['sources'],1) for r in sub])
            summary[f'problem{mode}'][method]={'runs':len(sub),'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in sub)),
                'mean_v_per_source':float(per.mean()),'median_v_per_source':float(np.median(per)),
                'mean_distance_per_source':float(np.mean([r['distance_m']/max(r['sources'],1) for r in sub])),
                'mean_measures_per_source':float(np.mean([r['measure_calls']/max(r['sources'],1) for r in sub])),
                'mean_clear_calls_per_source':float(np.mean([r['clear_calls']/max(r['sources'],1) for r in sub]))}
        g=summary[f'problem{mode}']['G25OR']
        o1=summary[f'problem{mode}']['O1']; o2=summary[f'problem{mode}']['O2']; o3=summary[f'problem{mode}']['O3']; o0=summary[f'problem{mode}']['O0']
        summary[f'problem{mode}']['gaps_s_per_source']={
            'channel_G_minus_O1':g['mean_v_per_source']-o1['mean_v_per_source'],
            'location_G_minus_O2':g['mean_v_per_source']-o2['mean_v_per_source'],
            'search_O3_minus_O0':o3['mean_v_per_source']-o0['mean_v_per_source'],
            'total_headroom_G_minus_O0':g['mean_v_per_source']-o0['mean_v_per_source']}
    out={'n':args.n,'start':args.start,'sanity':sanity_checks(),'summary':summary,'wall_s':time.time()-t0}
    Path(args.summary_out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)[:8000])
if __name__=='__main__': main()