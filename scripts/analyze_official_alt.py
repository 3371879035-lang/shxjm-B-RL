"""汇总官方S25/S21交替演练结果。"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import stats

def collect(root: Path, coverage: str):
    rows=[]
    for d in sorted(root.glob(f'{coverage}_run_*')):
        sp=d/'summary.json'
        if not sp.exists(): continue
        r=json.loads(sp.read_text(encoding='utf-8'))
        rows.append({'dir':d.name,'cleared':int(r.get('cleared',0)),'sources':int(r.get('sources',r.get('cleared',0))),
                     'V':float(r.get('virtual_time_s',float('nan'))),'success':bool(r.get('success',False)),
                     'measures':int(r.get('measure_calls',0)),'distance':float(r.get('distance_m',0))})
    return rows

def stats_for(rows):
    if not rows: return {}
    V=np.array([r['V'] for r in rows]); S=np.array([r['sources'] for r in rows])
    per=V/np.maximum(S,1)
    return {'runs':len(rows),'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in rows)),
            'full_clear_runs':int(sum(1 for r in rows if r['success'] and r['cleared']==r['sources'])),
            'total_sources':int(S.sum()),'total_V':float(V.sum()),
            'aggregate_V_per_source':float(V.sum()/max(S.sum(),1)),
            'mean_run_V_per_source':float(per.mean()),'median_run_V_per_source':float(np.median(per)),
            'std_run_V_per_source':float(per.std(ddof=1)) if len(per)>1 else 0.0,
            'min_run_V_per_source':float(per.min()),'max_run_V_per_source':float(per.max()),
            'mean_measures':float(np.mean([r['measures'] for r in rows])),
            'mean_distance_m':float(np.mean([r['distance'] for r in rows]))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('dir'); args=ap.parse_args()
    root=Path(args.dir)
    a=collect(root,'S25'); b=collect(root,'S21')
    sa=stats_for(a); sb=stats_for(b)
    # 独立分布检验（官方案例不可重放）
    va=np.array([r['V']/max(r['sources'],1) for r in a]); vb=np.array([r['V']/max(r['sources'],1) for r in b])
    cmp={}
    if len(va)>1 and len(vb)>1:
        cmp={'mean_diff_S21_minus_S25_s_per_source':float(vb.mean()-va.mean()),
             'pct_change':float((vb.mean()-va.mean())/va.mean()*100) if va.mean()!=0 else float('nan'),
             'mannwhitney_p':float(stats.mannwhitneyu(va,vb,alternative='two-sided').pvalue)}
        rng=np.random.default_rng(7); diffs=[]
        for _ in range(5000):
            ia=rng.integers(0,len(va),len(va)); ib=rng.integers(0,len(vb),len(vb))
            diffs.append(float(vb[ib].mean()-va[ia].mean()))
        cmp['bootstrap_ci95']=[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
        # zero-failure upper bounds
        n_a=len(a); n_b=len(b)
        cmp['zero_failure_95_upper_bound']={'S25':float(1-0.05**(1/max(n_a,1))),'S21':float(1-0.05**(1/max(n_b,1)))}
    out={'dir':str(root),'S25':sa,'S21':sb,'comparison':cmp,
         'note':'官方案例随机不可重放，交替顺序只控制时间/批次效应；这是分布比较，不是同案例配对。'}
    txt=json.dumps(out,ensure_ascii=False,indent=2)
    print(txt)
    (root/'official_alt_summary.json').write_text(txt,encoding='utf-8')

if __name__=='__main__': main()