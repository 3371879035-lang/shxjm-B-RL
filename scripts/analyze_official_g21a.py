"""汇总官方问题4 G21A vs G25OR-S21 各100局。"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(r'D:\数学建模\B_RL')

def collect(d):
    rows=[]
    for sr in sorted((ROOT/d).glob('run_*/summary.json')):
        r=json.loads(sr.read_text(encoding='utf-8'))
        rows.append({'cleared':int(r.get('cleared',0)),'sources':int(r.get('sources',r.get('cleared',0))),
                     'V':float(r.get('virtual_time_s',float('nan'))),'success':bool(r.get('success',False)),
                     'measures':int(r.get('measure_calls',0)),'distance':float(r.get('distance_m',0)),
                     'opp':int(r.get('opportunistic_measure_count',0)),'opp_dir':int(r.get('opportunistic_direction_count',0)),
                     'opp_ns':int(r.get('opportunistic_no_signal_count',0)),'legs':int(r.get('legs_selected',0)),
                     'became_cr':int(r.get('became_clear_ready_count',0))})
    return rows

def summarise(rows):
    if not rows: return {}
    V=np.array([r['V'] for r in rows]); S=np.array([r['sources'] for r in rows]); per=V/np.maximum(S,1)
    return {'runs':len(rows),'all_clear':bool(all(r['success'] and r['cleared']==r['sources'] for r in rows)),
            'full_clear_runs':int(sum(1 for r in rows if r['success'] and r['cleared']==r['sources'])),
            'total_sources':int(S.sum()),'total_V':float(V.sum()),
            'aggregate_V_per_source':float(V.sum()/max(S.sum(),1)),
            'mean_run_V_per_source':float(per.mean()),'median_run_V_per_source':float(np.median(per)),
            'p90_run_V_per_source':float(np.percentile(per,90)),'p95_run_V_per_source':float(np.percentile(per,95)),
            'std_run_V_per_source':float(per.std(ddof=1)),
            'mean_measures':float(np.mean([r['measures'] for r in rows])),
            'mean_distance_m':float(np.mean([r['distance'] for r in rows])),
            'zero_failure_95_upper_bound':float(1-0.05**(1/max(len(rows),1))),
            'mean_opp':float(np.mean([r['opp'] for r in rows])),
            'mean_opp_dir':float(np.mean([r['opp_dir'] for r in rows])),
            'mean_opp_ns':float(np.mean([r['opp_ns'] for r in rows])),
            'mean_legs':float(np.mean([r['legs'] for r in rows])),
            'mean_became_cr':float(np.mean([r['became_cr'] for r in rows]))}

def main():
    g=collect('results/extreme_plan/official_g21a_p4_100')
    b=collect('results/extreme_plan/official_s21_p4_100')
    sg=summarise(g); sb=summarise(b)
    va=np.array([r['V']/max(r['sources'],1) for r in b]); vg=np.array([r['V']/max(r['sources'],1) for r in g])
    cmp={'mean_diff_g21a_minus_s21_s_per_source':float(vg.mean()-va.mean()),
         'pct_change':float((vg.mean()-va.mean())/va.mean()*100) if va.mean()!=0 else float('nan'),
         'mannwhitney_p':float(stats.mannwhitneyu(va,vg,alternative='two-sided').pvalue),
         'welch_p':float(stats.ttest_ind(va,vg,equal_var=False).pvalue)}
    rng=np.random.default_rng(7); diffs=[]
    for _ in range(10000):
        ia=rng.integers(0,len(va),len(va)); ig=rng.integers(0,len(vg),len(vg))
        diffs.append(float(vg[ig].mean()-va[ia].mean()))
    cmp['bootstrap_ci95']=[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
    out={'G21A':sg,'S21':sb,'comparison':cmp,
         'note':'官方问题4各100局，随机不可重放；分布比较，不是同案例配对。'}
    txt=json.dumps(out,ensure_ascii=False,indent=2); print(txt)
    (ROOT/'results'/'extreme_plan'/'official_g21a_s21_p4_100_summary.json').write_text(txt,encoding='utf-8')
if __name__=='__main__': main()