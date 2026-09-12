"""汇总所有问题4官方S25/S21演练数据（不同批次分布比较）。"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]

def collect_dirs(dirs):
    rows=[]
    for label,d in dirs:
        p=ROOT/d
        if not p.exists(): continue
        for sr in sorted(p.glob('run_*/summary.json')):
            r=json.loads(sr.read_text(encoding='utf-8'))
            rows.append({'batch':label,'dir':str(sr.parent),
                         'cleared':int(r.get('cleared',0)),'sources':int(r.get('sources',r.get('cleared',0))),
                         'V':float(r.get('virtual_time_s',float('nan'))),'success':bool(r.get('success',False)),
                         'measures':int(r.get('measure_calls',0)),'distance':float(r.get('distance_m',0))})
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
            'std_run_V_per_source':float(per.std(ddof=1)) if len(per)>1 else 0.0,
            'mean_measures':float(np.mean([r['measures'] for r in rows])),
            'mean_distance_m':float(np.mean([r['distance'] for r in rows])),
            'zero_failure_95_upper_bound':float(1-0.05**(1/max(len(rows),1)))}

def main():
    ap=argparse.ArgumentParser(); args=ap.parse_args()
    s25=collect_dirs([('b1_3','results/extreme_plan/official_b1_s25_p4_3'),
                      ('b1_7','results/extreme_plan/official_b1_s25_p4_7'),
                      ('alt','results/extreme_plan/official_alt_p4_50/S25_run_*'),
                      ('seq','results/extreme_plan/official_seq_s25_p4_20')])
    s21=collect_dirs([('b1_3','results/extreme_plan/official_b1_s21_p4_3'),
                      ('b1_7','results/extreme_plan/official_b1_s21_p4_7'),
                      ('alt','results/extreme_plan/official_alt_p4_50/S21_run_*'),
                      ('seq','results/extreme_plan/official_seq_s21_p4_20')])
    # alt glob issue: collect_dirs expects directory parents; handle explicitly
    def collect_alt(coverage):
        rows=[]
        base=ROOT/'results'/'extreme_plan'/'official_alt_p4_50'
        for d in sorted(base.glob(f'{coverage}_run_*')):
            sp=d/'summary.json'
            if not sp.exists(): continue
            r=json.loads(sp.read_text(encoding='utf-8'))
            rows.append({'batch':'alt','dir':str(d),'cleared':int(r.get('cleared',0)),
                         'sources':int(r.get('sources',r.get('cleared',0))),'V':float(r.get('virtual_time_s',float('nan'))),
                         'success':bool(r.get('success',False)),'measures':int(r.get('measure_calls',0)),
                         'distance':float(r.get('distance_m',0))})
        return rows
    # rebuild including alt correctly
    s25=collect_dirs([('b1_3','results/extreme_plan/official_b1_s25_p4_3'),('b1_7','results/extreme_plan/official_b1_s25_p4_7'),
                      ('seq','results/extreme_plan/official_seq_s25_p4_20')])+collect_alt('S25')
    s21=collect_dirs([('b1_3','results/extreme_plan/official_b1_s21_p4_3'),('b1_7','results/extreme_plan/official_b1_s21_p4_7'),
                      ('seq','results/extreme_plan/official_seq_s21_p4_20')])+collect_alt('S21')
    a=summarise(s25); b=summarise(s21)
    va=np.array([r['V']/max(r['sources'],1) for r in s25]); vb=np.array([r['V']/max(r['sources'],1) for r in s21])
    cmp={}
    if len(va)>1 and len(vb)>1:
        cmp={'mean_diff_S21_minus_S25_s_per_source':float(vb.mean()-va.mean()),
             'pct_change':float((vb.mean()-va.mean())/va.mean()*100),
             'mannwhitney_p':float(stats.mannwhitneyu(va,vb,alternative='two-sided').pvalue),
             'welch_ttest_p':float(stats.ttest_ind(va,vb,equal_var=False).pvalue)}
        rng=np.random.default_rng(7); diffs=[]
        for _ in range(10000):
            diffs.append(float(rng.choice(vb,size=len(vb),replace=True).mean()-rng.choice(va,size=len(va),replace=True).mean()))
        cmp['bootstrap_ci95']=[float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
    out={'S25':a,'S21':b,'comparison':cmp,'batches':{'S25':[r['batch'] for r in s25],'S21':[r['batch'] for r in s21]},
         'note':'官方案例随机不可重放；合并不同批次的分布比较，不是同案例配对。'}
    txt=json.dumps(out,ensure_ascii=False,indent=2); print(txt)
    (ROOT/'results'/'extreme_plan'/'official_b1_p4_all_summary.json').write_text(txt,encoding='utf-8')

if __name__=='__main__': main()