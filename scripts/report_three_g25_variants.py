"""官方G25O / G25OR / G25O-R完整版 三方对比分析。"""
from __future__ import annotations
import csv, json, sqlite3, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
DB=Path(r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full\JammersSimulatorData\practice-statistics-queue.sqlite3")
OUT=ROOT/"results"; DOCS=ROOT/"docs"/"records"; FIG=OUT/"figures"
for d in (OUT,DOCS,FIG): d.mkdir(parents=True,exist_ok=True)

VARIANTS={
 "G25O":{3:set(range(89,189)),4:set(range(189,289))},
 "G25OR":{3:set(range(289,389)),4:set(range(389,489))},
 "G25OR_Full":{3:set(range(489,589)),4:set(range(589,689))},
}
LABELS={"G25O":"G25O","G25OR":"G25OR","G25OR_Full":"G25O-R full"}

def load():
 con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True); con.row_factory=sqlite3.Row; cur=con.cursor()
 cur.execute("select * from practice_statistics_tasks where team_no='202610094088' and entered=1 and program_run_duration_ms is not null order by id")
 rows=[dict(r) for r in cur.fetchall()]; con.close(); return rows

def rows_for(rows,prob,v):
 return [r for r in rows if r['problem_no']==prob and r['id'] in VARIANTS[v][prob]]

def summarize(rs):
 v=np.array([r['virtual_time_us']/1e6 for r in rs]); src=np.array([max(r['jammer_count'],1) for r in rs]); clr=np.array([r['cleared_jammer_count'] for r in rs]); prog=np.array([r['program_run_duration_ms']/1000 for r in rs]); meas=np.array([r['measure_accepted_count'] for r in rs]); sw=np.array([r['channel_switch_count'] for r in rs]); fail=np.array([r['clear_failure_count'] for r in rs])
 # 固定时间 = 5*measure + switch + 3*fail + 5*success_clear
 fixed=5*meas+sw+3*fail+5*clr
 move_time=v-fixed
 return {
  'runs':len(rs),'sources':int(src.sum()),'cleared':int(clr.sum()),'full_clear':int((clr==src).sum()),'success_rate':float((clr==src).mean()),
  'mean_sources':float(src.mean()),'mean_V':float(v.mean()),'median_V':float(np.median(v)),'mean_V_per_source':float((v/src).mean()),'median_V_per_source':float(np.median(v/src)),
  'mean_program_s':float(prog.mean()),'median_program_s':float(np.median(prog)),'p95_program_s':float(np.quantile(prog,0.95)),'max_program_s':float(prog.max()),
  'mean_measures':float(meas.mean()),'mean_switches':float(sw.mean()),'mean_clear_failures':float(fail.mean()),
  'mean_fixed_time_s':float(fixed.mean()),'mean_move_time_s':float(move_time.mean()),'move_share':float(move_time.mean()/v.mean()),
  'mean_move_distance_est_m':float(move_time.mean()*5),
 }

def main():
 rows=load(); out={'variants':{},'pairwise':{},'by_source_count':{}}
 for prob in (3,4):
  out['variants'][f'problem{prob}']={}
  for v in VARIANTS:
   out['variants'][f'problem{prob}'][v]=summarize(rows_for(rows,prob,v))
  out['pairwise'][f'problem{prob}']={}
  for a,b in [('G25O','G25OR'),('G25O','G25OR_Full'),('G25OR','G25OR_Full')]:
   ra=rows_for(rows,prob,a); rb=rows_for(rows,prob,b)
   va=np.array([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in ra]); vb=np.array([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in rb])
   out['pairwise'][f'problem{prob}'][f'{a}_vs_{b}']={'pct':float((vb.mean()-va.mean())/va.mean()*100),'mean_diff':float(vb.mean()-va.mean()),'mannwhitney_p':float(stats.mannwhitneyu(va,vb,alternative='two-sided').pvalue)}
  # by source count
  out['by_source_count'][f'problem{prob}']={}
  for n in range(10,17):
   out['by_source_count'][f'problem{prob}'][str(n)]={}
   for v in VARIANTS:
    rs=[r for r in rows_for(rows,prob,v) if r['jammer_count']==n]
    out['by_source_count'][f'problem{prob}'][str(n)][v]={'runs':len(rs),'mean_v_per_source':float(np.mean([r['virtual_time_us']/1e6/n for r in rs])) if rs else None}
 (OUT/'three_g25_variants_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 # CSV
 with open(DOCS/'three_g25_variants_runs.csv','w',newline='',encoding='utf-8-sig') as f:
  fields=['variant','problem','run_id','case_code','sources','cleared','full_clear','virtual_time_s','v_per_source_s','program_run_s','measure_calls','switches','clear_failures']
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for v in VARIANTS:
   for prob in (3,4):
    for r in rows_for(rows,prob,v):
     src=max(r['jammer_count'],1)
     w.writerow({'variant':LABELS[v],'problem':prob,'run_id':r['id'],'case_code':r['case_code'],'sources':r['jammer_count'],'cleared':r['cleared_jammer_count'],'full_clear':int(r['cleared_jammer_count']==r['jammer_count']),'virtual_time_s':round(r['virtual_time_us']/1e6,6),'v_per_source_s':round((r['virtual_time_us']/1e6)/src,6),'program_run_s':round(r['program_run_duration_ms']/1000,6),'measure_calls':r['measure_accepted_count'],'switches':r['channel_switch_count'],'clear_failures':r['clear_failure_count']})
 # figures
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,prob in zip(axes,(3,4)):
  data=[]; labels=[]
  for v in VARIANTS:
   rs=rows_for(rows,prob,v); data.append([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in rs]); labels.append(LABELS[v])
  ax.boxplot(data,tick_labels=labels,showmeans=True); ax.set_title(f"Problem {prob}: official V/source"); ax.set_ylabel("s/source"); ax.grid(alpha=.25)
 fig.tight_layout(); fig.savefig(FIG/'three_g25_variants_v_per_source.png',dpi=180); plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,prob in zip(axes,(3,4)):
  vals=[out['variants'][f'problem{prob}'][v]['mean_program_s'] for v in VARIANTS]
  ax.bar([LABELS[v] for v in VARIANTS],vals,color=['tab:blue','tab:orange','tab:green']); ax.set_title(f"Problem {prob}: mean program runtime"); ax.set_ylabel("s"); ax.grid(alpha=.25)
 fig.tight_layout(); fig.savefig(FIG/'three_g25_variants_program_runtime.png',dpi=180); plt.close(fig)
 # markdown
 lines=['# 三个G25O变体官方100+100局对比','','数据来自官方模拟器practice-statistics-queue.sqlite3。','']
 for prob in (3,4):
  lines.append(f'## 问题{prob}'); lines.append('| 指标 | G25O | G25OR | G25O-R full |'); lines.append('| --- | ---: | ---: | ---: |')
  keys=[('runs','局数'),('sources','源总数'),('cleared','清除数'),('success_rate','全清率'),('mean_V_per_source','平均虚拟时间/源(s)'),('median_V_per_source','中位虚拟时间/源(s)'),('mean_program_s','平均程序运行时间(s)'),('mean_measures','平均测量次数'),('mean_switches','平均切频次数'),('mean_clear_failures','平均失败清除'),('move_share','移动时间占比'),('mean_move_distance_est_m','反推平均移动距离(m)')]
  for k,label in keys:
   vals=[]
   for v in VARIANTS:
    x=out['variants'][f'problem{prob}'][v][k]; vals.append(f'{x:.4f}' if isinstance(x,float) else str(x))
   lines.append(f'| {label} | {vals[0]} | {vals[1]} | {vals[2]} |')
  for pair,d in out['pairwise'][f'problem{prob}'].items():
   lines.append(f'| {pair} | | | {d["pct"]:+.2f}%, p={d["mannwhitney_p"]:.3f} |')
  lines.append('')
 (OUT/'three_g25_variants_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
