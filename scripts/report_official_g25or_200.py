"""汇总G25OR官方100+100局，并与Traditional/OldPlanB/G25O比较。"""
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

GROUPS={
 "Traditional":{3:{1,3,5,6,7,11,12,13,26,27,28,29,30,31},4:{2,4,8,9,10,15,16,17,18,19,20,21,22,23,24,25}},
 "OldPlanB":{3:set(range(36,56)),4:set(range(57,77))},
 "G25O":{3:set(range(89,189)),4:set(range(189,289))},
 "G25OR":{3:set(range(289,389)),4:set(range(389,489))},
}
ORDER=["Traditional","OldPlanB","G25O","G25OR"]

def load():
 con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True); con.row_factory=sqlite3.Row; cur=con.cursor()
 cur.execute("select * from practice_statistics_tasks where team_no='202610094088' and entered=1 and program_run_duration_ms is not null order by id")
 rows=[dict(r) for r in cur.fetchall()]; con.close(); return rows

def summ(rs):
 if not rs: return {}
 v=np.array([r['virtual_time_us']/1e6 for r in rs]); src=np.array([max(r['jammer_count'],1) for r in rs]); clr=np.array([r['cleared_jammer_count'] for r in rs]); prog=np.array([r['program_run_duration_ms']/1000 for r in rs])
 return {'runs':len(rs),'sources':int(src.sum()),'cleared':int(clr.sum()),'full_clear':int((clr==src).sum()),'success_rate':float((clr==src).mean()),'mean_v_s':float(v.mean()),'median_v_s':float(np.median(v)),'mean_v_per_source':float((v/src).mean()),'median_v_per_source':float(np.median(v/src)),'mean_program_s':float(prog.mean()),'median_program_s':float(np.median(prog)),'mean_measures':float(np.mean([r['measure_accepted_count'] for r in rs])),'mean_switches':float(np.mean([r['channel_switch_count'] for r in rs])),'mean_clear_failures':float(np.mean([r['clear_failure_count'] for r in rs]))}

def main():
 rows=load(); summary={}
 for p in (3,4):
  summary[f'problem{p}']={}
  for g in ORDER:
   summary[f'problem{p}'][g]=summ([r for r in rows if r['problem_no']==p and r['id'] in GROUPS[g][p]])
  a=np.array([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in rows if r['problem_no']==p and r['id'] in GROUPS['G25O'][p]])
  b=np.array([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in rows if r['problem_no']==p and r['id'] in GROUPS['G25OR'][p]])
  summary[f'problem{p}']['G25OR_vs_G25O']={'mean_diff':float(b.mean()-a.mean()),'pct':float((b.mean()-a.mean())/a.mean()*100),'mannwhitney_p':float(stats.mannwhitneyu(a,b,alternative='two-sided').pvalue)}
 (OUT/'official_g25or_200_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 # CSV
 fields=['group','problem','run_id','case_code','sources','cleared','full_clear','virtual_time_s','avg_clear_time_s','program_run_s','measure_calls','switches','clear_failures']
 with open(DOCS/'official_g25or_200_runs.csv','w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for p in (3,4):
   for g in ORDER:
    for r in [x for x in rows if x['problem_no']==p and x['id'] in GROUPS[g][p]]:
     src=max(r['jammer_count'],1)
     w.writerow({'group':g,'problem':p,'run_id':r['id'],'case_code':r['case_code'],'sources':r['jammer_count'],'cleared':r['cleared_jammer_count'],'full_clear':int(r['cleared_jammer_count']==r['jammer_count']),'virtual_time_s':round(r['virtual_time_us']/1e6,6),'avg_clear_time_s':round((r['virtual_time_us']/1e6)/src,6),'program_run_s':round(r['program_run_duration_ms']/1000,6),'measure_calls':r['measure_accepted_count'],'switches':r['channel_switch_count'],'clear_failures':r['clear_failure_count']})
 # figures
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,p in zip(axes,(3,4)):
  data=[]; labels=[]
  for g in ORDER:
   data.append([r['virtual_time_us']/1e6/max(r['jammer_count'],1) for r in rows if r['problem_no']==p and r['id'] in GROUPS[g][p]]); labels.append(g)
  ax.boxplot(data,tick_labels=labels,showmeans=True); ax.set_title(f"Problem {p}: official V/source"); ax.set_ylabel("s/source"); ax.grid(alpha=.25)
 fig.tight_layout(); fig.savefig(FIG/'official_g25or200_v_per_source.png',dpi=180); plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for ax,p in zip(axes,(3,4)):
  vals=[summary[f'problem{p}'][g].get('mean_program_s',0) for g in ORDER]
  ax.bar(ORDER,vals,color=['gray','orange','tab:blue','green']); ax.set_title(f"Problem {p}: mean program runtime"); ax.set_ylabel("s"); ax.grid(alpha=.25)
 fig.tight_layout(); fig.savefig(FIG/'official_g25or200_program_runtime.png',dpi=180); plt.close(fig)
 # markdown
 lines=['# G25OR官方100+100局与历史方案比较','','数据来自官方模拟器practice-statistics-queue.sqlite3。','']
 for p in (3,4):
  lines += [f'## 问题{p}','| 指标 | Traditional | OldPlanB | G25O | G25OR |','| --- | ---: | ---: | ---: | ---: |']
  keys=[('runs','局数'),('sources','源总数'),('cleared','清除数'),('success_rate','全清率'),('mean_v_per_source','平均虚拟时间/源(s)'),('median_v_per_source','中位虚拟时间/源(s)'),('mean_program_s','平均程序运行时间(s)'),('mean_measures','平均测量次数'),('mean_switches','平均切频次数'),('mean_clear_failures','平均失败清除')]
  for k,label in keys:
   vals=[]
   for g in ORDER:
    v=summary[f'problem{p}'][g].get(k,0); vals.append(f'{v:.3f}' if isinstance(v,float) else str(v))
   lines.append(f'| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |')
  d=summary[f'problem{p}']['G25OR_vs_G25O']; lines.append(f'| G25OR相对G25O | | | | {d["pct"]:+.2f}%, p={d["mannwhitney_p"]:.3f} |'); lines.append('')
 (OUT/'official_g25or_200_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
