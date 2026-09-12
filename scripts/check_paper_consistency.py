"""检查论文关键数字与数据文件是否一致。"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT=Path(r'D:\数学建模\数学建模B2-RL')
PAPER=ROOT/'论文'/'B题_方案B_几何保证主动搜索_论文稿.md'
DATA=Path(r'D:\数学建模\B_RL\results\extreme_plan')

def load(name): return json.loads((DATA/name).read_text(encoding='utf-8'))

def main():
    text=PAPER.read_text(encoding='utf-8')
    checks=[]
    q=load('q1q2_audit.json')
    checks.append(('q1_mec', abs(q['q1']['mec_radius_m']-11.547005)<1e-3, '11.547'))
    checks.append(('q2_worst', any(abs(w-817.749)<0.01 for w in q['q2']['worst_each_candidate_m']), '817.749'))
    s21=load('s21_certificate.json')['info']
    checks.append(('s21_cells', s21['verified_cells']==3832, '3832'))
    checks.append(('s21_max_dist', abs(s21['max_support_corner_dist']-998.7888916)<0.01, '998.7889'))
    integ=load('s21_integration_check.json')
    checks.append(('route25', abs(integ['route_s25_m']-18173.8513)<0.01, '18173.851'))
    checks.append(('route21', abs(integ['route_s21_m']-17908.9301)<0.01, '17908.930'))
    rep=load('b1_replication_summary.json')
    checks.append(('b1_rep_uniform_pct', abs(rep['uniform']['pct']-(-3.4871))<0.05, '3.49'))
    checks.append(('b1_rep_edge_pct', abs(rep['edge']['pct']-(-4.6634))<0.05, '4.66'))
    stress=load('b1_stress_paired_summary.json')
    for k in ('cluster','edge','minradius','outward'):
        checks.append((f'stress_{k}', abs(stress[k]['pct']-(-2.77))<1.8 or abs(stress[k]['pct']-(-4.40))<0.3 or abs(stress[k]['pct']-(-3.17))<0.3 or abs(stress[k]['pct']-(-4.26))<0.3, 'pressure'))
    # 论文文本存在性
    for val in ['11.547','817.749','998.7889','18173.851','17908.930','3.49%','4.66%','2.77%','4.40%','3.17%','4.26%']:
        checks.append((f'text_{val}', val in text, val))
    bad=[c for c in checks if not c[1]]
    print(json.dumps({'checks':len(checks),'failed':bad},ensure_ascii=False,indent=2))
    if bad: raise SystemExit(1)
if __name__=='__main__': main()