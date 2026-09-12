"""比较 G21A-v1、修复版、G25OR-S21 基线。"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(r'D:\数学建模\B_RL')

def load(path):
    rows=list(csv.DictReader(open(path,encoding='utf-8-sig')))
    out={}
    for r in rows:
        key=(r['family'],int(r['n_sources']),int(r['seed']),r['method'])
        out[key]=r
    return out

def main():
    v1=load(ROOT/'results'/'extreme_plan'/'g21a_matrix_v1.csv')
    fx=load(ROOT/'results'/'extreme_plan'/'g21a_matrix_fixed.csv')
    families=['uniform','edge','cluster','minradius','outward','mixed']
    summary={}
    for fam in families:
        base=[];v1g=[];fgg=[]
        for key,r in v1.items():
            if key[0]!=fam or key[3]!='G21A': continue
            key_base=(key[0],key[1],key[2],'G25OR-S21')
            key_fx=(key[0],key[1],key[2],'G21A')
            if key_base not in v1 or key_fx not in fx: continue
            base.append(float(v1[key_base]['virtual_time_s'])/max(int(v1[key_base]['cleared']),1))
            v1g.append(float(r['virtual_time_s'])/max(int(r['cleared']),1))
            fgg.append(float(fx[key_fx]['virtual_time_s'])/max(int(fx[key_fx]['cleared']),1))
        base=np.array(base);v1g=np.array(v1g);fgg=np.array(fgg)
        def pct(a,b): return float((b.mean()-a.mean())/a.mean()*100)
        summary[fam]={'n_pairs':len(base),'base_mean_v_per_source':float(base.mean()),'v1_mean':float(v1g.mean()),'fixed_mean':float(fgg.mean()),
                      'v1_vs_base_pct':pct(base,v1g),'fixed_vs_base_pct':pct(base,fgg),'fixed_vs_v1_pct':pct(v1g,fgg),
                      'fixed_vs_v1_p':float(stats.ttest_rel(v1g,fgg).pvalue),'fixed_vs_base_p':float(stats.ttest_rel(base,fgg).pvalue)}
    out={'summary':summary}
    # overall
    va=[];vv=[];vf=[]
    for key,r in v1.items():
        if key[3]!='G21A': continue
        kb=(key[0],key[1],key[2],'G25OR-S21'); kf=(key[0],key[1],key[2],'G21A')
        if kb not in v1 or kf not in fx: continue
        va.append(float(v1[kb]['virtual_time_s'])/max(int(v1[kb]['cleared']),1))
        vv.append(float(r['virtual_time_s'])/max(int(r['cleared']),1))
        vf.append(float(fx[kf]['virtual_time_s'])/max(int(fx[kf]['cleared']),1))
    va=np.array(va);vv=np.array(vv);vf=np.array(vf)
    out['overall']={'n_pairs':len(va),'base_mean':float(va.mean()),'v1_mean':float(vv.mean()),'fixed_mean':float(vf.mean()),
                    'v1_vs_base_pct':float((vv.mean()-va.mean())/va.mean()*100),
                    'fixed_vs_base_pct':float((vf.mean()-va.mean())/va.mean()*100),
                    'fixed_vs_v1_pct':float((vf.mean()-vv.mean())/vv.mean()*100),
                    'fixed_vs_v1_p':float(stats.ttest_rel(vv,vf).pvalue),
                    'fixed_vs_base_p':float(stats.ttest_rel(va,vf).pvalue)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    (ROOT/'results'/'extreme_plan'/'g21a_fixed_vs_v1_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()