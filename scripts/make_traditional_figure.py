from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
results=Path(r'D:\数学建模\B_RL\results')
target=Path(r'D:\数学建模\数学建模B2-RL\论文\figures')
three=json.loads((results/'three_g25_variants_summary.json').read_text(encoding='utf-8'))
trad=json.loads((results/'traditional_vs_rl_comparison.json').read_text(encoding='utf-8'))
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
for prob,ax in zip((3,4),axes):
    od=trad['official_distribution_comparison'][f'problem{prob}']
    v=three['variants'][f'problem{prob}']
    labels=['传统方法','旧安全PPO','G25O','G25OR']
    means=[od['traditional']['mean_V_per_source'],od['rl']['mean_V_per_source'],
           v['G25O']['mean_V_per_source'],v['G25OR']['mean_V_per_source']]
    bars=ax.bar(labels,means,color=['tab:gray','tab:blue','tab:orange','tab:green'])
    ax.set_ylabel('平均虚拟时间 / (s·源$^{-1}$)')
    ax.set_title(f'问题{prob}方法与方案B对比')
    ax.grid(axis='y',alpha=0.25)
    for b,m in zip(bars,means): ax.text(b.get_x()+b.get_width()/2,m+6,f'{m:.1f}',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(target/'traditional_vs_planb_v_per_source.png',dpi=300)
print('saved',(target/'traditional_vs_planb_v_per_source.png').stat().st_size)