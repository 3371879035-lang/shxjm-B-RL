"""生成G25O覆盖布局与配对比较图。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from brl.coverage import s25_points
from brl.g25o import route_open

FIG=ROOT/"results"/"figures"; FIG.mkdir(parents=True,exist_ok=True)
pts=s25_points()
order=route_open(pts,[0,0])
route=np.vstack([[0,0],pts[order]])

fig,ax=plt.subplots(figsize=(6,6))
th=np.linspace(0,2*np.pi,400)
ax.plot(1800*np.cos(th),1800*np.sin(th),'k--',lw=1,label='target boundary')
ax.plot(np.r_[pts[13:,0],pts[13,0]],np.r_[pts[13:,1],pts[13,1]],'b-',lw=0.8,alpha=0.5)
ax.plot(np.r_[pts[1:13,0],pts[1,0]],np.r_[pts[1:13,1],pts[1,1]],'g-',lw=0.8,alpha=0.5)
ax.plot(route[:,0],route[:,1],'r.-',lw=1,ms=6,label='open route')
ax.scatter(pts[0,0],pts[0,1],c='k',marker='x',s=50,label='origin')
ax.set_aspect('equal'); ax.grid(alpha=0.25)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title('S25 coverage points and open route')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG/'g25o_coverage.png',dpi=180); plt.close(fig)

summary=json.loads((ROOT/"results"/"g25o_paired_summary.json").read_text(encoding='utf-8'))
labels=[]; red=[]; a=[]; b=[]
for key in ['uniform_mode3','edge_mode3','uniform_mode4','edge_mode4']:
    d=summary[key]; labels.append(key.replace('_','\n')); red.append(d['reduction_pct']); a.append(d['A0_mean']); b.append(d['G25_mean'])
x=np.arange(len(labels))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].bar(x-0.2,a,width=0.4,label='A0'); axes[0].bar(x+0.2,b,width=0.4,label='G25O')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylabel('mean virtual time (s)'); axes[0].legend(); axes[0].grid(alpha=0.25)
axes[1].bar(x,red,color='tab:green'); axes[1].set_xticks(x); axes[1].set_xticklabels(labels); axes[1].set_ylabel('reduction vs A0 (%)'); axes[1].grid(alpha=0.25)
for i,v in enumerate(red): axes[1].text(i,v+0.5,f'{v:.1f}%',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'g25o_comparison.png',dpi=180); plt.close(fig)
print('figures generated')
