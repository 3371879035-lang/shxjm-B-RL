from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path(r'D:\数学建模\B_RL\results\figures')
TGT=Path(r'D:\数学建模\数学建模B2-RL\论文\figures')
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(8.2,5.2)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def box(x,y,w,h,text,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.03',fc=fc,ec='black',lw=1.0))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.5)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,lw=1.2,color='black'))
box(0.04,0.78,0.24,0.14,'初始化：位置、频道\n覆盖点、账本','#dbeafe')
box(0.36,0.78,0.26,0.14,'生成合法动作\nSCAN/REFINE/RESOLVE/EXIT','#dcfce7')
box(0.70,0.78,0.26,0.14,'A0确定性调度\n（PPO为可选消融）','#fef9c3')
box(0.70,0.48,0.26,0.14,'执行宏动作\n更新观测与虚拟时间','#e0e7ff')
box(0.36,0.48,0.26,0.14,'是否有清除证书？','#fae8ff')
box(0.04,0.48,0.24,0.14,'双侧定位或光学保底','#fee2e2')
box(0.36,0.18,0.26,0.14,'完成证书成立？','#e5e7eb')
box(0.70,0.18,0.26,0.14,'调用 /exit 退出','#e5e7eb')
arrow(0.28,0.85,0.36,0.85); arrow(0.62,0.85,0.70,0.85)
arrow(0.83,0.78,0.83,0.62); arrow(0.70,0.55,0.62,0.55); arrow(0.36,0.55,0.28,0.55)
arrow(0.62,0.48,0.83,0.48); arrow(0.83,0.48,0.83,0.32); arrow(0.62,0.25,0.70,0.25)
ax.text(0.18,0.38,'否',fontsize=10); ax.text(0.24,0.58,'是',fontsize=10); ax.text(0.84,0.70,'否',fontsize=10); ax.text(0.88,0.28,'是',fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'paper_algorithm_flow.png',dpi=300); 
fig.savefig(TGT/'paper_algorithm_flow.png',dpi=300); plt.close(fig)
print('flow updated')