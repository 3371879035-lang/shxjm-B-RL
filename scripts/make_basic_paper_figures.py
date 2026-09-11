from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(r'D:\数学建模\B_RL')
sys.path.insert(0,str(ROOT))
from brl.coverage import s25_points
from brl.g25o import route_open
from brl.geometry import wedge_halfplanes, intersect_halfplanes, minimum_enclosing_circle
OUT=ROOT/'results'/'figures'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']; plt.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size':11,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
pts=s25_points(); order=route_open(pts,[0,0]); route=np.vstack([[0,0],pts[order]])
fig,ax=plt.subplots(figsize=(6.2,6.2))
th=np.linspace(0,2*np.pi,500)
ax.plot(1800*np.cos(th),1800*np.sin(th),'k--',lw=1.0,label='目标区域边界')
ax.plot(np.r_[pts[13:,0],pts[13,0]],np.r_[pts[13:,1],pts[13,1]],'b-',lw=1.0,alpha=0.55,label='外环')
ax.plot(np.r_[pts[1:13,0],pts[1,0]],np.r_[pts[1:13,1],pts[1,1]],'g-',lw=1.0,alpha=0.55,label='内环')
ax.plot(route[:,0],route[:,1],'r.-',lw=1.2,ms=6,label='开路访问路线')
ax.scatter(pts[0,0],pts[0,1],c='k',marker='x',s=55,label='原点')
ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.set_xlabel('x / m'); ax.set_ylabel('y / m'); ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(OUT/'paper_s25_coverage.png',dpi=300); plt.close(fig)
s=np.array([300.0,-200.0]); true=np.array([1100.0,500.0])
ang=np.degrees(np.arctan2(true[1]-s[1],true[0]-s[0]))
p=intersect_halfplanes(wedge_halfplanes(s,ang)); c=minimum_enclosing_circle(p)
fig,ax=plt.subplots(figsize=(6.2,5.4))
th=np.linspace(0,2*np.pi,500)
ax.plot(1800*np.cos(th),1800*np.sin(th),'k--',lw=0.9,label='目标区域边界')
if len(p): ax.fill(p[:,0],p[:,1],color='tab:orange',alpha=0.35,label='定位区域 P')
ax.scatter([s[0]],[s[1]],c='tab:blue',s=45,label='检测点')
ax.scatter([true[0]],[true[1]],c='tab:red',s=45,label='真实位置')
ax.add_patch(Circle(c.center,c.radius,fill=False,color='tab:green',lw=1.6,label='最小包围圆'))
ax.set_aspect('equal'); ax.grid(alpha=0.25); ax.set_xlabel('x / m'); ax.set_ylabel('y / m'); ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(OUT/'paper_geometry_certificate.png',dpi=300); plt.close(fig)
print('done')