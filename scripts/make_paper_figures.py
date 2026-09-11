"""鐢熸垚璁烘枃鐢ㄩ珮璐ㄩ噺鍥撅細妗嗘灦鍥俱€佺畻娉曞浘銆佽鐩栧浘銆佺粨鏋滃姣斿拰鏁忔劅鎬у浘銆?""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"figures"; OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"font.size":11,"axes.titlesize":13,"axes.labelsize":12,
                     "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10})

def box(ax,x,y,w,h,text,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc,ec="black",lw=1.0))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10.5)

def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.2,color="black"))

def framework():
    fig,ax=plt.subplots(figsize=(7.2,8.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ys=[0.90,0.78,0.66,0.54,0.42,0.30,0.18]
    box(ax,0.22,ys[0],0.56,0.08,"鏈哄櫒鐙?/enter 鍒濆鍖?, "#dbeafe")
    box(ax,0.22,ys[1],0.56,0.08,"S3 鎴?S25 淇濊瘉瑕嗙洊鎵弿", "#dcfce7")
    box(ax,0.22,ys[2],0.56,0.08,"direction / near / no_signal", "#fef9c3")
    box(ax,0.22,ys[3],0.56,0.08,"棰戦亾璐︽湰涓庝繚瀹堝彲琛屽煙鏇存柊", "#e0e7ff")
    box(ax,0.22,ys[4],0.56,0.08,"鍙屼晶鍖洪棿瀹氫綅 / 浜や細 / 鍏夊淇濆簳", "#fee2e2")
    box(ax,0.22,ys[5],0.56,0.08,"娓呴櫎璇佷功锛歁EC 鍗婂緞 鈮?19.5 m", "#fae8ff")
    box(ax,0.22,ys[6],0.56,0.08,"鍏ㄩ儴娓呴櫎鎴?16 棰戦亾鍚?/exit", "#e5e7eb")
    for i in range(len(ys)-1):
        arrow(ax,0.5,ys[i],0.5,ys[i+1]+0.08)
    ax.text(0.04,0.93,"闂3/4缁熶竴妗嗘灦",rotation=90,va="center",fontsize=12,fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"paper_framework.png",dpi=300); plt.close(fig)

def algorithm_flow():
    fig,ax=plt.subplots(figsize=(8.2,5.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    box(ax,0.04,0.78,0.24,0.14,"鍒濆鍖栵細浣嶇疆銆侀閬揬n瑕嗙洊鐐广€佽处鏈?,"#dbeafe")
    box(ax,0.36,0.78,0.26,0.14,"鐢熸垚鍚堟硶鍔ㄤ綔\nSCAN/REFINE/RESOLVE/EXIT","#dcfce7")
    box(ax,0.70,0.78,0.26,0.14,"瀹夊叏PPO鎴朅0閫夊姩浣?,"#fef9c3")
    box(ax,0.70,0.48,0.26,0.14,"鎵ц瀹忓姩浣淺n鏇存柊瑙傛祴涓庤櫄鎷熸椂闂?,"#e0e7ff")
    box(ax,0.36,0.48,0.26,0.14,"鏄惁鏈夋竻闄よ瘉涔︼紵","#fae8ff")
    box(ax,0.04,0.48,0.24,0.14,"鍙屼晶瀹氫綅鎴栧厜瀛︿繚搴?,"#fee2e2")
    box(ax,0.36,0.18,0.26,0.14,"瀹屾垚璇佷功鎴愮珛锛?,"#e5e7eb")
    box(ax,0.70,0.18,0.26,0.14,"璋冪敤 /exit 閫€鍑?,"#e5e7eb")
    arrow(ax,0.28,0.85,0.36,0.85); arrow(ax,0.62,0.85,0.70,0.85)
    arrow(ax,0.83,0.78,0.83,0.62); arrow(ax,0.70,0.55,0.62,0.55)
    arrow(ax,0.36,0.55,0.28,0.55)
    arrow(ax,0.62,0.48,0.83,0.48); arrow(ax,0.83,0.48,0.83,0.32)
    arrow(ax,0.62,0.25,0.70,0.25)
    ax.text(0.18,0.38,"鍚?,fontsize=10); ax.text(0.24,0.58,"鏄?,fontsize=10)
    ax.text(0.84,0.70,"鍚?,fontsize=10); ax.text(0.88,0.28,"鏄?,fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/"paper_algorithm_flow.png",dpi=300); plt.close(fig)

def coverage():
    import sys
    sys.path.insert(0,str(ROOT))
    from brl.coverage import s25_points
    from brl.g25o import route_open
    pts=s25_points(); order=route_open(pts,[0,0]); route=np.vstack([[0,0],pts[order]])
    fig,ax=plt.subplots(figsize=(6.2,6.2))
    th=np.linspace(0,2*np.pi,500)
    ax.plot(1800*np.cos(th),1800*np.sin(th),"k--",lw=1.0,label="鐩爣鍖哄煙杈圭晫")
    ax.plot(np.r_[pts[13:,0],pts[13,0]],np.r_[pts[13:,1],pts[13,1]],"b-",lw=1.0,alpha=0.55,label="澶栫幆")
    ax.plot(np.r_[pts[1:13,0],pts[1,0]],np.r_[pts[1:13,1],pts[1,1]],"g-",lw=1.0,alpha=0.55,label="鍐呯幆")
    ax.plot(route[:,0],route[:,1],"r.-",lw=1.2,ms=6,label="寮€璺闂矾绾?)
    ax.scatter(pts[0,0],pts[0,1],c="k",marker="x",s=55,label="鍘熺偣")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m"); ax.set_title("S25瑕嗙洊鐐逛笌寮€璺矾绾?)
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT/"paper_s25_coverage.png",dpi=300); plt.close(fig)

def geometry():
    sys.path.insert(0,str(ROOT)) if False else None
    from brl.geometry import wedge_halfplanes, intersect_halfplanes, minimum_enclosing_circle
    s=np.array([300.0,-200.0]); true=np.array([1100.0,500.0])
    ang=np.degrees(np.arctan2(true[1]-s[1],true[0]-s[0]))
    p=intersect_halfplanes(wedge_halfplanes(s,ang)); c=minimum_enclosing_circle(p)
    fig,ax=plt.subplots(figsize=(6.2,5.4))
    th=np.linspace(0,2*np.pi,500)
    ax.plot(1800*np.cos(th),1800*np.sin(th),"k--",lw=0.9,label="鐩爣鍖哄煙杈圭晫")
    if len(p): ax.fill(p[:,0],p[:,1],color="tab:orange",alpha=0.35,label="瀹氫綅鍖哄煙 P")
    ax.scatter([s[0]],[s[1]],c="tab:blue",s=45,label="妫€娴嬬偣")
    ax.scatter([true[0]],[true[1]],c="tab:red",s=45,label="鐪熷疄浣嶇疆")
    ax.add_patch(Circle(c.center,c.radius,fill=False,color="tab:green",lw=1.6,label="鏈€灏忓寘鍥村渾"))
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m"); ax.set_title("绀哄悜妤斿舰浜や細涓庢渶灏忓寘鍥村渾")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(OUT/"paper_geometry_certificate.png",dpi=300); plt.close(fig)

def variant_results():
    data=json.loads((ROOT/"results"/"three_g25_variants_summary.json").read_text(encoding="utf-8"))
    labels=["G25O","G25OR","G25O-R full"]
    keys=["G25O","G25OR","G25OR_Full"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    x=np.arange(3)
    for j,(prob,ax) in enumerate(zip((3,4),axes)):
        vals=[data["variants"][f"problem{prob}"][k]["mean_V_per_source"] for k in keys]
        bars=ax.bar(x,vals,width=0.55,color=["tab:blue","tab:orange","tab:green"])
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("骞冲潎铏氭嫙鏃堕棿 / (s路婧?^{-1}$)")
        ax.set_title(f"闂{prob}瀹樻柟100灞€骞冲潎鏃堕棿")
        ax.grid(axis="y",alpha=0.25)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+2,f"{v:.1f}",ha="center",fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"paper_three_variants_time.png",dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for prob,ax in zip((3,4),axes):
        vals=[data["variants"][f"problem{prob}"][k]["mean_program_s"] for k in keys]
        bars=ax.bar(labels,vals,color=["tab:blue","tab:orange","tab:green"])
        ax.set_ylabel("骞冲潎绋嬪簭杩愯鏃堕棿 / s"); ax.set_title(f"闂{prob}绋嬪簭杩愯鏃堕棿")
        ax.grid(axis="y",alpha=0.25)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.03,f"{v:.3f}",ha="center",fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"paper_three_variants_program.png",dpi=300); plt.close(fig)

def source_count_sensitivity():
    data=json.loads((ROOT/"results"/"three_g25_variants_summary.json").read_text(encoding="utf-8"))
    labels={"G25O":"G25O","G25OR":"G25OR","G25OR_Full":"G25O-R full"}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for prob,ax in zip((3,4),axes):
        xs=sorted(int(k) for k in data["by_source_count"][f"problem{prob}"].keys())
        for k in ("G25O","G25OR","G25OR_Full"):
            ys=[data["by_source_count"][f"problem{prob}"][str(n)][k]["mean_v_per_source"] for n in xs]
            ax.plot(xs,ys,marker="o",label=labels[k])
        ax.set_xlabel("姣忓眬婧愭暟"); ax.set_ylabel("骞冲潎铏氭嫙鏃堕棿 / (s路婧?^{-1}$)")
        ax.set_title(f"闂{prob}鍒嗘簮鏁版晱鎰熸€?); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"paper_source_count_sensitivity.png",dpi=300); plt.close(fig)

if __name__=="__main__":
    framework(); algorithm_flow(); coverage(); geometry(); variant_results(); source_count_sensitivity()
    print("paper figures generated")

