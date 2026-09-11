from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(r"D:\数学建模\B_RL")
OUT=ROOT/"results"/"figures"
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"font.size":11,"axes.titlesize":13,"axes.labelsize":12,
                     "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10})

def variant_results():
    data=json.loads((ROOT/"results"/"three_g25_variants_summary.json").read_text(encoding="utf-8"))
    labels=["G25O","G25OR","G25O-R full"]
    keys=["G25O","G25OR","G25OR_Full"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for prob,ax in zip((3,4),axes):
        vals=[data["variants"][f"problem{prob}"][k]["mean_V_per_source"] for k in keys]
        bars=ax.bar(np.arange(3),vals,width=0.55,color=["tab:blue","tab:orange","tab:green"])
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
        ax.set_ylabel("平均虚拟时间 / (s·源$^{-1}$)")
        ax.set_title(f"问题{prob}官方100局平均时间")
        ax.grid(axis="y",alpha=0.25)
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+2,f"{v:.1f}",ha="center",fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/"paper_three_variants_time.png",dpi=300); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.4))
    for prob,ax in zip((3,4),axes):
        vals=[data["variants"][f"problem{prob}"][k]["mean_program_s"] for k in keys]
        bars=ax.bar(labels,vals,color=["tab:blue","tab:orange","tab:green"])
        ax.set_ylabel("平均程序运行时间 / s"); ax.set_title(f"问题{prob}程序运行时间")
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
        ax.set_xlabel("每局源数"); ax.set_ylabel("平均虚拟时间 / (s·源$^{-1}$)")
        ax.set_title(f"问题{prob}分源数敏感性"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"paper_source_count_sensitivity.png",dpi=300); plt.close(fig)

if __name__=="__main__":
    variant_results(); source_count_sensitivity(); print("ok")