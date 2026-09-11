"""生成论文用图表与结果表，并替换 paper.md 中的占位标记。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brl.coverage import s3_points, s4_points
from brl.geometry import minimum_enclosing_circle, wedge_halfplanes, intersect_halfplanes
from brl.resolver import optical_fallback_points, second_point_candidates

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PAPER = ROOT / "paper" / "paper.md"


def plot_coverage():
    p3 = s3_points(); p4 = s4_points()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    th = np.linspace(0, 2*np.pi, 400)
    for ax, pts, title in [(axes[0], p3, "S3: 7 points"),
                           (axes[1], p4, "S4: 31 triangular-lattice points")]:
        ax.plot(1800*np.cos(th), 1800*np.sin(th), "k-", lw=1, label="target boundary")
        ax.scatter(pts[:,0], pts[:,1], c="tab:red", s=28, label="measurement point")
        ax.scatter([0],[0], c="tab:blue", s=40, marker="x", label="origin")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "coverage_points.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_geometry():
    s = np.array([300.0, -200.0])
    true = np.array([1100.0, 500.0])
    ang = np.degrees(np.arctan2(true[1]-s[1], true[0]-s[0]))
    p = intersect_halfplanes(wedge_halfplanes(s, ang))
    c = minimum_enclosing_circle(p)
    fig, ax = plt.subplots(figsize=(6,5))
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(1800*np.cos(th), 1800*np.sin(th), "k--", lw=0.8)
    if len(p):
        ax.fill(p[:,0], p[:,1], color="tab:orange", alpha=0.35, label="intersection P")
    ax.scatter([s[0]],[s[1]], c="tab:blue", label="detector s1")
    ax.scatter([true[0]],[true[1]], c="tab:red", label="true source z")
    circle = plt.Circle(c.center, c.radius, fill=False, color="tab:green", lw=1.5, label="min enclosing circle")
    ax.add_patch(circle)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Wedge intersection and clear certificate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "geometry_certificate.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    return out


def load_train_logs():
    out = {}
    for mode in (3,4):
        files = sorted((ROOT/"results"/f"mode{mode}").glob("train_log_*.json"))
        if files:
            out[mode] = json.loads(files[-1].read_text(encoding="utf-8"))
    return out


def plot_training(logs):
    outs = []
    for mode, data in logs.items():
        h = data.get("history", [])
        if not h:
            continue
        steps = [x.get("global_step", i) for i,x in enumerate(h)]
        vt = [x.get("mean_episode_virtual_time", 0.0) for x in h]
        ret = [x.get("mean_episode_return", 0.0) for x in h]
        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(steps, vt, "b-", label="mean episode virtual time")
        ax1.set_xlabel("environment steps"); ax1.set_ylabel("virtual time (s)", color="b")
        ax2 = ax1.twinx()
        ax2.plot(steps, ret, "r--", label="mean episode return")
        ax2.set_ylabel("return", color="r")
        ax1.grid(alpha=0.25)
        fig.tight_layout()
        out = FIG / f"training_curve_mode{mode}.png"
        fig.savefig(out, dpi=180); plt.close(fig)
        outs.append((mode,out))
    return outs


def read_jsonl(path):
    rows=[]
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_rows(rows):
    if not rows: return None
    n=len(rows)
    succ=np.array([1.0 if r["success"] else 0.0 for r in rows])
    vt=np.array([r["virtual_time"] for r in rows], dtype=float)
    ratio=np.array([r["cleared"]/max(r["n_sources"],1) for r in rows])
    avg=np.array([r["avg_clear_time"] for r in rows], dtype=float)
    fallback=np.array([1.0 if r["fallback_actions"]>0 else 0.0 for r in rows])
    return {"n":n,"success_rate":float(succ.mean()),"clear_ratio":float(ratio.mean()),
            "mean_virtual_time":float(vt.mean()),"median_virtual_time":float(np.median(vt)),
            "mean_avg_clear_time":float(np.nanmean(avg)),"fallback_rate":float(fallback.mean())}


def paired_rows(path_a, path_b):
    a=read_jsonl(path_a); b=read_jsonl(path_b)
    n=min(len(a),len(b))
    if n==0: return None
    d=np.array([b[i]["virtual_time"]-a[i]["virtual_time"] for i in range(n)], dtype=float)
    return {"n":n,"mean_A":float(np.mean([x["virtual_time"] for x in a[:n]])),
            "mean_B":float(np.mean([x["virtual_time"] for x in b[:n]])),
            "mean_diff_B_minus_A":float(d.mean()),"median_diff":float(np.median(d)),
            "wins_B":int((d<0).sum()),"ties":int((np.abs(d)<1e-9).sum()),"wins_A":int((d>0).sum())}


def make_auto_md(logs):
    lines=[]
    lines.append("#### 9.3.1 训练曲线")
    for mode,out in plot_training(logs):
        lines.append(f"\n![问题{mode}训练曲线]({out.as_posix()})\n")
    if not logs:
        lines.append("\n（训练日志尚未生成；运行 scripts/train_ppo.py 后自动更新。）\n")
    lines.append("")
    lines.append("#### 9.4.1 本地配对结果")
    table_rows=[]
    for mode in (3,4):
        base=ROOT/"results"/f"eval_baseline_mode{mode}.jsonl"
        ppo=ROOT/"results"/f"eval_ppo_mode{mode}.jsonl"
        if not (base.exists() and ppo.exists()):
            continue
        sb=summarize_rows(read_jsonl(base)); sp=summarize_rows(read_jsonl(ppo)); pr=paired_rows(base,ppo)
        if sb: table_rows.append((mode,"A0",sb))
        if sp: table_rows.append((mode,"PPO+B",sp))
        if pr: table_rows.append((mode,"配对差",pr))
    if table_rows:
        lines.append("| 问题 | 算法 | 局数 | 完成率 | 清除比例 | 平均虚拟时间(s) | 中位虚拟时间(s) | 平均清除时间(s) | 保底触发率 | 配对差(B-A,s) | B胜/平/A胜 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode,label,s in table_rows:
            if label=="配对差":
                lines.append(f"| {mode} | {label} | {s['n']} | - | - | - | - | - | - | {s['mean_diff_B_minus_A']:.1f} | {s['wins_B']}/{s['ties']}/{s['wins_A']} |")
            else:
                lines.append(f"| {mode} | {label} | {s['n']} | {s['success_rate']:.3f} | {s['clear_ratio']:.3f} | {s['mean_virtual_time']:.1f} | {s['median_virtual_time']:.1f} | {s['mean_avg_clear_time']:.1f} | {s['fallback_rate']:.3f} | - | - |")
    else:
        lines.append("\n（配对评估结果尚未生成；训练完成后运行 scripts/evaluate_pair.py 并保存 JSONL。）\n")
    # paired scatter
    for mode in (3,4):
        base=ROOT/"results"/f"eval_baseline_mode{mode}.jsonl"
        ppo=ROOT/"results"/f"eval_ppo_mode{mode}.jsonl"
        if base.exists() and ppo.exists():
            a=read_jsonl(base); b=read_jsonl(ppo); n=min(len(a),len(b))
            if n:
                fig,ax=plt.subplots(figsize=(5,5))
                x=[a[i]["virtual_time"] for i in range(n)]; y=[b[i]["virtual_time"] for i in range(n)]
                m=max(max(x),max(y))*1.05
                ax.plot([0,m],[0,m],'k--',lw=0.8)
                ax.scatter(x,y,c="tab:blue",s=18)
                ax.set_xlabel("A0 virtual time (s)"); ax.set_ylabel("PPO+B virtual time (s)")
                ax.set_title(f"Paired local test, mode {mode}")
                ax.grid(alpha=0.25); fig.tight_layout()
                out=FIG/f"paired_mode{mode}.png"; fig.savefig(out,dpi=180); plt.close(fig)
                lines.append(f"\n![问题{mode}配对散点]({out.as_posix()})\n")
    return "\n".join(lines)


def main():
    plot_coverage(); plot_geometry()
    logs=load_train_logs()
    auto=make_auto_md(logs)
    if PAPER.exists():
        text=PAPER.read_text(encoding="utf-8")
        marker = "#### 9.4.1 本地配对结果"
        if marker in auto:
            train_part, pair_part = auto.split(marker, 1)
            pair_part = marker + pair_part
        else:
            train_part = auto
            pair_part = auto
        text=text.replace("（此处由 make_report_assets.py 自动插入训练曲线和表格。）", train_part)
        text=text.replace("（此处由 make_report_assets.py 自动插入A0与方案B的配对结果表和散点图。）", pair_part)
        PAPER.write_text(text,encoding="utf-8")
    (ROOT/"results"/"auto_results.md").write_text(auto,encoding="utf-8")
    print("assets generated")


if __name__ == "__main__":
    main()