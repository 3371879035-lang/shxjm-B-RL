"""汇总官方G25O 200局演练（100+100），生成CSV/Markdown/图表。"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = Path(r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full\JammersSimulatorData\practice-statistics-queue.sqlite3")
OUT = ROOT / "results"
DOCS = ROOT / "docs" / "records"
FIG = ROOT / "results" / "figures"
for d in (OUT, DOCS, FIG):
    d.mkdir(parents=True, exist_ok=True)

# 当前官方数据库中的G25O演练快照id范围
G25O_IDS = {3: range(89, 189), 4: range(189, 289)}
OLD_PLANB_IDS = {3: range(36, 56), 4: range(57, 77)}
# 传统方法的官方演练记录id（旧方案之前，人工发起）
TRADITIONAL_IDS = {
    3: {1, 3, 5, 6, 7, 11, 12, 13, 26, 27, 28, 29, 30, 31},
    4: {2, 4, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
}


def load_db():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from practice_statistics_tasks where team_no='202610094088' and entered=1 and program_run_duration_ms is not null order by id")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def group_rows(rows, problem, group):
    ids = G25O_IDS[problem]
    if group == "G25O":
        return [r for r in rows if r["problem_no"] == problem and r["id"] in ids]
    if group == "OldPlanB":
        return [r for r in rows if r["problem_no"] == problem and r["id"] in OLD_PLANB_IDS[problem]]
    if group == "Traditional":
        return [r for r in rows if r["problem_no"] == problem and r["id"] in TRADITIONAL_IDS[problem]]
    raise ValueError(group)


def summarize(rows):
    if not rows:
        return {}
    v = np.asarray([r["virtual_time_us"] / 1e6 for r in rows], dtype=float)
    src = np.asarray([max(r["jammer_count"], 1) for r in rows], dtype=float)
    clr = np.asarray([r["cleared_jammer_count"] for r in rows], dtype=float)
    prog = np.asarray([r["program_run_duration_ms"] / 1000.0 for r in rows], dtype=float)
    return {
        "runs": len(rows),
        "total_sources": int(src.sum()),
        "total_cleared": int(clr.sum()),
        "full_clear_runs": int((clr == src).sum()),
        "success_rate": float((clr == src).mean()),
        "mean_sources": float(src.mean()),
        "mean_virtual_time_s": float(v.mean()),
        "median_virtual_time_s": float(np.median(v)),
        "mean_v_per_source_s": float((v / src).mean()),
        "median_v_per_source_s": float(np.median(v / src)),
        "mean_program_s": float(prog.mean()),
        "median_program_s": float(np.median(prog)),
        "min_program_s": float(prog.min()),
        "max_program_s": float(prog.max()),
        "mean_measures": float(np.mean([r["measure_accepted_count"] for r in rows])),
        "mean_switches": float(np.mean([r["channel_switch_count"] for r in rows])),
        "mean_clear_failures": float(np.mean([r["clear_failure_count"] for r in rows])),
    }


def write_csv(all_groups):
    path = DOCS / "official_g25o_200_runs.csv"
    fields = ["group", "problem", "run_id", "case_code", "sources", "cleared", "full_clear",
              "virtual_time_s", "avg_clear_time_s", "program_run_s", "measure_calls", "switches",
              "clear_failures", "end_reason"]
    rows=[]
    for (problem, group), items in all_groups.items():
        for r in items:
            src=max(r["jammer_count"],1)
            rows.append({
                "group":group,"problem":problem,"run_id":r["id"],"case_code":r["case_code"],
                "sources":r["jammer_count"],"cleared":r["cleared_jammer_count"],
                "full_clear":int(r["cleared_jammer_count"]==r["jammer_count"]),
                "virtual_time_s":round(r["virtual_time_us"]/1e6,6),
                "avg_clear_time_s":round((r["virtual_time_us"]/1e6)/src,6),
                "program_run_s":round(r["program_run_duration_ms"]/1000.0,6),
                "measure_calls":r["measure_accepted_count"],"switches":r["channel_switch_count"],
                "clear_failures":r["clear_failure_count"],"end_reason":r["end_reason"],
            })
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def make_figures(all_groups):
    # 每个问题：G25O / OldPlanB / Traditional 的V/源箱线图
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,problem in zip(axes,(3,4)):
        data=[]; labels=[]
        for group in ("Traditional","OldPlanB","G25O"):
            items=all_groups.get((problem,group),[])
            if items:
                data.append([r["virtual_time_us"]/1e6/max(r["jammer_count"],1) for r in items])
                labels.append(group)
        ax.boxplot(data,tick_labels=labels,showmeans=True)
        ax.set_title(f"Problem {problem}: official virtual time per source")
        ax.set_ylabel("s/source"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG/"official_g25o200_v_per_source.png",dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,problem in zip(axes,(3,4)):
        vals=[]; labels=[]
        for group in ("Traditional","OldPlanB","G25O"):
            items=all_groups.get((problem,group),[])
            if items:
                vals.append(np.mean([r["program_run_duration_ms"]/1000 for r in items])); labels.append(group)
        ax.bar(labels,vals,color=['tab:gray','tab:orange','tab:green'])
        ax.set_title(f"Problem {problem}: mean program runtime")
        ax.set_ylabel("s"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG/"official_g25o200_program_runtime.png",dpi=180); plt.close(fig)


def main():
    rows=load_db()
    all_groups={(p,g):group_rows(rows,p,g) for p in (3,4) for g in ("Traditional","OldPlanB","G25O")}
    summary={f"problem{p}_{g}":summarize(all_groups[(p,g)]) for p in (3,4) for g in ("Traditional","OldPlanB","G25O")}
    (OUT/"official_g25o_200_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(all_groups)
    make_figures(all_groups)
    lines=["# 官方G25O 200局演练与历史方案比较","",
           "数据来自官方模拟器 practice-statistics-queue.sqlite3，G25O为问题3/4各100局自动演练。",""]
    for p in (3,4):
        lines.append(f"## 问题{p}")
        lines.append("| 指标 | Traditional | OldPlanB | G25O |")
        lines.append("| --- | ---: | ---: | ---: |")
        keys=[("runs","局数"),("total_sources","源总数"),("total_cleared","清除数"),("success_rate","全清率"),
              ("mean_v_per_source_s","平均虚拟时间/源(s)"),("median_v_per_source_s","中位虚拟时间/源(s)"),
              ("mean_program_s","平均程序运行时间(s)"),("median_program_s","程序时间中位数(s)"),
              ("mean_measures","平均测量次数"),("mean_switches","平均切频次数"),
              ("mean_clear_failures","平均失败清除")]
        for k,label in keys:
            vals=[]
            for g in ("Traditional","OldPlanB","G25O"):
                v=summary[f"problem{p}_{g}"].get(k,0)
                vals.append(f"{v:.3f}" if isinstance(v,float) else str(v))
            lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")
    (OUT/"official_g25o_200_comparison.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("saved", OUT/"official_g25o_200_summary.json")


if __name__=="__main__":
    main()
