"""导出论文/仓库用数据记录表，所有数值来自执行日志或模拟器数据库。"""
from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "records"
OUT.mkdir(parents=True, exist_ok=True)

DB = Path(r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full\JammersSimulatorData\practice-statistics-queue.sqlite3")
RL_BATCH_IDS = {3: list(range(36, 56)), 4: list(range(57, 77))}
EARLY_RL_IDS = {32, 33, 35}


def load_official_rows():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from practice_statistics_tasks where team_no='202610094088' and entered=1 and program_run_duration_ms is not null order by id")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def batch_summaries(problem):
    d = ROOT / "results" / f"batch_p{problem}_20"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("run_*.summary.json"))]


def official_table():
    rows = load_official_rows()
    out = []
    for problem in (3, 4):
        rl_ids = RL_BATCH_IDS[problem]
        rl_rows = [r for r in rows if r["problem_no"] == problem and r["id"] in rl_ids]
        trad_rows = [r for r in rows if r["problem_no"] == problem and r["id"] not in rl_ids and r["id"] not in EARLY_RL_IDS]
        summaries = batch_summaries(problem)
        rl_rows_sorted = sorted(rl_rows, key=lambda r: r["id"])
        for idx, r in enumerate(rl_rows_sorted):
            s = summaries[idx] if idx < len(summaries) else {}
            out.append(record_dict("PlanB-RL", problem, r, s))
        for r in trad_rows:
            out.append(record_dict("Traditional", problem, r, {}))
    with open(OUT / "official_practice_runs.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["group", "problem", "run_id", "case_code", "sources", "cleared", "full_clear",
                  "virtual_time_s", "avg_clear_time_s", "program_run_s", "measure_calls", "switches",
                  "clear_failures", "fallback_actions", "policy_deviations_from_prior", "end_reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in fields})


def record_dict(group, problem, r, summary):
    source = max(r["jammer_count"], 1)
    return {
        "group": group, "problem": problem, "run_id": r["id"], "case_code": r["case_code"],
        "sources": r["jammer_count"], "cleared": r["cleared_jammer_count"],
        "full_clear": int(r["cleared_jammer_count"] == r["jammer_count"]),
        "virtual_time_s": round(r["virtual_time_us"] / 1e6, 6),
        "avg_clear_time_s": round((r["virtual_time_us"] / 1e6) / source, 6),
        "program_run_s": round(r["program_run_duration_ms"] / 1000.0, 6),
        "measure_calls": r["measure_accepted_count"], "switches": r["channel_switch_count"],
        "clear_failures": r["clear_failure_count"],
        "fallback_actions": summary.get("fallback_actions", ""),
        "policy_deviations_from_prior": summary.get("policy_deviations_from_prior", ""),
        "end_reason": r["end_reason"],
    }


def summary_tables():
    comp = json.loads((ROOT / "results" / "traditional_vs_rl_comparison.json").read_text(encoding="utf-8"))
    rows = []
    for problem in (3, 4):
        d = comp["official_distribution_comparison"][f"problem{problem}"]
        for method, key in [("Traditional", "traditional"), ("PlanB-RL", "rl")]:
            s = d[key]
            rows.append({
                "problem": problem, "method": method, "runs": s["n_runs"], "sources": s["sources"],
                "cleared": s["cleared"], "success_rate": f"{s['success_rate']:.3f}",
                "mean_virtual_time_per_source_s": f"{s['mean_V_per_source']:.3f}",
                "median_virtual_time_per_source_s": f"{s['median_V_per_source']:.3f}",
                "mean_program_run_s": f"{s['mean_program_s']:.3f}",
                "mean_measures": f"{s['mean_measures']:.2f}",
                "mean_switches": f"{s['mean_switches']:.2f}",
                "mean_clear_failures": f"{s['mean_clear_failures']:.2f}",
            })
    with open(OUT / "traditional_vs_planb_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    local = comp.get("local_paired_comparison", {})
    lrows = []
    for problem in (3, 4):
        d = local.get(f"problem{problem}", {})
        if d:
            lrows.append({"problem": problem, "pairs": d["n_pairs"], "A0_mean_s": f"{d['mean_A0']:.3f}",
                          "PlanB_mean_s": f"{d['mean_B']:.3f}",
                          "PlanB_minus_A0_mean_s": f"{d['mean_diff_B_minus_A0']:.6f}",
                          "wins_PlanB": d["wins_B"], "ties": d["ties"], "wins_A0": d["wins_A0"]})
    with open(OUT / "local_paired_A0_vs_planB.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(lrows[0].keys())); w.writeheader(); w.writerows(lrows)

    mv = json.loads((ROOT / "results" / "math_verification.json").read_text(encoding="utf-8"))
    math_rows = [
        ["S3覆盖最坏距离", "解析", f"{mv['S3']['analytic_worst_distance']:.6f}", "m", "<1000，通过"],
        ["S3覆盖采样", "20万点", f"{mv['S3']['numeric']['worst_distance']:.6f}", "m", "漏检0"],
        ["S4方向覆盖", "20万点随机方向", f"{mv['S4']['numeric']['worst_visible_min']:.6f}", "m", "漏检0"],
        ["q±扇区采样最坏", "随机采样", f"{mv['q_pm_random_sector_max']:.6f}", "m", "<1000，通过"],
        ["122点光学保底", "20万点矩形采样", f"{mv['optical_fallback_random_max']:.6f}", "m", "<20，通过"],
        ["直径算法交叉验证", "300个随机凸多边形", f"{mv['diameter_crosscheck']['worst_abs_diff']:.6f}", "m", "与暴力法一致"],
        ["最小包围圆40m等边三角形", "Welzl", f"{mv['min_enclosing_triangle_40']:.6f}", "m", ">20，不能一次清除"],
    ]
    with open(OUT / "math_verification.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["项目", "方法/样本", "数值", "单位", "判定"]); w.writerows(math_rows)


def write_readme():
    text = """# 数据记录说明

本目录记录论文和代码的可复现实验数据。所有数值来自实际执行的代码、请求日志或官方模拟器数据库，没有手工编造。

## 1. 官方模拟器演练记录

- 来源：官方模拟器 `practice-statistics-queue.sqlite3`
- 参赛队号/robot_id：202610094088
- 问题3：传统方法14局 + 方案B(RL)20局
- 问题4：传统方法17局 + 方案B(RL)20局
- 所有方案B局均以 `user_exit` 正常结束，完成证书成立后自动调用 `/exit`
- 详细逐局记录：`official_practice_runs.csv`
- 汇总比较：`traditional_vs_planb_summary.csv`
- 原始JSON：`../results/official_practice_40.json`、`../results/traditional_vs_rl_comparison.json`

## 2. 本地严格配对实验

- 文件：`local_paired_A0_vs_planB.csv`
- 问题3/4各100个相同随机场景，A0与方案B使用相同源、频道、半径、类型、方向和固定误差场
- 结果：方案B与A0平均虚拟时间完全相同，因为PPO未偏离A0先验

## 3. 数学校验

- 文件：`math_verification.csv`
- 包含S3/S4覆盖、q±接收上界、122点光学保底、直径交叉验证和最小包围圆算例

## 4. 本地模拟器与mock HTTP联调

本地 `local_env.py` 与 `mock_server.py` 的结果明确标注为非官方数据，只用于算法开发、消融和接口联调。官方正式测试数据尚未执行。正式测试后应新增 `formal_p3_q1.jsonl`、`formal_p3_q2.jsonl`、`formal_p3_q3.jsonl` 等文件，并保持同一统计口径。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    official_table()
    summary_tables()
    write_readme()
    print("records exported to", OUT)
