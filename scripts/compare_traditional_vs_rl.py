"""官方演练传统方法 vs 方案B(RL) 的分布比较。

注意：官方演练案例随机生成且不能重放，所以这是非配对分布比较；
本地A0 vs PPO 的严格配对结果另行报告。
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DB = Path(r"C:\Users\huani\Desktop\CUMCM2026B\Jammers-simulator-full-win64\Jammers-simulator-full\JammersSimulatorData\practice-statistics-queue.sqlite3")
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

# 当前数据库中的方案B演练记录id（问题3：早期2局 + 20局批量；问题4：早期1局 + 20局批量）
OLD_PLANB_IDS = {
    3: set(range(36, 56)),
    4: set(range(57, 77)),
}
# 有效的传统方法官方演练记录（问题4的id14为q3-mode-on-q4无效局，已剔除）
TRADITIONAL_IDS = {
    3: {1, 3, 5, 6, 7, 11, 12, 13, 26, 27, 28, 29, 30, 31},
    4: {2, 4, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
}


def load_rows():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from practice_statistics_tasks where team_no='202610094088' and entered=1 and program_run_duration_ms is not null order by id")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def metrics(rows, problem, method):
    out = []
    for r in rows:
        if r["problem_no"] != problem:
            continue
        if method == "Traditional":
            if r["id"] not in TRADITIONAL_IDS[problem]:
                continue
        elif method in ("RL", "OldPlanB"):
            if r["id"] not in OLD_PLANB_IDS[problem]:
                continue
        else:
            continue
        src = max(r["jammer_count"], 1)
        out.append({
            "id": r["id"], "case_code": r["case_code"], "sources": r["jammer_count"],
            "cleared": r["cleared_jammer_count"], "success": int(r["cleared_jammer_count"] == r["jammer_count"]),
            "V": r["virtual_time_us"] / 1e6, "V_per_source": (r["virtual_time_us"] / 1e6) / src,
            "program_s": r["program_run_duration_ms"] / 1000.0,
            "measures": r["measure_accepted_count"], "switches": r["channel_switch_count"],
            "clear_failures": r["clear_failure_count"], "end_reason": r["end_reason"],
        })
    return out


def summarize(items):
    if not items:
        return {}
    vals = lambda k: np.asarray([x[k] for x in items], dtype=float)
    return {
        "n_runs": len(items),
        "sources": int(vals("sources").sum()),
        "cleared": int(vals("cleared").sum()),
        "success_rate": float(vals("success").mean()),
        "full_clear_runs": int(vals("success").sum()),
        "mean_sources": float(vals("sources").mean()),
        "mean_V": float(vals("V").mean()),
        "median_V": float(np.median(vals("V"))),
        "mean_V_per_source": float(vals("V_per_source").mean()),
        "median_V_per_source": float(np.median(vals("V_per_source"))),
        "std_V_per_source": float(vals("V_per_source").std(ddof=1)) if len(items) > 1 else 0.0,
        "mean_program_s": float(vals("program_s").mean()),
        "median_program_s": float(np.median(vals("program_s"))),
        "mean_measures": float(vals("measures").mean()),
        "mean_switches": float(vals("switches").mean()),
        "mean_clear_failures": float(vals("clear_failures").mean()),
    }


def bootstrap_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diffs.append(bb.mean() - aa.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def main():
    rows = load_rows()
    report = {"official_distribution_comparison": {}, "local_paired_comparison": {}}
    for problem in (3, 4):
        trad = metrics(rows, problem, "Traditional")
        rl = metrics(rows, problem, "RL")
        st, sr = summarize(trad), summarize(rl)
        # 仅对完整清除局做敏感性分析
        trad_ok = [x for x in trad if x["success"] == 1]
        rl_ok = [x for x in rl if x["success"] == 1]
        a = [x["V_per_source"] for x in trad]
        b = [x["V_per_source"] for x in rl]
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            u, p = float("nan"), float("nan")
        diff, lo, hi = bootstrap_diff(a, b)
        report["official_distribution_comparison"][f"problem{problem}"] = {
            "traditional": st, "rl": sr,
            "traditional_full_clear_only": summarize(trad_ok),
            "rl_full_clear_only": summarize(rl_ok),
            "mannwhitney_u": float(u), "mannwhitney_p": float(p),
            "mean_diff_V_per_source_rl_minus_trad": diff,
            "bootstrap_ci95": [lo, hi],
        }

    # 本地严格配对结果
    for problem in (3, 4):
        base = RES / f"eval_baseline_mode{problem}.jsonl"
        ppo = RES / f"eval_ppo_mode{problem}.jsonl"
        if base.exists() and ppo.exists():
            ba = [json.loads(x) for x in base.read_text(encoding="utf-8").splitlines() if x.strip()]
            pp = [json.loads(x) for x in ppo.read_text(encoding="utf-8").splitlines() if x.strip()]
            n = min(len(ba), len(pp))
            a = np.asarray([ba[i]["virtual_time"] for i in range(n)], dtype=float)
            b = np.asarray([pp[i]["virtual_time"] for i in range(n)], dtype=float)
            d = b - a
            report["local_paired_comparison"][f"problem{problem}"] = {
                "n_pairs": n, "mean_A0": float(a.mean()), "mean_B": float(b.mean()),
                "mean_diff_B_minus_A0": float(d.mean()), "median_diff": float(np.median(d)),
                "wins_B": int((d < -1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
                "wins_A0": int((d > 1e-9).sum()),
            }

    (RES / "traditional_vs_rl_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图1：V/源箱线图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, problem in zip(axes, (3, 4)):
        trad = metrics(rows, problem, "Traditional")
        rl = metrics(rows, problem, "RL")
        data = [[x["V_per_source"] for x in trad], [x["V_per_source"] for x in rl]]
        ax.boxplot(data, labels=["Traditional", "Plan B (RL)"], showmeans=True)
        ax.set_title(f"Problem {problem}: virtual time per source")
        ax.set_ylabel("s/source")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "comparison_v_per_source_box.png", dpi=180)
    plt.close(fig)

    # 图2：成功率与程序运行时间
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(2)
    w = 0.35
    for j, problem in enumerate((3, 4)):
        trad = metrics(rows, problem, "Traditional")
        rl = metrics(rows, problem, "RL")
        st, sr = summarize(trad), summarize(rl)
        axes[0].bar(x + (j - 0.5) * w * 2, [st["success_rate"], sr["success_rate"]],
                    width=w, label=f"Problem {problem}")
        axes[1].bar(x + (j - 0.5) * w * 2, [st["mean_program_s"], sr["mean_program_s"]],
                    width=w, label=f"Problem {problem}")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Traditional", "Plan B (RL)"])
    axes[0].set_ylabel("full-clear rate"); axes[0].set_ylim(0, 1.1); axes[0].legend()
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Traditional", "Plan B (RL)"])
    axes[1].set_ylabel("mean program runtime (s)"); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "comparison_success_runtime.png", dpi=180)
    plt.close(fig)

    # 生成 Markdown
    lines = ["# 传统方法 vs 方案B(RL) 比较", "", "注意：官方演练案例随机生成、不能重放，以下为非配对分布比较。", ""]
    for problem in (3, 4):
        d = report["official_distribution_comparison"][f"problem{problem}"]
        lines.append(f"## 问题{problem}")
        lines.append("| 指标 | 传统方法 | 方案B(RL) |")
        lines.append("| --- | ---: | ---: |")
        keys = [("n_runs", "局数"), ("sources", "源总数"), ("cleared", "清除数"), ("success_rate", "全清率"),
                ("mean_V_per_source", "平均虚拟时间/源(s)"), ("median_V_per_source", "中位虚拟时间/源(s)"),
                ("mean_program_s", "平均程序运行时间(s)"), ("mean_measures", "平均测量次数"),
                ("mean_switches", "平均切频次数"), ("mean_clear_failures", "平均失败清除")]
        for k, label in keys:
            v1, v2 = d["traditional"].get(k), d["rl"].get(k)
            if isinstance(v1, float):
                lines.append(f"| {label} | {v1:.3f} | {v2:.3f} |")
            else:
                lines.append(f"| {label} | {v1} | {v2} |")
        lines.append(f"| Mann-Whitney p |  | {d['mannwhitney_p']:.4f} |")
        lines.append(f"| RL-Trad 均值差(95% CI) |  | {d['mean_diff_V_per_source_rl_minus_trad']:.3f} ({d['bootstrap_ci95'][0]:.3f}, {d['bootstrap_ci95'][1]:.3f}) |")
        lines.append("")
    lines.append("## 本地严格配对结果")
    lines.append("| 问题 | 配对数 | A0平均(s) | 方案B平均(s) | B-A0均值(s) | B胜/平/A0胜 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for problem in (3, 4):
        d = report["local_paired_comparison"].get(f"problem{problem}")
        if d:
            lines.append(f"| {problem} | {d['n_pairs']} | {d['mean_A0']:.1f} | {d['mean_B']:.1f} | {d['mean_diff_B_minus_A0']:.3f} | {d['wins_B']}/{d['ties']}/{d['wins_A0']} |")
    lines.append("")
    lines.append("解释：本地严格配对中方案B与A0完全相同，因为PPO在100个测试场景中没有偏离A0先验。官方演练是分布比较，问题3方案B较快，问题4两组接近；差异可能来自随机案例和共用模块版本，不能单独归因于RL。")
    (RES / "traditional_vs_rl_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
