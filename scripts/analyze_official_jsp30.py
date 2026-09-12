"""Analyze the frozen 15-block ISR-v1 versus JSP official-practice batch."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def describe(rows: list[dict]) -> dict:
    if not rows:
        return {"runs": 0, "sources": 0, "cleared": 0,
                "mean_seconds_per_source": None, "aggregate_seconds_per_source": None,
                "median_seconds_per_source": None, "p90_seconds_per_source": None,
                "p95_seconds_per_source": None, "max_seconds_per_source": None,
                "mean_distance_per_source_m": None, "measure_calls": 0, "switches": 0,
                "failed_clear": 0, "program_wall_p95_s": None}
    values = np.asarray([float(row["virtual_time_s"]) / int(row["jammer_count"])
                         for row in rows])
    walls = np.asarray([float(row["program_run_duration_ms"]) / 1000.0 for row in rows])
    distances = np.asarray([
        float(row.get("runner_summary", {}).get("distance_m", np.nan)) /
        int(row["jammer_count"]) for row in rows])
    return {
        "runs": len(rows), "sources": sum(int(row["jammer_count"]) for row in rows),
        "cleared": sum(int(row["cleared_jammer_count"]) for row in rows),
        "mean_seconds_per_source": float(values.mean()),
        "aggregate_seconds_per_source": float(
            sum(float(row["virtual_time_s"]) for row in rows) /
            sum(int(row["jammer_count"]) for row in rows)),
        "median_seconds_per_source": float(np.median(values)),
        "p90_seconds_per_source": float(np.quantile(values, .90)),
        "p95_seconds_per_source": float(np.quantile(values, .95)),
        "max_seconds_per_source": float(values.max()),
        "mean_distance_per_source_m": float(np.nanmean(distances)),
        "measure_calls": sum(int(row["measure_accepted_count"]) for row in rows),
        "switches": sum(int(row["channel_switch_count"]) for row in rows),
        "failed_clear": sum(int(row["clear_failure_count"]) for row in rows),
        "program_wall_p95_s": float(np.quantile(walls, .95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    attempts = [row for row in progress["attempts"] if row.get("phase") == "main"]
    valid = [row for row in attempts if row.get("status") == "verified_full_clear"
             and row.get("db_match_status") == "verified"
             and int(row["jammer_count"]) == int(row["cleared_jammer_count"])]
    modes = sorted({int(row["mode"]) for row in attempts if int(row.get("mode", -1)) in (3, 4)})
    if len(modes) != 1:
        raise RuntimeError("JSP30 ledger must contain exactly one problem mode")
    mode = modes[0]
    baseline = [row for row in valid if row["variant"] == "ISR"]
    candidate = [row for row in valid if row["variant"] == "JSP"]
    base_by_block = {int(row["block"]): float(row["virtual_time_s"]) /
                     int(row["jammer_count"]) for row in baseline}
    jsp_by_block = {int(row["block"]): float(row["virtual_time_s"]) /
                    int(row["jammer_count"]) for row in candidate}
    blocks = sorted(set(base_by_block) & set(jsp_by_block))
    base = describe(baseline); jsp = describe(candidate)
    delta = np.asarray([jsp_by_block[block] - base_by_block[block] for block in blocks])
    if len(delta):
        rng = np.random.default_rng(args.bootstrap_seed)
        indices = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
        bootstrap = delta[indices].mean(axis=1)
        ci95 = [float(value) for value in np.quantile(bootstrap, [.025, .975])]
        relative = (jsp["mean_seconds_per_source"] / base["mean_seconds_per_source"] - 1) * 100
    else:
        ci95, relative = [None, None], None
    checks = {
        "batch_complete_30": progress.get("state") == "complete" and len(attempts) == 30,
        "all_attempts_verified_full_clear": len(valid) == 30,
        "fifteen_per_arm": len(baseline) == 15 and len(candidate) == 15,
        "mean_improves_3pct": relative is not None and relative <= -3.0,
        "block_ci95_upper_below_zero": ci95[1] is not None and ci95[1] < 0.0,
        "p95_within_103pct": bool(baseline and candidate and
            jsp["p95_seconds_per_source"] <= 1.03 * base["p95_seconds_per_source"]),
        "max_within_103pct": bool(baseline and candidate and
            jsp["max_seconds_per_source"] <= 1.03 * base["max_seconds_per_source"]),
    }
    output = {
        "experiment_state": progress.get("state"), "mode": mode,
        "attempts": len(attempts), "verified_full_clear_attempts": len(valid),
        "technical_failures": sum(bool(row.get("technical_failure")) for row in attempts),
        "confirmed_not_clear": sum(bool(row.get("confirmed_not_clear")) for row in attempts),
        "design": "15 randomized time blocks; one independent ISR and JSP practice per block",
        "not_same_scenario_paired": True, "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples, "complete_time_blocks": len(blocks),
        "ISR": base, "JSP": jsp, "relative_change_percent": float(relative),
        "block_mean_delta_seconds_per_source": float(delta.mean()) if len(delta) else None,
        "block_bootstrap_mean_delta_ci95": ci95,
        "checks": checks, "adopt_jsp": all(checks.values()),
        "recommendation": "JSP" if all(checks.values()) else "ISR-v1",
    }
    (root / "official_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    ci = output["block_bootstrap_mean_delta_ci95"]
    fmt = lambda value: "NA" if value is None else f"{value:.2f}"
    lines = [
        f"# Q{mode} JSP：30 局官方演练分析", "",
        f"批次状态：`{output['experiment_state']}`；全部尝试 {len(attempts)} 局，"
        f"数据库核对全清 {len(valid)} 局，技术故障 {output['technical_failures']} 局，"
        f"确认未全清 {output['confirmed_not_clear']} 局。", "",
        "官方场景不可复放。15 个时间区组只用于控制批次时段影响，不能解释为同场景配对。", "",
        "| 策略 | 局数 | 平均秒/源 | 聚合秒/源 | P90 | P95 | 最慢局 |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| ISR-v1 | {base['runs']} | {fmt(base['mean_seconds_per_source'])} | "
        f"{fmt(base['aggregate_seconds_per_source'])} | {fmt(base['p90_seconds_per_source'])} | "
        f"{fmt(base['p95_seconds_per_source'])} | {fmt(base['max_seconds_per_source'])} |",
        f"| JSP | {jsp['runs']} | {fmt(jsp['mean_seconds_per_source'])} | "
        f"{fmt(jsp['aggregate_seconds_per_source'])} | {fmt(jsp['p90_seconds_per_source'])} | "
        f"{fmt(jsp['p95_seconds_per_source'])} | {fmt(jsp['max_seconds_per_source'])} |", "",
        (f"相对变化：**{relative:+.2f}%**；区组均值差 95% bootstrap 区间："
         f"**[{ci[0]:.2f}, {ci[1]:.2f}] 秒/源**。" if relative is not None else
         "当前没有完整 ISR/JSP 时间区组，不能计算性能区间。"), "",
        f"采用判定：**{'采用 JSP' if output['adopt_jsp'] else '保留 ISR-v1'}**。", "",
    ]
    (root / "official_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    fields = ["run_id", "block", "block_position", "mode", "variant", "db_id",
              "practice_run_no", "case_code", "jammer_count", "cleared_jammer_count",
              "virtual_time_s", "program_run_duration_ms", "measure_accepted_count",
              "channel_switch_count", "clear_failure_count", "end_reason", "status"]
    with (root / "official_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(attempts)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
