"""Analyze the frozen 20-run official ISR-v1 versus ISRV2 practice batch."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def describe(rows: list[dict]) -> dict:
    v = np.asarray([float(r["virtual_time_s"]) / int(r["jammer_count"]) for r in rows])
    distance = np.asarray([float(r["runner_summary"].get("distance_m", np.nan)) / int(r["jammer_count"]) for r in rows])
    wall = np.asarray([float(r["program_run_duration_ms"]) / 1000.0 for r in rows])
    return {
        "runs": len(rows), "sources": int(sum(int(r["jammer_count"]) for r in rows)),
        "cleared": int(sum(int(r["cleared_jammer_count"]) for r in rows)),
        "mean_seconds_per_source": float(v.mean()),
        "aggregate_seconds_per_source": float(sum(float(r["virtual_time_s"]) for r in rows) /
                                              sum(int(r["jammer_count"]) for r in rows)),
        "median_seconds_per_source": float(np.median(v)),
        "p90_seconds_per_source": float(np.quantile(v, 0.90)),
        "p95_seconds_per_source": float(np.quantile(v, 0.95)),
        "max_seconds_per_source": float(v.max()),
        "mean_distance_per_source_m": float(np.nanmean(distance)),
        "measure_calls": int(sum(int(r["measure_accepted_count"]) for r in rows)),
        "switches": int(sum(int(r["channel_switch_count"]) for r in rows)),
        "failed_clear": int(sum(int(r["clear_failure_count"]) for r in rows)),
        "program_wall_p95_s": float(np.quantile(wall, 0.95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--bootstrap-seed", type=int, default=20260912)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    args = ap.parse_args()
    root = Path(args.experiment_dir)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    attempts = [a for a in progress["attempts"] if a.get("phase") == "main"]
    verified = [a for a in attempts if a.get("status") == "verified_full_clear"
                and a.get("db_match_status") == "verified"
                and int(a["jammer_count"]) == int(a["cleared_jammer_count"])]
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = []
    for mode in (3, 4):
        baseline = [a for a in verified if int(a["mode"]) == mode and a["variant"] == "ISR"]
        candidate = [a for a in verified if int(a["mode"]) == mode and a["variant"] == "ISRV2"]
        bmap = {int(a["block"]): float(a["virtual_time_s"]) / int(a["jammer_count"]) for a in baseline}
        cmap = {int(a["block"]): float(a["virtual_time_s"]) / int(a["jammer_count"]) for a in candidate}
        blocks = sorted(set(bmap) & set(cmap))
        delta = np.asarray([cmap[x] - bmap[x] for x in blocks])
        idx = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
        means = delta[idx].mean(axis=1)
        b = describe(baseline); c = describe(candidate)
        base_values = np.asarray([float(a["virtual_time_s"]) / int(a["jammer_count"]) for a in baseline])
        candidate_values = np.asarray([float(a["virtual_time_s"]) / int(a["jammer_count"]) for a in candidate])
        pooled = np.concatenate((base_values, candidate_values))
        observed = float(candidate_values.mean() - base_values.mean())
        permuted = []
        for chosen in itertools.combinations(range(len(pooled)), len(candidate_values)):
            mask = np.zeros(len(pooled), dtype=bool); mask[list(chosen)] = True
            permuted.append(float(pooled[mask].mean() - pooled[~mask].mean()))
        permuted = np.asarray(permuted)
        # Add-one form remains valid if this is later changed from exhaustive to sampled permutations.
        one_sided_p = float((np.sum(permuted <= observed) + 1) / (len(permuted) + 1))
        two_sided_p = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))
        comparisons.append({
            "mode": mode, "time_blocks": len(blocks), "all_attempts_verified_full_clear": len(baseline) == 5 and len(candidate) == 5,
            "ISR": b, "ISRV2": c,
            "relative_change_percent": float((c["mean_seconds_per_source"] / b["mean_seconds_per_source"] - 1.0) * 100.0),
            "block_mean_delta_seconds_per_source": float(delta.mean()),
            "block_bootstrap_mean_delta_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "block_bootstrap_mean_delta_ci97_5": [float(x) for x in np.quantile(means, [0.0125, 0.9875])],
            "p95_ratio": float(c["p95_seconds_per_source"] / b["p95_seconds_per_source"]),
            "exact_independent_permutation_p_candidate_faster": one_sided_p,
            "exact_independent_permutation_p_two_sided": two_sided_p,
            "exploratory_decision": "ISRV2_FASTER_IN_SAMPLE" if c["mean_seconds_per_source"] < b["mean_seconds_per_source"] else "ISR_FASTER_IN_SAMPLE",
            "adoption_decision": "KEEP_ISR_V1",
            "adoption_reason": "Only five independent official practices per arm; local development gate was not passed.",
        })
    output = {
        "experiment_state": progress.get("state"), "attempts": len(attempts),
        "verified_full_clear_attempts": len(verified), "technical_failures": sum(bool(a.get("technical_failure")) for a in attempts),
        "confirmed_not_clear": sum(bool(a.get("confirmed_not_clear")) for a in attempts),
        "design": "5 randomized time blocks; one independent practice for each Q3/Q4 x ISR/ISRV2 cell per block",
        "not_same_scenario_paired": True, "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed, "comparisons": comparisons,
        "overall_recommendation": {"Q3": "ISR-v1", "Q4": "ISR-v1"},
    }
    (root / "official_analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ISR-v1 与 ISRV2：20 局官方演练分析", "",
        f"批次状态：`{output['experiment_state']}`。共 {len(attempts)} 局，官方数据库核对全清 {len(verified)} 局，"
        f"技术故障 {output['technical_failures']} 局，未全清 {output['confirmed_not_clear']} 局。", "",
        "每个区组包含 Q3/Q4 × ISR/ISRV2 各一局，顺序随机。官方场景不可复放，区组只控制批次时间影响，不能视为同场景配对。", "",
        "| 题目 | ISR-v1 均值 | ISRV2 均值 | 相对变化 | 95% 区组 bootstrap 差值 | ISR P95 | ISRV2 P95 | 当前建议 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in comparisons:
        ci = x["block_bootstrap_mean_delta_ci95"]
        lines.append(f"| Q{x['mode']} | {x['ISR']['mean_seconds_per_source']:.2f} | {x['ISRV2']['mean_seconds_per_source']:.2f} | "
                     f"{x['relative_change_percent']:+.2f}% | [{ci[0]:.2f}, {ci[1]:.2f}] | "
                     f"{x['ISR']['p95_seconds_per_source']:.2f} | {x['ISRV2']['p95_seconds_per_source']:.2f} | 保留 ISR-v1 |")
    lines.extend(["", "正值表示 ISRV2 更慢，负值表示 ISRV2 更快。每臂只有 5 局，区间很宽；结合 ISRV2 未通过本地开发门槛，20 局结果不用于替换 ISR-v1。", ""])
    (root / "official_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    fields = ["run_id", "block", "block_position", "mode", "variant", "db_id", "practice_run_no", "case_code",
              "jammer_count", "cleared_jammer_count", "virtual_time_s", "program_run_duration_ms", "measure_accepted_count",
              "channel_switch_count", "clear_failure_count", "end_reason", "status"]
    with (root / "official_runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(attempts)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
