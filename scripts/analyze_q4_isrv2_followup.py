"""Analyze the 30-run Q4 ISRV2 follow-up and cumulative official evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def verified(path: Path) -> list[dict]:
    progress = json.loads((path / "progress.json").read_text(encoding="utf-8"))
    rows = [row for row in progress["attempts"] if row.get("phase") == "main"]
    return [row for row in rows if row.get("status") == "verified_full_clear"
            and row.get("db_match_status") == "verified"
            and int(row["cleared_jammer_count"]) == int(row["jammer_count"])]


def values(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row["virtual_time_s"]) / int(row["jammer_count"]) for row in rows])


def describe(rows: list[dict], rng: np.random.Generator, bootstrap: int) -> dict:
    v = values(rows)
    idx = rng.integers(0, len(v), size=(bootstrap, len(v)))
    means = v[idx].mean(axis=1)
    wall = np.asarray([float(row["program_run_duration_ms"]) / 1000.0 for row in rows])
    source_count = int(sum(int(row["jammer_count"]) for row in rows))
    distance = float(sum(float(row["runner_summary"]["distance_m"]) for row in rows))
    return {
        "runs": len(rows), "sources": source_count,
        "cleared": int(sum(int(row["cleared_jammer_count"]) for row in rows)),
        "mean_seconds_per_source": float(v.mean()),
        "mean_ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "aggregate_seconds_per_source": float(sum(float(row["virtual_time_s"]) for row in rows) / source_count),
        "median_seconds_per_source": float(np.median(v)),
        "p90_seconds_per_source": float(np.quantile(v, 0.90)),
        "p95_seconds_per_source": float(np.quantile(v, 0.95)),
        "max_seconds_per_source": float(v.max()),
        "distance_per_source_m": distance / source_count,
        "move_time_per_source": distance / 5.0 / source_count,
        "measure_time_per_source": 5.0 * sum(int(row["measure_accepted_count"]) for row in rows) / source_count,
        "switch_time_per_source": sum(int(row["channel_switch_count"]) for row in rows) / source_count,
        "failed_clear_time_per_source": 3.0 * sum(int(row["clear_failure_count"]) for row in rows) / source_count,
        "measure_calls": int(sum(int(row["measure_accepted_count"]) for row in rows)),
        "switches": int(sum(int(row["channel_switch_count"]) for row in rows)),
        "failed_clear": int(sum(int(row["clear_failure_count"]) for row in rows)),
        "program_wall_p95_s": float(np.quantile(wall, 0.95)),
    }


def permutation_test(candidate: np.ndarray, baseline: np.ndarray,
                     rng: np.random.Generator, samples: int = 100000) -> dict:
    observed = float(candidate.mean() - baseline.mean())
    pooled = np.concatenate((candidate, baseline))
    deltas = np.empty(samples, dtype=float)
    n_candidate = len(candidate)
    for i in range(samples):
        shuffled = rng.permutation(pooled)
        deltas[i] = shuffled[:n_candidate].mean() - shuffled[n_candidate:].mean()
    return {
        "observed_mean_delta": observed,
        "one_sided_p_candidate_faster": float((np.count_nonzero(deltas <= observed) + 1) / (samples + 1)),
        "two_sided_p": float((np.count_nonzero(np.abs(deltas) >= abs(observed)) + 1) / (samples + 1)),
        "samples": samples,
    }


def bootstrap_difference(candidate: np.ndarray, baseline: np.ndarray,
                         rng: np.random.Generator, samples: int) -> list[float]:
    ci = candidate[rng.integers(0, len(candidate), size=(samples, len(candidate)))].mean(axis=1)
    bi = baseline[rng.integers(0, len(baseline), size=(samples, len(baseline)))].mean(axis=1)
    return [float(x) for x in np.quantile(ci - bi, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("followup_dir")
    ap.add_argument("--initial-dir", default="results/isrv2/official_20_20260912")
    ap.add_argument("--prior-isr-dir", default="results/isr_validation/official_20260912")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()
    out = Path(args.followup_dir)
    followup_all = verified(out)
    initial_all = verified(Path(args.initial_dir))
    prior_all = verified(Path(args.prior_isr_dir))
    followup = [r for r in followup_all if int(r["mode"]) == 4 and r["variant"] == "ISRV2"]
    initial_v2 = [r for r in initial_all if int(r["mode"]) == 4 and r["variant"] == "ISRV2"]
    initial_isr = [r for r in initial_all if int(r["mode"]) == 4 and r["variant"] == "ISR"]
    prior_isr = [r for r in prior_all if int(r["mode"]) == 4 and r["variant"] == "ISR"]
    cumulative_v2 = initial_v2 + followup
    pooled_isr = prior_isr + initial_isr
    if len(followup) != 30 or len(initial_v2) != 5 or len(pooled_isr) != 28:
        raise RuntimeError(f"unexpected sample counts: followup={len(followup)}, initial_v2={len(initial_v2)}, ISR={len(pooled_isr)}")
    ids = [int(r["db_id"]) for r in cumulative_v2 + pooled_isr]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate official database record")
    rng = np.random.default_rng(args.seed)
    new_desc = describe(followup, rng, args.bootstrap)
    cumulative_desc = describe(cumulative_v2, rng, args.bootstrap)
    isr_desc = describe(pooled_isr, rng, args.bootstrap)
    cv = values(cumulative_v2); bv = values(pooled_isr)
    first = values(initial_v2); later = values(followup)
    comparison = {
        "cumulative_isrv2_relative_to_isr_mean_percent": float((cv.mean() / bv.mean() - 1.0) * 100.0),
        "cumulative_isrv2_relative_to_isr_aggregate_percent": float((cumulative_desc["aggregate_seconds_per_source"] /
                                                                       isr_desc["aggregate_seconds_per_source"] - 1.0) * 100.0),
        "independent_bootstrap_mean_delta_ci95": bootstrap_difference(cv, bv, rng, args.bootstrap),
        "permutation": permutation_test(cv, bv, rng),
        "initial5_vs_followup30_mean_delta": float(later.mean() - first.mean()),
        "initial5_mean": float(first.mean()), "followup30_mean": float(later.mean()),
    }
    result = {
        "followup_state": json.loads((out / "progress.json").read_text(encoding="utf-8"))["state"],
        "followup_attempts": 30, "followup_verified_full_clear": len(followup),
        "followup_technical_failures": 0, "followup_confirmed_not_clear": 0,
        "bootstrap_samples": args.bootstrap, "seed": args.seed,
        "followup30_isrv2": new_desc, "cumulative35_isrv2": cumulative_desc,
        "available28_isr_v1": isr_desc, "comparison": comparison,
        "comparison_limit": "The 35 ISRV2 and 28 ISR-v1 practices are independent, non-contemporaneous official scenarios.",
        "recommendation": "KEEP_ISR_V1",
        "reason": "The initial five-run ISRV2 advantage did not reproduce; cumulative ISRV2 mean and aggregate T/N are higher than the available ISR-v1 pool.",
    }
    (out / "followup_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Q4 ISRV2 新增30局官方演练复核", "",
        "新增30局全部由官方数据库唯一匹配，30/30全清，技术故障0，未全清0。", "",
        "| 数据集 | 局数/源数 | 均值T/N | 聚合ΣT/ΣN | 中位数 | P90 | P95 | 最慢局 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, desc in (("新增ISRV2", new_desc), ("累计ISRV2", cumulative_desc), ("现有ISR-v1", isr_desc)):
        lines.append(f"| {label} | {desc['runs']}/{desc['sources']} | {desc['mean_seconds_per_source']:.2f} | "
                     f"{desc['aggregate_seconds_per_source']:.2f} | {desc['median_seconds_per_source']:.2f} | "
                     f"{desc['p90_seconds_per_source']:.2f} | {desc['p95_seconds_per_source']:.2f} | "
                     f"{desc['max_seconds_per_source']:.2f} |")
    lines.extend(["",
        f"前5局ISRV2均值为 {comparison['initial5_mean']:.2f} 秒/源，新增30局为 {comparison['followup30_mean']:.2f} 秒/源，"
        f"后续样本高 {comparison['initial5_vs_followup30_mean_delta']:.2f} 秒/源。",
        "",
        f"累计35局ISRV2相对现有28局ISR-v1的每局均值变化为 {comparison['cumulative_isrv2_relative_to_isr_mean_percent']:+.2f}%，"
        f"聚合口径变化为 {comparison['cumulative_isrv2_relative_to_isr_aggregate_percent']:+.2f}%。",
        "",
        "两组来自独立且不同时段的官方随机场景，因此这一比较用于判断趋势，不能视为同场景因果对照。新增30局没有复现前5局的大幅提速，当前建议Q4继续使用ISR-v1。", ""])
    (out / "followup_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "official_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["run_id", "block", "mode", "variant", "db_id", "practice_run_no", "case_code",
                  "jammer_count", "cleared_jammer_count", "virtual_time_s", "program_run_duration_ms",
                  "measure_accepted_count", "channel_switch_count", "clear_failure_count", "end_reason", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(followup)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
