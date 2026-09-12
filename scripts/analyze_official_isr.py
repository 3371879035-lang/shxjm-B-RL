"""Analyze a completed practice-only ISR experiment from its verified ledger."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def describe(values: np.ndarray) -> dict:
    return {
        "runs": int(len(values)), "mean": float(values.mean()),
        "median": float(np.median(values)), "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)), "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    attempts = [a for a in progress["attempts"] if a.get("phase") == "main"]
    all_attempts = len(attempts)
    verified = [
        a for a in attempts
        if a.get("status") == "verified_full_clear"
        and a.get("db_match_status") == "verified"
        and int(a["cleared_jammer_count"]) == int(a["jammer_count"])
    ]
    exclusions = [a for a in attempts if a not in verified]
    comparisons = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for mode in (3, 4):
        groups = {}
        for variant in ("G25OR", "ISR"):
            rows = [a for a in verified if int(a["mode"]) == mode and a["variant"] == variant]
            values = np.asarray([float(a["virtual_time_s"]) / int(a["jammer_count"]) for a in rows])
            groups[variant] = {"rows": rows, "values": values, **(describe(values) if len(values) else {})}
        base = groups["G25OR"]["values"]
        candidate = groups["ISR"]["values"]
        block_base = {int(a["block"]): float(a["virtual_time_s"]) / int(a["jammer_count"])
                      for a in groups["G25OR"]["rows"]}
        block_isr = {int(a["block"]): float(a["virtual_time_s"]) / int(a["jammer_count"])
                     for a in groups["ISR"]["rows"]}
        common = sorted(set(block_base) & set(block_isr))
        deltas = np.asarray([block_isr[b] - block_base[b] for b in common])
        ci = []
        if len(deltas):
            indices = rng.integers(0, len(deltas), size=(args.bootstrap_samples, len(deltas)))
            ci = [float(x) for x in np.quantile(deltas[indices].mean(axis=1), [0.025, 0.975])]
        relative = float((candidate.mean() / base.mean() - 1.0) * 100.0) if len(base) and len(candidate) else None
        all_group_attempts = [a for a in attempts if int(a.get("mode", -1)) == mode]
        correctness = bool(
            len(all_group_attempts) == 200
            and all(a.get("status") == "verified_full_clear" for a in all_group_attempts)
        )
        adopt = bool(
            correctness and relative is not None and relative <= -3.0
            and len(ci) == 2 and ci[1] < 0.0
            and groups["ISR"].get("p95", float("inf")) <= groups["G25OR"].get("p95", 0.0) * 1.03
        )
        comparisons.append({
            "mode": mode, "correctness_gate": correctness,
            "G25OR": {k: v for k, v in groups["G25OR"].items() if k not in ("rows", "values")},
            "ISR": {k: v for k, v in groups["ISR"].items() if k not in ("rows", "values")},
            "relative_change_percent": relative, "common_time_blocks": len(common),
            "block_bootstrap_mean_delta_ci95": ci,
            "tail_gate": bool(len(base) and len(candidate) and np.quantile(candidate, .95) <= np.quantile(base, .95) * 1.03),
            "decision": "ADOPT_ISR" if adopt else "KEEP_G25OR",
        })

    output = {
        "experiment_state": progress.get("state"), "main_attempts": all_attempts,
        "verified_main_attempts": len(verified), "excluded_attempts": len(exclusions),
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed,
        "adoption_rule": "all 200 attempts per mode verified full-clear; ISR mean >=3% faster; CI upper <0; ISR P95 <=103% of baseline",
        "comparisons": comparisons,
        "exclusions": [{k: a.get(k) for k in ("run_id", "mode", "variant", "status", "error")} for a in exclusions],
    }
    (root / "official_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# ISR official practice analysis", "",
        f"Experiment state: `{output['experiment_state']}`. Main attempts: {all_attempts}; "
        f"verified attempts: {len(verified)}; excluded attempts: {len(exclusions)}.", "",
        "Official practice cases cannot be replayed. The fixed randomized time blocks control "
        "batch order, but these results are not same-case paired observations.", "",
        "| Problem | G25OR mean | ISR mean | Change | 95% block CI | G25OR P95 | ISR P95 | Decision |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for item in comparisons:
        ci = item["block_bootstrap_mean_delta_ci95"]
        ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "unavailable"
        change_text = (f"{item['relative_change_percent']:.2f}%"
                       if item["relative_change_percent"] is not None else "unavailable")
        lines.append(
            f"| Q{item['mode']} | {item['G25OR'].get('mean', float('nan')):.2f} | "
            f"{item['ISR'].get('mean', float('nan')):.2f} | "
            f"{change_text} | {ci_text} | "
            f"{item['G25OR'].get('p95', float('nan')):.2f} | "
            f"{item['ISR'].get('p95', float('nan')):.2f} | {item['decision']} |"
        )
    lines.extend(["", f"Frozen adoption rule: {output['adoption_rule']}", ""])
    if exclusions:
        lines.extend([
            "## Excluded or failed attempts", "",
            "Every attempt remains in the ledger. The presence of an exclusion prevents the "
            "correctness gate for its problem from passing.", "",
        ])
        lines.extend(f"- `{x.get('run_id')}`: {x.get('status')} {x.get('error') or ''}" for x in output["exclusions"])
        lines.append("")
    (root / "official_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    with (root / "official_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "run_id", "block", "block_position", "mode", "variant", "db_id",
            "practice_run_no", "case_code", "jammer_count", "cleared_jammer_count",
            "virtual_time_s", "program_run_duration_ms", "measure_accepted_count",
            "channel_switch_count", "clear_failure_count", "end_reason", "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(attempts)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
