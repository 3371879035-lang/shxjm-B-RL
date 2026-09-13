"""Offline full-suffix audit of CVR decisions on existing development seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.cvr import CVRCandidate
from brl.cvr.policy import CVRPolicy
from brl.local_env import RadioEnv
from scripts.run_g25o_compare import generate_sources


def _metrics(policy: CVRPolicy) -> dict:
    env = policy.env
    return {
        "time_s": float(env.virtual_time), "distance_m": float(env.move_distance),
        "measure_calls": int(env.n_measure), "switches": int(env.n_switch),
        "clear_calls": int(env.n_clear), "failed_clear": int(env.n_clear_fail),
        "cleared": int(env.cleared_count()),
    }


def _delta(after: dict, before: dict) -> dict:
    return {key: after[key] - before[key] for key in (
        "time_s", "distance_m", "measure_calls", "switches", "clear_calls", "failed_clear")}


def _capture_count(mode: int, seed: int, kind: str) -> int:
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                   step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    count = 0
    def hook(*_):
        nonlocal count
        count += 1
    CVRCandidate(mode).run(env, decision_hook=hook)
    return count


def _audit_seed(mode: int, seed: int, kind: str) -> list[dict]:
    decision_count = _capture_count(mode, seed, kind)
    if decision_count == 0:
        return []
    selected_indices = set(map(int, np.linspace(0, decision_count - 1,
                                                min(4, decision_count)).round()))
    sources = generate_sources(seed, mode, kind)
    env = RadioEnv(mode=mode, n_sources=len(sources), seed=seed,
                   step_limit=20000, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    records = []
    index = 0

    def hook(policy, segment, hard_job, snapshot, exit_position):
        nonlocal index
        current = index
        index += 1
        if current not in selected_indices:
            return
        selected = policy.checkpoint()
        baseline = policy.checkpoint()
        before = _metrics(selected)

        selected._apply_segment(segment)
        selected.finish_current_plan()

        baseline_snapshot = baseline._public_snapshot()
        baseline_segment = baseline.planner._baseline(
            baseline_snapshot, baseline.ledger, tuple(map(float, exit_position)),
            "suffix_isr_compatible", forced_first_node=str(hard_job[1]))
        baseline._apply_segment(baseline_segment)
        baseline.finish_current_plan()

        selected_delta = _delta(_metrics(selected), before)
        baseline_delta = _delta(_metrics(baseline), before)
        records.append({
            "mode": mode, "seed": seed, "kind": kind,
            "decision_index": current, "decision_count": decision_count,
            "segment_reason": segment.reason,
            "predicted_delta_s": float(segment.estimated_cost_s - segment.baseline_cost_s),
            "selected_suffix": selected_delta, "baseline_suffix": baseline_delta,
            "actual_delta_s": selected_delta["time_s"] - baseline_delta["time_s"],
            "actual_distance_delta_m": selected_delta["distance_m"] - baseline_delta["distance_m"],
            "actual_measure_delta": selected_delta["measure_calls"] - baseline_delta["measure_calls"],
            "actual_switch_delta": selected_delta["switches"] - baseline_delta["switches"],
            "actual_clear_delta": selected_delta["clear_calls"] - baseline_delta["clear_calls"],
            "both_complete": selected._complete() and baseline._complete(),
        })

    CVRCandidate(mode).run(env, decision_hook=hook)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    path = args.out_dir / "suffixes.jsonl"
    for mode in (3, 4):
        for offset in range(10):
            seed = 160000 + offset
            kind = "edge" if offset % 3 == 0 else "uniform"
            for row in _audit_seed(mode, seed, kind):
                rows.append(row)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    actual = np.asarray([row["actual_delta_s"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_delta_s"] for row in rows], dtype=float)
    chosen = [row for row in rows if row["segment_reason"] == "certified_replacement"]
    summary = {
        "schema_version": 1, "scenes": 20, "suffix_branches": len(rows) * 2,
        "snapshots": len(rows), "all_complete": all(row["both_complete"] for row in rows),
        "mean_actual_delta_s": float(actual.mean()) if len(actual) else None,
        "median_actual_delta_s": float(np.median(actual)) if len(actual) else None,
        "predicted_actual_correlation": (float(np.corrcoef(predicted, actual)[0, 1])
                                         if len(rows) > 1 and predicted.std() and actual.std() else None),
        "certified_replacement_snapshots": len(chosen),
        "replacement_actual_win_rate": (sum(row["actual_delta_s"] < 0 for row in chosen) / len(chosen)
                                        if chosen else None),
        "replacement_mean_actual_delta_s": (float(np.mean([row["actual_delta_s"] for row in chosen]))
                                             if chosen else None),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
