"""Explain paired JSP changes by official virtual-time cost components."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    args = parser.parse_args()
    root = Path(args.experiment_dir)
    with (root / "runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    decisions = []
    path = root / "decisions.jsonl"
    if path.exists():
        decisions = [json.loads(line) for line in path.open(encoding="utf-8")]
    output = {"experiment_dir": str(root.resolve()), "comparisons": []}
    for mode in sorted({int(row["mode"]) for row in rows}):
        for error_mode in sorted({row["error_mode"] for row in rows if int(row["mode"]) == mode}):
            group = [row for row in rows if int(row["mode"]) == mode and row["error_mode"] == error_mode]
            baseline = {int(row["seed"]): row for row in group if row["arm"] == "ISR"}
            for arm in sorted({row["arm"] for row in group if row["arm"] != "ISR"}):
                candidate = {int(row["seed"]): row for row in group if row["arm"] == arm}
                seeds = sorted(set(baseline) & set(candidate))
                def mean_delta(field, multiplier=1.0):
                    return float(np.mean([
                        multiplier * (float(candidate[seed][field]) - float(baseline[seed][field])) /
                        float(baseline[seed]["sources"]) for seed in seeds]))
                events = [row for row in decisions if int(row["mode"]) == mode and row["arm"] == arm
                          and int(row["seed"]) in seeds]
                reasons = collections.Counter(row.get("selection", {}).get("reason", "unknown")
                                              for row in events)
                kinds = collections.Counter(row.get("kind", "unknown") for row in events)
                components = {
                    "movement_seconds_per_source": mean_delta("distance_m", .2),
                    "measurement_seconds_per_source": mean_delta("measure_calls", 5.0),
                    "switch_seconds_per_source": mean_delta("switches", 1.0),
                    "failed_clear_seconds_per_source": mean_delta("failed_clear", 3.0),
                    "successful_clear_seconds_per_source": 0.0,
                }
                output["comparisons"].append({
                    "mode": mode, "error_mode": error_mode, "candidate": arm,
                    "pairs": len(seeds), "component_delta_candidate_minus_isr": components,
                    "component_sum_seconds_per_source": float(sum(components.values())),
                    "observed_total_delta_seconds_per_source": float(np.mean([
                        float(candidate[seed]["seconds_per_source"]) -
                        float(baseline[seed]["seconds_per_source"]) for seed in seeds])),
                    "decision_count": len(events), "selection_reasons": dict(reasons),
                    "chosen_action_types": dict(kinds),
                })
    (root / "cost_decomposition.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
