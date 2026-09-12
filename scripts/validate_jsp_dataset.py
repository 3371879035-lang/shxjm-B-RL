"""Validate completeness, uniqueness and cost evidence for a JSP dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    root = Path(args.dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    branches = [json.loads(line) for line in (root / "branches.jsonl").open(encoding="utf-8")]
    labels = [json.loads(line) for line in (root / "labels.jsonl").open(encoding="utf-8")]
    key_fields = ("mode", "split", "seed", "trajectory", "snapshot_index", "candidate_index")
    branch_keys = [tuple(row[field] for field in key_fields) for row in branches]
    label_keys = [tuple(row[field] for field in key_fields) for row in labels]
    expected_cases = sum(int(manifest["split_seeds"][name][1])
                         for name in manifest["splits"]) * len(manifest["modes"]) * 2
    features = np.asarray([row["features"] for row in labels], dtype=float)
    groups = {}
    for row in labels:
        key = tuple(row[field] for field in key_fields[:-1])
        groups.setdefault(key, []).append(row)
    baseline_errors = []
    for key, rows in groups.items():
        baseline = [row for row in rows if row["baseline"]]
        if len(baseline) != 1:
            baseline_errors.append({"key": key, "baseline_rows": len(baseline)})
        elif abs(float(baseline[0]["delta_to_baseline_s"])) > 1e-9:
            baseline_errors.append({"key": key, "baseline_delta": baseline[0]["delta_to_baseline_s"]})
    seed_errors = []
    for mode in manifest["modes"]:
        for split in manifest["splits"]:
            start, count = manifest["split_seeds"][split]
            expected = set(range(int(start), int(start) + int(count)))
            for trajectory in ("JSPBASE", "JSPG"):
                actual = {int(row["seed"]) for row in labels
                          if int(row["mode"]) == int(mode) and row["split"] == split
                          and row["trajectory"] == trajectory}
                if actual != expected:
                    seed_errors.append({"mode": mode, "split": split,
                                        "trajectory": trajectory,
                                        "missing": sorted(expected - actual)[:20],
                                        "unexpected": sorted(actual - expected)[:20]})
    checks = {
        "progress_complete": progress.get("state") == "complete",
        "expected_case_count": int(progress.get("completed_cases", -1)) == expected_cases,
        "branch_label_count_equal": len(branches) == len(labels),
        "branch_keys_unique": len(branch_keys) == len(set(branch_keys)),
        "label_keys_unique": len(label_keys) == len(set(label_keys)),
        "branch_label_keys_equal": set(branch_keys) == set(label_keys),
        "all_success": all(row.get("success") is True for row in branches),
        "all_cost_residuals_valid": all(abs(float(row["cost_identity_error_s"])) <= 1e-6
                                           for row in branches),
        "feature_version_consistent": all(
            row["feature_version"] == manifest["feature_version"] for row in labels),
        "features_finite": bool(np.all(np.isfinite(features))),
        "one_zero_baseline_per_snapshot": not baseline_errors,
        "seed_coverage_exact": not seed_errors,
    }
    output = {
        "dataset_dir": str(root.resolve()), "expected_cases": expected_cases,
        "reported_cases": progress.get("completed_cases"), "branches": len(branches),
        "labels": len(labels), "feature_count": int(features.shape[1]),
        "max_cost_identity_error_s": max(abs(float(row["cost_identity_error_s"]))
                                          for row in branches),
        "baseline_errors": baseline_errors[:20], "seed_errors": seed_errors,
        "checks": checks,
        "valid": all(checks.values()),
    }
    (root / "validation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
