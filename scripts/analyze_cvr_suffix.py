"""Summarize the already-recorded CVR suffix audit without rerunning branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows: list[dict]) -> dict:
    groups = []
    for mode in sorted({int(row["mode"]) for row in rows}):
        subset = [row for row in rows if int(row["mode"]) == mode]
        replacements = [row for row in subset
                        if row["segment_reason"] == "certified_replacement"]
        delta = np.asarray([row["actual_delta_s"] for row in subset], dtype=float)
        predicted = np.asarray([row["predicted_delta_s"] for row in subset], dtype=float)
        groups.append({
            "mode": mode, "snapshots": len(subset), "suffix_branches": 2 * len(subset),
            "all_complete": all(row["both_complete"] for row in subset),
            "mean_actual_delta_s": float(delta.mean()),
            "median_actual_delta_s": float(np.median(delta)),
            "predicted_actual_correlation": (float(np.corrcoef(predicted, delta)[0, 1])
                                             if predicted.std() and delta.std() else None),
            "certified_replacement_snapshots": len(replacements),
            "replacement_actual_win_rate": (sum(row["actual_delta_s"] < 0 for row in replacements)
                                            / len(replacements) if replacements else None),
            "replacement_mean_actual_delta_s": (float(np.mean([row["actual_delta_s"] for row in replacements]))
                                                 if replacements else None),
            "replacement_min_actual_delta_s": (min((row["actual_delta_s"] for row in replacements), default=None)),
            "replacement_max_actual_delta_s": (max((row["actual_delta_s"] for row in replacements), default=None)),
            "replacement_mean_distance_delta_m": (float(np.mean([row["actual_distance_delta_m"] for row in replacements]))
                                                   if replacements else None),
        })
    return {
        "schema_version": 1, "snapshots": len(rows),
        "suffix_branches": 2 * len(rows),
        "all_complete": all(row["both_complete"] for row in rows),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suffixes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.suffixes.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = summarize(rows)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
