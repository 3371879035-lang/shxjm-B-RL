from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.run_cvr_experiments import MECHANISM_FIELDS, _complete_rows, summarize


def write_analysis(rows: list[dict], out: Path, stage: str) -> None:
    rows = _complete_rows(rows)
    summary = summarize(rows, stage)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    decompositions = []
    mechanisms = []
    for mode in sorted({int(row["mode"]) for row in rows}):
        for arm in sorted({row["arm"] for row in rows if int(row["mode"]) == mode}):
            group = [row for row in rows if int(row["mode"]) == mode and row["arm"] == arm]
            if not group:
                continue
            per_source = {
                "move_s_per_source": np.mean([float(row["distance_m"]) / 5.0 / int(row["sources"]) for row in group]),
                "measure_s_per_source": np.mean([5.0 * float(row["measure_calls"]) / int(row["sources"]) for row in group]),
                "switch_s_per_source": np.mean([float(row["switches"]) / int(row["sources"]) for row in group]),
                "failed_clear_s_per_source": np.mean([3.0 * float(row["failed_clear"]) / int(row["sources"]) for row in group]),
                "success_clear_s_per_source": np.mean([5.0 * float(row["cleared"]) / int(row["sources"]) for row in group]),
            }
            decompositions.append({"mode": mode, "arm": arm, **{key: float(value) for key, value in per_source.items()}})
            mechanisms.append({
                "mode": mode,
                "arm": arm,
                **{
                    field: float(np.mean([float(row.get(field, 0) or 0) for row in group]))
                    for field in MECHANISM_FIELDS
                    if field != "fallback_reason"
                },
            })
    (out / "cost_decomposition.json").write_text(json.dumps(decompositions, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "mechanism.json").write_text(json.dumps(mechanisms, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["dev", "acceptance"], default="dev")
    args = parser.parse_args()
    with args.runs.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_analysis(rows, args.out_dir, args.stage)


if __name__ == "__main__":
    main()
