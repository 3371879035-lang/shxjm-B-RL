"""Tie completed experiment rows to their action journals and audit denominators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cvr_certificates import audit_events


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    with (run_dir / "runs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row.get("status") == "complete"]
    started = [row for row in rows if row.get("status") == "started"]
    failed = [row for row in rows if row.get("status") == "failed"]
    audits = []
    linked: set[Path] = set()
    for row in complete:
        stem = f"q{row['mode']}_{row['seed']}_{row['error_mode']}_{row['arm']}"
        candidates = sorted((run_dir / "decisions").glob(stem + "*.jsonl"))
        valid_candidates = []
        for path in candidates:
            result = audit_events(path)
            if result["valid"]:
                valid_candidates.append((path, result))
        if len(valid_candidates) == 1:
            path, result = valid_candidates[0]
            linked.add(path.resolve())
        else:
            path = candidates[0] if candidates else None
            result = {"valid": False,
                      "errors": [f"expected one valid journal, found {len(valid_candidates)}"]}
        truth_ok = (str(row.get("success", "")).lower() == "true"
                    and int(row["sources"]) == int(row["cleared"])
                    and abs(float(row["cost_identity_error_s"])) <= 1e-6)
        audits.append({
            "mode": int(row["mode"]), "seed": int(row["seed"]), "arm": row["arm"],
            "journal": None if path is None else str(path.relative_to(run_dir)),
            "journal_sha256": None if path is None else _hash(path),
            "journal_valid": bool(result["valid"]), "journal_errors": result["errors"],
            "offline_truth_and_cost_ok": truth_ok,
        })
    all_files = sorted((run_dir / "decisions").glob("*.jsonl"))
    unlinked = []
    for path in all_files:
        if path.resolve() not in linked:
            result = audit_events(path)
            unlinked.append({"journal": str(path.relative_to(run_dir)),
                             "valid": result["valid"], "errors": result["errors"]})
    result = {
        "schema_version": 1,
        "csv_rows": len(rows), "started_rows": len(started),
        "complete_rows": len(complete), "failed_rows": len(failed),
        "completed_journals_valid": sum(item["journal_valid"] for item in audits),
        "completed_truth_and_cost_valid": sum(item["offline_truth_and_cost_ok"] for item in audits),
        "all_completed_valid": all(item["journal_valid"] and item["offline_truth_and_cost_ok"]
                                   for item in audits),
        "unlinked_or_incomplete_journals": unlinked,
        "completed": audits,
    }
    output = run_dir / "runset_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"completed", "unlinked_or_incomplete_journals"}},
                     ensure_ascii=False, indent=2))
    print(f"unlinked_or_incomplete_journals={len(unlinked)}")
    raise SystemExit(0 if result["all_completed_valid"] else 1)


if __name__ == "__main__":
    main()
