from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.coverage import s25_points, s3_points
from brl.cvr.certify import CoverageCertificateEngine


def _iter_event_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.rglob("*.jsonl"))


def audit_events(path: Path) -> dict:
    errors: list[str] = []
    last_version = 0
    accepted: dict[int, list[tuple[float, float]]] = {}
    request_ids: set[str] = set()
    nodes: dict[str, dict] = {}
    mode: int | None = None

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid json: {exc}")
            continue
        event = row.get("event")
        version = int(row.get("plan_version", row.get("plan_version_before", 0)))
        if version and version < last_version:
            errors.append(f"line {line_number}: plan version moved backward")

        if event == "plan_initial":
            if nodes:
                errors.append(f"line {line_number}: duplicate initial plan")
                continue
            mode = int(row["mode"])
            fixed = s3_points() if mode == 3 else s25_points()
            for serialized in row.get("nodes", []):
                node_id = str(serialized["node_id"])
                if node_id in nodes:
                    errors.append(f"line {line_number}: duplicate node id {node_id}")
                    continue
                station = serialized.get("legacy_station")
                position = tuple(map(float, serialized["position"]))
                if station is not None:
                    if not 0 <= int(station) < len(fixed):
                        errors.append(f"line {line_number}: invalid legacy station {station}")
                    elif float(np.linalg.norm(np.asarray(position) - fixed[int(station)])) > 1e-6:
                        errors.append(f"line {line_number}: legacy coordinate/index mismatch")
                nodes[node_id] = {
                    "position": position,
                    "channels": set(map(int, serialized.get("channels", []))),
                    "legacy_station": station,
                }
            last_version = max(last_version, version)
            continue

        if event == "measurement":
            request_id = str(row.get("request_id", ""))
            if not request_id or request_id in request_ids:
                errors.append(f"line {line_number}: missing or duplicate request id")
            request_ids.add(request_id)
            channel = int(row["channel"])
            result = row.get("result")
            if row.get("accepted") is True and result == "no_signal":
                accepted.setdefault(channel, []).append(tuple(map(float, row["position"])))
            node_id = str(row.get("node_id", ""))
            if node_id:
                if node_id not in nodes:
                    errors.append(f"line {line_number}: measurement names a non-pending node")
                else:
                    nodes[node_id]["channels"].discard(channel)
                    if not nodes[node_id]["channels"]:
                        nodes.pop(node_id)
            if result in {"direction", "near"}:
                for pending in nodes.values():
                    pending["channels"].discard(channel)
                nodes = {
                    node_key: pending
                    for node_key, pending in nodes.items()
                    if pending["channels"]
                }
            station = row.get("legacy_station")
            if station is not None and mode is not None:
                fixed = s3_points() if mode == 3 else s25_points()
                position = np.asarray(row["position"], dtype=float)
                if not 0 <= int(station) < len(fixed) or np.linalg.norm(position - fixed[int(station)]) > 1e-6:
                    errors.append(f"line {line_number}: measurement legacy coordinate/index mismatch")

        elif event == "plan_mutation":
            before = int(row.get("plan_version_before", -1))
            after = int(row.get("plan_version", -1))
            if before != last_version or after != before + 1:
                errors.append(f"line {line_number}: mutation versions are not consecutive")
            removed = list(map(str, row.get("removed", [])))
            if not removed or any(node_id not in nodes for node_id in removed):
                errors.append(f"line {line_number}: mutation removed a non-pending node")
            for node_id in removed:
                nodes.pop(node_id, None)
            for serialized in row.get("added", []):
                node_id = str(serialized["node_id"])
                if node_id in nodes:
                    errors.append(f"line {line_number}: mutation reused node id {node_id}")
                    continue
                nodes[node_id] = {
                    "position": tuple(map(float, serialized["position"])),
                    "channels": set(map(int, serialized.get("channels", []))),
                    "legacy_station": None,
                }
            if mode is not None:
                engine = CoverageCertificateEngine(mode)
                for certificate in row.get("channel_certificates", []):
                    channel = int(certificate["channel"])
                    points = tuple(accepted.get(channel, ())) + tuple(
                        node["position"] for node in nodes.values() if channel in node["channels"]
                    )
                    replay = engine.certify(points)
                    if not replay.covered:
                        errors.append(f"line {line_number}: uncertified plan mutation for channel {channel}")
            last_version = max(last_version, after)

        elif event == "certified_absent":
            if row.get("future_points"):
                errors.append(f"line {line_number}: future evidence used for completed absence")
                continue
            channel = int(row["channel"])
            serialized = tuple(tuple(map(float, point)) for point in row.get("accepted_negative_points", []))
            if serialized != tuple(accepted.get(channel, ())):
                errors.append(f"line {line_number}: absence uses unaccepted or wrong-channel evidence")
                continue
            event_mode = int(row.get("mode", mode or 0))
            replay = CoverageCertificateEngine(event_mode).certify(serialized)
            if not replay.covered:
                errors.append(f"line {line_number}: absence certificate does not replay")
            for pending in nodes.values():
                pending["channels"].discard(channel)
            nodes = {
                node_key: pending
                for node_key, pending in nodes.items()
                if pending["channels"]
            }

        last_version = max(last_version, version)

    return {
        "valid": not errors,
        "errors": errors,
        "last_plan_version": last_version,
        "accepted_request_count": len(request_ids),
    }


def audit_path(path: Path) -> dict:
    files = list(_iter_event_files(path))
    results = {str(file): audit_events(file) for file in files}
    return {
        "valid": bool(files) and all(result["valid"] for result in results.values()),
        "files": results,
        "file_count": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    result = audit_path(args.events)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
