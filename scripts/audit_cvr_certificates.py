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
    initialized = False
    terminal_rows: list[tuple[int, dict]] = []
    registered: dict[str, dict] = {}
    finished_requests: set[str] = set()
    confirmed_measurements: list[dict] = []
    last_time = -float("inf")
    last_sequence = 0
    saw_failed_action = False

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
        sequence = int(row.get("sequence", 0))
        if sequence:
            if sequence != last_sequence + 1:
                errors.append(f"line {line_number}: non-consecutive event sequence")
            last_sequence = sequence
        version = int(row.get("plan_version", row.get("plan_version_before", 0)))
        if version and version < last_version:
            errors.append(f"line {line_number}: plan version moved backward")

        if event == "plan_initial":
            if initialized:
                errors.append(f"line {line_number}: duplicate initial plan")
                continue
            initialized = True
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

        if not initialized:
            errors.append(f"line {line_number}: event before initial plan")

        if event == "action_registered":
            request_id = str(row.get("request_id", ""))
            if not request_id or request_id in registered:
                errors.append(f"line {line_number}: missing or duplicate registered request id")
            else:
                registered[request_id] = row
            continue

        if event in {"action_confirmed", "action_failed"}:
            if event == "action_failed":
                saw_failed_action = True
            request_id = str(row.get("request_id", ""))
            original = registered.get(request_id)
            if original is None or request_id in finished_requests:
                errors.append(f"line {line_number}: action completion lacks one pending registration")
                continue
            finished_requests.add(request_id)
            for key in ("action_kind", "channel", "position"):
                if row.get(key) != original.get(key):
                    errors.append(f"line {line_number}: action completion changed {key}")
            if event == "action_confirmed":
                response = row.get("response", {})
                if response.get("accepted") is not True:
                    errors.append(f"line {line_number}: unaccepted action was confirmed")
                before = row.get("before", {})
                after = row.get("after", {})
                try:
                    before_time = float(before["virtual_time_s"])
                    after_time = float(after["virtual_time_s"])
                    if not np.isfinite(before_time) or not np.isfinite(after_time):
                        errors.append(f"line {line_number}: non-finite action time")
                    if after_time + 1e-12 < before_time or before_time + 1e-12 < last_time:
                        errors.append(f"line {line_number}: action time moved backward")
                    last_time = after_time
                    for counter in ("distance_m", "measure_calls", "switches", "clear_calls", "failed_clear"):
                        if float(after[counter]) + 1e-12 < float(before[counter]):
                            errors.append(f"line {line_number}: counter {counter} moved backward")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"line {line_number}: incomplete action accounting")
                if row.get("action_kind") == "measure":
                    station = row.get("coverage_idx")
                    if station is not None and mode is not None:
                        fixed = s3_points() if mode == 3 else s25_points()
                        try:
                            index = int(station)
                            point = np.asarray(row.get("position"), dtype=float)
                            if not 0 <= index < len(fixed) or np.linalg.norm(point - fixed[index]) > 1e-8:
                                errors.append(f"line {line_number}: confirmed coverage index/coordinate mismatch")
                        except (TypeError, ValueError):
                            errors.append(f"line {line_number}: invalid confirmed coverage index")
                    confirmed_measurements.append(row)
            continue

        if event == "terminal":
            terminal_rows.append((line_number, row))
            continue

        if event == "measurement":
            request_id = str(row.get("request_id", ""))
            if not request_id or request_id in request_ids:
                errors.append(f"line {line_number}: missing or duplicate request id")
            request_ids.add(request_id)
            channel = int(row["channel"])
            result = row.get("result")
            position_tuple = tuple(map(float, row["position"]))
            matching = next((
                action for action in reversed(confirmed_measurements)
                if int(action.get("channel", -1)) == channel
                and np.linalg.norm(np.asarray(action.get("position"), dtype=float)
                                      - np.asarray(position_tuple)) <= 1e-8
                and action.get("response", {}).get("measure_result") == result
            ), None)
            if matching is None:
                errors.append(f"line {line_number}: coverage measurement lacks a matching confirmed action")
            if row.get("accepted") is True and result == "no_signal":
                accepted.setdefault(channel, []).append(position_tuple)
            node_id = str(row.get("node_id", ""))
            if node_id:
                if node_id not in nodes:
                    errors.append(f"line {line_number}: measurement names a non-pending node")
                elif row.get("accepted") is not True:
                    errors.append(f"line {line_number}: unaccepted measurement completed an obligation")
                else:
                    if np.linalg.norm(np.asarray(nodes[node_id]["position"]) - np.asarray(position_tuple)) > 1e-8:
                        errors.append(f"line {line_number}: measurement coordinate differs from plan node")
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

    if not initialized:
        errors.append("missing initial plan")
    if len(terminal_rows) != 1:
        errors.append("journal must contain exactly one terminal record")
    else:
        terminal_line, terminal = terminal_rows[0]
        last_content_line = max((number for number, value in enumerate(lines, 1) if value.strip()), default=0)
        if terminal_line != last_content_line:
            errors.append("terminal record must be the last event")
        if terminal.get("status") != "complete" or terminal.get("success") is not True \
                or terminal.get("completion_certificate") is not True:
            errors.append("terminal record does not confirm successful completion")
        try:
            if abs(float(terminal.get("virtual_time_s")) - last_time) > 1e-6:
                errors.append("terminal time differs from last confirmed action")
        except (TypeError, ValueError):
            errors.append("terminal record has invalid time")
    unfinished = set(registered) - finished_requests
    if unfinished:
        errors.append(f"unfinished registered actions: {sorted(unfinished)}")
    if saw_failed_action:
        errors.append("successful journal contains a failed action")

    return {
        "valid": not errors,
        "errors": errors,
        "last_plan_version": last_version,
        "accepted_request_count": len(request_ids),
        "confirmed_action_count": len(finished_requests),
        "terminal_count": len(terminal_rows),
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
