import json
from pathlib import Path

from scripts.audit_cvr_certificates import audit_events
from brl.cvr import CVRCandidate
from brl.local_env import RadioEnv


def test_auditor_rejects_future_point_as_completed_evidence(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "certified_absent",
                "channel": 1,
                "accepted_negative_points": [],
                "future_points": [[0.0, 0.0]],
                "mode": 3,
                "plan_version": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_events(path)
    assert not result["valid"]
    assert any("future evidence" in error for error in result["errors"])


def test_auditor_rejects_duplicate_requests(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "event": "measurement",
            "request_id": "same",
            "channel": 1,
            "position": [0.0, 0.0],
            "legacy_station": None,
            "result": "no_signal",
            "accepted": True,
            "plan_version": 1,
        },
        {
            "event": "measurement",
            "request_id": "same",
            "channel": 1,
            "position": [1.0, 1.0],
            "legacy_station": None,
            "result": "no_signal",
            "accepted": True,
            "plan_version": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = audit_events(path)
    assert not result["valid"]
    assert any("duplicate request" in error for error in result["errors"])


def test_auditor_replays_a_real_cvr_policy_log(tmp_path: Path):
    env = RadioEnv(mode=3, n_sources=10, seed=8123, bearing_decimals=2)
    path = tmp_path / "real.jsonl"
    CVRCandidate(3).run(env, journal_path=path)
    result = audit_events(path)
    assert result["valid"], result["errors"]


def test_auditor_rejects_empty_initial_only_and_truncated_logs(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert not audit_events(empty)["valid"]

    initial = tmp_path / "initial.jsonl"
    initial.write_text(json.dumps({
        "sequence": 1, "event": "plan_initial", "mode": 3,
        "plan_version": 1, "nodes": [],
    }) + "\n", encoding="utf-8")
    result = audit_events(initial)
    assert not result["valid"]
    assert any("terminal" in error for error in result["errors"])


def test_auditor_rejects_plan_node_coordinate_spoof(tmp_path: Path):
    path = tmp_path / "spoof.jsonl"
    rows = [
        {"sequence": 1, "event": "plan_initial", "mode": 3, "plan_version": 1,
         "nodes": [{"node_id": "v", "position": [0.0, 0.0], "channels": [1],
                    "legacy_station": None}]},
        {"sequence": 2, "event": "measurement", "request_id": "semantic-1",
         "node_id": "v", "channel": 1, "position": [1.0, 0.0],
         "legacy_station": None, "result": "no_signal", "accepted": True,
         "plan_version": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = audit_events(path)
    assert not result["valid"]
    assert any("coordinate differs" in error for error in result["errors"])
