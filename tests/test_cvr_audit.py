import json
from pathlib import Path

from scripts.audit_cvr_certificates import audit_events
from brl.cvr.policy import CVRPolicy
from brl.independent_candidate import CheckedView
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
    assert "future evidence" in result["errors"][0]


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
    policy = CVRPolicy(CheckedView(env))
    policy.run()
    path = tmp_path / "real.jsonl"
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in policy.events), encoding="utf-8"
    )
    result = audit_events(path)
    assert result["valid"], result["errors"]
