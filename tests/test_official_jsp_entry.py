from __future__ import annotations

import sys
import sqlite3
import time
import json

import pytest

from scripts.auto_official_isr_experiment import db_marker, make_schedule, match_database_record
from scripts import run_official_g25o
from scripts import analyze_official_jsp30


def test_jsp_learned_model_is_required_before_enter(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_official_g25o.py", "--mode", "4", "--variant", "JSP",
        "--planner", "learned", "--coverage", "S25", "--out-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        run_official_g25o.main()
    assert list(tmp_path.iterdir()) == []


def test_jsp_rejects_non_s25_label_before_enter(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_official_g25o.py", "--mode", "4", "--variant", "JSP",
        "--coverage", "S21", "--out-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit):
        run_official_g25o.main()
    assert list(tmp_path.iterdir()) == []


def test_jsp30_schedule_has_fifteen_randomized_two_strategy_blocks():
    schedule = make_schedule(0, 15, 20260912, ["ISR", "JSP"], [4])
    assert len(schedule) == 30
    for block in range(1, 16):
        group = [row for row in schedule if row["block"] == block]
        assert len(group) == 2
        assert {row["variant"] for row in group} == {"ISR", "JSP"}
        assert {row["block_position"] for row in group} == {1, 2}


def test_database_match_uses_new_id_time_and_expected_sequence(tmp_path):
    path = tmp_path / "practice.sqlite3"
    columns = """id INTEGER, team_no TEXT, problem_no INTEGER, practice_run_no INTEGER,
        case_code TEXT, entered INTEGER, end_reason TEXT, cleared_jammer_count INTEGER,
        measure_accepted_count INTEGER, virtual_time_us INTEGER,
        program_run_duration_ms INTEGER, channel_switch_count INTEGER,
        clear_failure_count INTEGER, jammer_count INTEGER, state TEXT,
        created_at_ms INTEGER, updated_at_ms INTEGER"""
    now = int(time.time() * 1000)
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE practice_statistics_tasks ({columns})")
        connection.execute("INSERT INTO practice_statistics_tasks VALUES "
                           "(1,'team',4,5,'old',1,'done',10,1,1,1,1,0,10,'done',?,?)",
                           (now - 10000, now - 10000))
    assert db_marker(path, 4, "team") == (1, 5)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO practice_statistics_tasks VALUES "
                           "(2,'team',4,6,'new',1,'done',12,2,2,2,2,0,12,'done',?,?)",
                           (now, now))
    rows = match_database_record(path, 1, 4, "team", now, timeout_s=.1)
    assert len(rows) == 1 and rows[0]["practice_run_no"] == 6


def test_official_jsp30_analyzer_applies_frozen_gate(monkeypatch, tmp_path):
    attempts = []
    for block in range(1, 16):
        for variant, virtual in (("ISR", 1000.0), ("JSP", 900.0)):
            attempts.append({
                "run_id": f"{block}-{variant}", "phase": "main", "block": block,
                "block_position": 1 if variant == "ISR" else 2, "mode": 4,
                "variant": variant, "status": "verified_full_clear",
                "db_match_status": "verified", "jammer_count": 10,
                "cleared_jammer_count": 10, "virtual_time_s": virtual,
                "program_run_duration_ms": 1000, "measure_accepted_count": 20,
                "channel_switch_count": 5, "clear_failure_count": 0,
                "runner_summary": {"distance_m": 1000.0}, "db_id": block,
                "practice_run_no": block, "case_code": str(block), "end_reason": "done",
            })
    (tmp_path / "progress.json").write_text(json.dumps({"state": "complete", "attempts": attempts}))
    monkeypatch.setattr(sys, "argv", ["analyze_official_jsp30.py", str(tmp_path)])
    analyze_official_jsp30.main()
    output = json.loads((tmp_path / "official_analysis.json").read_text(encoding="utf-8"))
    assert output["adopt_jsp"] is True
    assert output["complete_time_blocks"] == 15
