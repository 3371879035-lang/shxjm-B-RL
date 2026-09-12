from __future__ import annotations

import json
import sys

from scripts import analyze_official_isr


def test_incomplete_official_analysis_reports_keep_baseline(tmp_path, monkeypatch):
    attempts = []
    for mode in (3, 4):
        for variant, virtual_time in (("G25OR", 1000.0), ("ISR", 900.0)):
            attempts.append({
                "run_id": f"q{mode}-{variant}", "phase": "main", "block": 1,
                "block_position": 1, "mode": mode, "variant": variant,
                "status": "verified_full_clear", "db_match_status": "verified",
                "db_id": len(attempts) + 1, "practice_run_no": 1, "case_code": "test",
                "jammer_count": 10, "cleared_jammer_count": 10,
                "virtual_time_s": virtual_time, "program_run_duration_ms": 100,
                "measure_accepted_count": 1, "channel_switch_count": 0,
                "clear_failure_count": 0, "end_reason": "user_exit",
            })
    (tmp_path / "progress.json").write_text(
        json.dumps({"state": "stopped", "attempts": attempts}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["analyze_official_isr.py", str(tmp_path)])
    analyze_official_isr.main()
    result = json.loads((tmp_path / "official_analysis.json").read_text(encoding="utf-8"))
    assert [item["decision"] for item in result["comparisons"]] == ["KEEP_G25OR", "KEEP_G25OR"]
    assert (tmp_path / "official_analysis.md").exists()
    assert (tmp_path / "official_runs.csv").exists()
