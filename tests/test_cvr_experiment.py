from scripts.run_cvr_experiments import jobs, summarize


def paired_fixture(
    baseline=100.0,
    candidate=95.0,
    pairs=50,
    physical_stations_removed=1,
    mean_distance_delta_m=-100.0,
):
    rows = []
    for seed in range(pairs):
        for arm, value, removed, distance in (
            ("ISR", baseline, 0, 1000.0),
            ("CVR", candidate, physical_stations_removed, 1000.0 + mean_distance_delta_m),
        ):
            rows.append(
                {
                    "mode": 4,
                    "seed": seed,
                    "kind": "uniform",
                    "error_mode": "spatial",
                    "arm": arm,
                    "sources": 10,
                    "cleared": 10,
                    "success": True,
                    "seconds_per_source": value,
                    "virtual_time_s": value * 10,
                    "distance_m": distance,
                    "measure_calls": 10,
                    "switches": 2,
                    "clear_calls": 10,
                    "failed_clear": 0,
                    "wall_s": 0.1,
                    "planning_wall_s": 0.01,
                    "cost_identity_error_s": 0.0,
                    "physical_stations_removed": removed,
                    "replacement_candidates": 1,
                    "certified_replacements": removed,
                    "executed_replacements": removed,
                    "restored_legacy_stations": 0,
                    "extra_measurements": 0,
                    "replans": 1,
                    "segment_interruptions": 0,
                    "certificate_wall_s": 0.01,
                    "fallback_reason": "",
                    "status": "complete",
                }
            )
    return rows


def test_development_schedule_has_exact_three_arms():
    rows = list(jobs("dev", [3, 4], ["ISR", "COMPAT", "CVR"]))
    assert len(rows) == 2 * 50 * 3
    assert {row[-1] for row in rows} == {"ISR", "COMPAT", "CVR"}


def test_candidate_below_three_percent_is_not_eligible():
    result = summarize(paired_fixture(baseline=100.0, candidate=98.0), "dev")
    assert result["development_selection"][0]["eligible"] == []


def test_mechanism_gate_requires_removed_station_or_lower_total_cost():
    rows = paired_fixture(
        baseline=100.0,
        candidate=95.0,
        physical_stations_removed=0,
        mean_distance_delta_m=20.0,
    )
    result = summarize(rows, "dev")
    assert not result["development_selection"][0]["checks"]["CVR"]["mechanism_gate"]


def test_started_record_is_ignored_when_complete_record_exists():
    rows = paired_fixture(pairs=2)
    started = dict(rows[0])
    started["status"] = "started"
    started["seconds_per_source"] = ""
    result = summarize([started, *rows], "dev")
    assert result["runs"] == 4
