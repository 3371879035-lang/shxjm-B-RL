from scripts.auto_official_isr_experiment import make_schedule


def test_schedule_has_three_smoke_and_one_hundred_main_runs_per_group():
    schedule = make_schedule(3, 100, 20260912)
    assert len(schedule) == 412
    for phase, count in (("smoke", 3), ("main", 100)):
        selected = [job for job in schedule if job["phase"] == phase]
        for mode in (3, 4):
            for variant in ("G25OR", "ISR"):
                assert sum(job["mode"] == mode and job["variant"] == variant for job in selected) == count
    assert len({job["run_id"] for job in schedule}) == len(schedule)


def test_schedule_is_seed_deterministic():
    assert make_schedule(3, 100, 20260912) == make_schedule(3, 100, 20260912)
    assert make_schedule(3, 100, 20260912) != make_schedule(3, 100, 20260913)
