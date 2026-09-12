from scripts.auto_official_isr_experiment import make_schedule


def test_q4_isrv2_only_schedule_has_exact_requested_count():
    schedule = make_schedule(
        smoke_runs=0,
        main_runs=30,
        seed=20260913,
        variants=["ISRV2"],
        modes=[4],
    )
    assert len(schedule) == 30
    assert {job["mode"] for job in schedule} == {4}
    assert {job["variant"] for job in schedule} == {"ISRV2"}
    assert [job["block"] for job in schedule] == list(range(1, 31))
    assert len({job["run_id"] for job in schedule}) == 30


def test_default_schedule_remains_backward_compatible():
    schedule = make_schedule(smoke_runs=1, main_runs=1, seed=7)
    assert len(schedule) == 8
    assert {(job["mode"], job["variant"]) for job in schedule} == {
        (3, "G25OR"), (3, "ISR"), (4, "G25OR"), (4, "ISR")
    }
