import numpy as np
import pytest

from brl.cvr import CVRCandidate, CVRConfig
from brl.local_env import RadioEnv, random_sources
from brl.protocol import ActionIOError
from brl.independent_candidate import IndependentCandidate


@pytest.mark.parametrize("mode,count", [(3, 10), (3, 16), (4, 10), (4, 16)])
def test_cvr_clears_all_sources_and_balances_cost(mode, count):
    seed = 7000 + mode * 100 + count
    sources = random_sources(mode, count, np.random.default_rng(seed))
    env = RadioEnv(mode=mode, n_sources=count, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    result = CVRCandidate(mode).run(env)
    assert result["success"]
    assert env.cleared_count() == count
    reconstructed = (
        env.move_distance / 5.0
        + 5.0 * env.n_measure
        + env.n_switch
        + 3.0 * env.n_clear_fail
        + 5.0 * env.cleared_count()
    )
    assert abs(env.virtual_time - reconstructed) <= 1e-6


def test_variable_measurement_never_claims_a_legacy_coverage_index(monkeypatch):
    env = RadioEnv(mode=3, n_sources=10, seed=7100, bearing_decimals=2)
    seen = []
    original = env.measure

    def capture(position, channel, coverage_idx=None, is_refine=False):
        seen.append((tuple(position), coverage_idx))
        return original(position, channel, coverage_idx, is_refine)

    monkeypatch.setattr(env, "measure", capture)
    CVRCandidate(3).run(env)
    assert all(
        index is None
        or np.linalg.norm(np.asarray(position) - env.coverage_points[index]) <= 1e-6
        for position, index in seen
    )


def test_transport_error_stops_before_a_different_action(monkeypatch):
    env = RadioEnv(mode=4, n_sources=10, seed=7200, bearing_decimals=2)
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise ActionIOError("response uncertain")

    monkeypatch.setattr(env, "measure", fail)
    with pytest.raises(ActionIOError):
        CVRCandidate(4).run(env)
    assert len(calls) == 1


def test_zero_budget_falls_back_to_isr_and_clears():
    env = RadioEnv(mode=4, n_sources=10, seed=7300, bearing_decimals=2)
    config = CVRConfig(max_plan_call_s=0.0, max_plan_total_s=0.0)
    result = CVRCandidate(4, config).run(env)
    assert result["success"] and env.cleared_count() == 10
    assert result["fallback_reason"] in {"per_call_budget", "episode_budget"}


@pytest.mark.parametrize("mode,seed", [(3, 160000), (3, 160001), (4, 160000), (4, 160001)])
def test_planner_disabled_executor_matches_frozen_isr_action_by_action(mode, seed):
    sources = random_sources(mode, 10, np.random.default_rng(seed))
    traces = []
    results = []
    for candidate in (
        IndependentCandidate(mode),
        CVRCandidate(mode, CVRConfig(planner_enabled=False)),
    ):
        env = RadioEnv(mode=mode, n_sources=10, seed=seed, bearing_decimals=2)
        env.reset(seed=seed, sources=sources)
        trace = []
        measure, clear = env.measure, env.clear
        def traced_measure(position, channel, coverage_idx=None, is_refine=False):
            response = measure(position, channel, coverage_idx=coverage_idx, is_refine=is_refine)
            trace.append(("measure", int(channel), tuple(map(float, position)),
                          coverage_idx, bool(is_refine), response.get("accepted"),
                          response.get("measure_result"), response.get("svd_deg")))
            return response
        def traced_clear(position, channel):
            response = clear(position, channel)
            trace.append(("clear", int(channel), tuple(map(float, position)),
                          None, False, response.get("accepted"),
                          response.get("clear_result")))
            return response
        env.measure = traced_measure
        env.clear = traced_clear
        results.append(candidate.run(env))
        traces.append(trace)
    assert len(traces[0]) == len(traces[1])
    for baseline, compatible in zip(*traces):
        assert baseline[:2] == compatible[:2]
        assert np.linalg.norm(np.asarray(baseline[2]) - np.asarray(compatible[2])) <= 1e-8
        assert baseline[3:] == compatible[3:]
    assert abs(results[0]["virtual_time_s"] - results[1]["virtual_time_s"]) <= 1e-6
