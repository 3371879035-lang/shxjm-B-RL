from __future__ import annotations

import numpy as np
import pytest

import brl.independent_candidate as candidate_module
from brl.independent_candidate import Baseline, IndependentCandidate, JointShell
from brl.local_env import RadioEnv, Source, random_sources
from brl.protocol import ActionIOError


@pytest.mark.parametrize("mode", [3, 4])
@pytest.mark.parametrize("count", [10, 16])
def test_candidate_clears_minimum_and_maximum_source_counts(mode, count):
    sources = random_sources(mode, count, np.random.default_rng(1000 + mode + count))
    env = RadioEnv(mode=mode, n_sources=count, seed=2000 + mode, bearing_decimals=2)
    env.reset(seed=2000 + mode, n_sources=count, sources=sources)
    result = IndependentCandidate(mode).run(env)
    assert result["success"]
    assert env.completion_certificate()
    assert env.cleared_count() == count


def test_near_at_origin_is_cleared_during_initial_scan():
    sources = [Source(1, np.array([3.0, 0.0]), 1000.0)]
    sources.extend(
        Source(ch, np.array([800.0 + ch, 20.0 * ch]), 1500.0)
        for ch in range(2, 11)
    )
    env = RadioEnv(mode=3, n_sources=10, seed=31, bearing_decimals=2)
    env.reset(seed=31, n_sources=10, sources=sources)
    IndependentCandidate(3).run(env)
    assert env.channels[1].status == "cleared"


def test_clear_distance_boundary_is_inclusive():
    source = Source(1, np.array([20.0, 0.0]), 1500.0)
    env = RadioEnv(mode=3, n_sources=1, seed=4)
    env.reset(seed=4, n_sources=1, sources=[source])
    assert env.clear([0.0, 0.0], 1)["clear_result"] == "success"


def test_q4_no_signal_does_not_remove_discovered_region():
    source = Source(1, np.array([500.0, 0.0]), 1000.0, "directional", 0.0)
    env = RadioEnv(mode=4, n_sources=1, seed=5, bearing_decimals=2)
    env.reset(seed=5, n_sources=1, sources=[source])
    first = env.measure([1000.0, 0.0], 1, is_refine=True)
    assert first["measure_result"] == "direction"
    before = env.channels[1].poly.copy()
    second = env.measure([0.0, 0.0], 1, is_refine=True)
    assert second["measure_result"] == "no_signal"
    assert np.array_equal(env.channels[1].poly, before)
    assert env.channels[1].status == "discovered"


def test_fast_budget_zero_falls_back_to_certified_path():
    sources = random_sources(3, 10, np.random.default_rng(73))
    env = RadioEnv(mode=3, n_sources=10, seed=73, bearing_decimals=2)
    env.reset(seed=73, n_sources=10, sources=sources)
    policy = JointShell(env, max_fast_steps=0)
    policy.max_region_radius = 800.0
    policy.run()
    assert env.completion_certificate()
    assert env.cleared_count() == 10


def test_failed_center_probe_continues_to_bilateral(monkeypatch):
    env = RadioEnv(mode=3, n_sources=1, seed=8)
    policy = Baseline(env, try_clear=True)
    env.channels[1].status = "discovered"
    policy.tracks[1] = {
        "first": np.zeros(2), "deg": 0.0,
        "P": np.array([[0.0, 0.0], [60.0, 0.0], [30.0, 50.0]]), "nobs": 1,
    }
    calls = {"clear": 0, "solve": 0}

    def clear(position, channel):
        calls["clear"] += 1
        return {"accepted": True, "clear_result": "no_target_in_range"}

    def solve(*args, **kwargs):
        calls["solve"] += 1
        env.channels[1].status = "cleared"

    monkeypatch.setattr(env, "clear", clear)
    monkeypatch.setattr(candidate_module, "solve_bilateral", solve)
    policy.resolve(1)
    assert calls == {"clear": 1, "solve": 1}


def test_bilateral_transport_failure_does_not_trigger_optical_actions(monkeypatch):
    env = RadioEnv(mode=3, n_sources=1, seed=81)
    policy = Baseline(env, try_clear=False)
    env.channels[1].status = "discovered"
    policy.tracks[1] = {
        "first": np.zeros(2), "deg": 0.0,
        "P": np.array([[0.0, 0.0], [60.0, 0.0], [30.0, 50.0]]), "nobs": 1,
    }
    calls = {"clear": 0}

    def clear(position, channel):
        calls["clear"] += 1
        return {"accepted": True, "clear_result": "no_target_in_range"}

    def solve(*args, **kwargs):
        raise ActionIOError("bilateral response lost")

    monkeypatch.setattr(env, "clear", clear)
    monkeypatch.setattr(candidate_module, "solve_bilateral", solve)
    with pytest.raises(ActionIOError, match="bilateral response lost"):
        policy.resolve(1)
    assert calls["clear"] == 0
    assert policy.solver_failures == 0


def test_transport_failure_after_failed_probe_stops_before_fallback(monkeypatch):
    env = RadioEnv(mode=3, n_sources=1, seed=82)
    policy = Baseline(env, try_clear=True)
    env.channels[1].status = "discovered"
    policy.tracks[1] = {
        "first": np.zeros(2), "deg": 0.0,
        "P": np.array([[0.0, 0.0], [60.0, 0.0], [30.0, 50.0]]), "nobs": 1,
    }
    calls = {"clear": 0}

    def clear(position, channel):
        calls["clear"] += 1
        return {"accepted": True, "clear_result": "no_target_in_range"}

    def solve(*args, **kwargs):
        raise ActionIOError("refine timeout")

    monkeypatch.setattr(env, "clear", clear)
    monkeypatch.setattr(candidate_module, "solve_bilateral", solve)
    with pytest.raises(ActionIOError, match="refine timeout"):
        policy.resolve(1)
    assert calls["clear"] == 1
    assert policy.solver_failures == 0


@pytest.mark.parametrize("policy_factory", [lambda env: IndependentCandidate(3), lambda env: None])
def test_transport_failure_is_not_converted_to_optical_fallback(monkeypatch, policy_factory):
    sources = random_sources(3, 10, np.random.default_rng(11))
    env = RadioEnv(mode=3, n_sources=10, seed=11)
    env.reset(seed=11, n_sources=10, sources=sources)
    monkeypatch.setattr(env, "measure", lambda *args, **kwargs: (_ for _ in ()).throw(ActionIOError("lost")))
    if policy_factory(env) is None:
        from brl.g25o import G25OPolicy
        run = lambda: G25OPolicy("G25OR", coverage="S25").run(env)
    else:
        run = lambda: policy_factory(env).run(env)
    with pytest.raises(ActionIOError):
        run()
