from __future__ import annotations

import numpy as np

from brl.g25o import initial_track_polygon
from brl.isr_v2 import (ISRV2Candidate, ISRV2Config, ISRV2Policy,
                        SoftSourceBelief, deterministic_samples,
                        route_open_multistart)
from brl.local_env import RadioEnv, Source, random_sources


def test_deterministic_samples_do_not_depend_on_environment_seed():
    poly = initial_track_polygon(np.zeros(2), 15.25)
    a = deterministic_samples(poly, channel=7, limit=128)
    b = deterministic_samples(poly, channel=7, limit=128)
    assert len(a) == 128
    assert np.array_equal(a, b)


def test_failed_clear_only_changes_soft_belief():
    poly = initial_track_polygon(np.zeros(2), 0.0)
    belief = SoftSourceBelief(3, 1, poly)
    hard_before = poly.copy()
    q = belief.live_points()[0].copy()
    belief.update_failed_clear(q)
    assert np.array_equal(poly, hard_before)
    assert np.all(np.linalg.norm(belief.live_points() - q, axis=1) > 20.0)


def test_empty_soft_posterior_is_disabled_not_a_certificate():
    poly = initial_track_polygon(np.zeros(2), 0.0)
    belief = SoftSourceBelief(3, 1, poly)
    belief.update_failed_clear(np.mean(belief.live_points(), axis=0))
    belief.alive[:] = False
    belief._disable_if_empty()
    assert not belief.active
    assert belief.disabled_reason == "soft_posterior_empty"


def test_q4_back_facing_no_signal_reweights_without_deleting_positions():
    poly = initial_track_polygon(np.zeros(2), 0.0)
    belief = SoftSourceBelief(4, 2, poly)
    n_positions = len(belief.points)
    alive_before = np.count_nonzero(belief.alive)
    belief.update_no_signal([0.0, 0.0])
    assert len(belief.points) == n_positions
    assert np.count_nonzero(belief.alive) <= alive_before


def test_multistart_route_is_never_longer_than_default():
    from brl.g25o import route_open
    points = np.asarray([[0, 10], [9, 1], [9, 9], [1, 1], [5, 5]], dtype=float)
    start = np.zeros(2)
    default = route_open(points, start)
    multi = route_open_multistart(points, start)
    def length(seq):
        ordered = points[seq]
        return np.linalg.norm(ordered[0] - start) + np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum()
    assert length(multi) <= length(default) + 1e-9


def test_soft_budget_zero_falls_back_and_still_completes():
    sources = random_sources(3, 10, np.random.default_rng(300))
    env = RadioEnv(mode=3, n_sources=10, seed=300, bearing_decimals=2)
    env.reset(seed=300, n_sources=10, sources=sources)
    cfg = ISRV2Config(max_plan_total_s=0.0, max_plan_call_s=0.0)
    result = ISRV2Candidate(3, cfg).run(env)
    assert result["success"]
    assert result["soft_disabled_reason"] in {"per_call_budget", "episode_budget"}
    assert env.cleared_count() == 10


def test_two_failed_soft_probes_continue_to_bilateral(monkeypatch):
    env = RadioEnv(mode=3, n_sources=1, seed=301)
    env.reset(seed=301, n_sources=1, sources=[Source(1, np.array([500.0, 0.0]), 1500.0)])
    policy = ISRV2Policy(env, ISRV2Config(multistart_route=False, scan_reprice=False))
    env.channels[1].status = "discovered"
    policy.tracks[1] = {"first": np.zeros(2), "deg": 0.0,
                        "P": np.array([[0.0, -40.0], [80.0, -40.0], [80.0, 40.0]]), "nobs": 1}
    policy.soft[1] = SoftSourceBelief(3, 1, policy.tracks[1]["P"])
    choices = iter([
        (0, 0, 0, 0, np.array([10.0, 0.0]), 0.8, 4.0, 2.0),
        (0, 0, 0, 0, np.array([40.0, 0.0]), 0.7, 3.0, 6.0),
        None,
    ])
    monkeypatch.setattr(policy, "_probe_choice", lambda ch: next(choices))
    calls = {"clear": 0, "parent": 0}
    def clear(q, ch):
        calls["clear"] += 1
        return {"accepted": True, "clear_result": "no_target_in_range"}
    monkeypatch.setattr(env, "clear", clear)
    # Avoid exercising geometry here; the invariant is exactly two soft calls
    # followed by the inherited resolver.
    import brl.independent_candidate as base_module
    def solve(*args, **kwargs):
        calls["parent"] += 1
        env.channels[1].status = "cleared"
    monkeypatch.setattr(base_module, "solve_bilateral", solve)
    policy.resolve(1)
    assert calls == {"clear": 2, "parent": 1}
    assert policy.probe_counts[1] == 2


def test_isrv2_clears_q3_and_q4_smoke():
    for mode in (3, 4):
        sources = random_sources(mode, 10, np.random.default_rng(400 + mode))
        env = RadioEnv(mode=mode, n_sources=10, seed=400 + mode, bearing_decimals=2)
        env.reset(seed=400 + mode, n_sources=10, sources=sources)
        result = ISRV2Candidate(mode).run(env)
        assert result["success"]
        assert env.cleared_count() == 10
