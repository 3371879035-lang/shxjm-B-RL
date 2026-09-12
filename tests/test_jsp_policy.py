from __future__ import annotations

import dataclasses
import json
import numpy as np
import pytest

from brl.jsp import JSPCandidate, JSPConfig, JSPPolicy
from brl.jsp.snapshot import candidate_features
from brl.local_env import RadioEnv, Source, random_sources


@pytest.mark.parametrize("mode", [3, 4])
@pytest.mark.parametrize("count", [10, 16])
def test_jsp_analytic_clears_minimum_and_maximum(mode, count):
    seed = 5000 + mode * 100 + count
    sources = random_sources(mode, count, np.random.default_rng(seed))
    env = RadioEnv(mode=mode, n_sources=count, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    result = JSPCandidate(mode).run(env)
    assert result["success"]
    assert env.cleared_count() == count
    residual = env.virtual_time - (env.move_distance / 5 + 5 * env.n_measure
                                   + env.n_switch + 3 * env.n_clear_fail
                                   + 5 * env.cleared_count())
    assert abs(residual) <= 1e-6


def test_public_snapshot_and_features_are_immutable():
    env = RadioEnv(mode=3, n_sources=10, seed=77, bearing_decimals=2)
    policy = JSPPolicy(env)
    snap = policy.public_snapshot()
    candidate = policy.candidates()[0]
    row = candidate_features(snap, policy._public_candidate(candidate))
    assert not row.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.current_channel = 5


def test_zero_planning_budget_falls_back_and_clears():
    seed = 88
    sources = random_sources(4, 10, np.random.default_rng(seed))
    env = RadioEnv(mode=4, n_sources=10, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    cfg = JSPConfig(max_plan_total_s=0.0, max_plan_call_s=0.0)
    result = JSPCandidate(4, config=cfg).run(env)
    assert result["success"] and env.cleared_count() == 10
    assert result["planning_disabled_reason"] in {"per_call_budget", "episode_budget"}


def test_candidate_set_always_contains_frozen_baseline_choice():
    env = RadioEnv(mode=4, n_sources=10, seed=99, bearing_decimals=2)
    policy = JSPPolicy(env)
    candidates = policy.candidates()
    assert sum(c.baseline for c in candidates) == 1


def test_partial_scan_prices_return_to_unfinished_station():
    env = RadioEnv(mode=3, n_sources=10, seed=100, bearing_decimals=2)
    policy = JSPPolicy(env)
    candidates = policy.candidates()
    partial = next(c for c in candidates if c.kind == "scan4" and c.key == 0)
    full = next(c for c in candidates if c.kind == "scanall" and c.key == 0)
    # At the same station, segmenting is allowed but must not look cheaper by
    # pretending the remaining 16 coverage measurements disappeared.
    assert partial.analytic_score >= full.analytic_score - 100.0


def test_candidate_preview_does_not_advance_locator_state():
    env = RadioEnv(mode=3, n_sources=10, seed=101, bearing_decimals=2)
    policy = JSPPolicy(env)
    policy._scan(0, 4)
    for state in policy.locators.values():
        before = (state.phase, state.rounds, state._current_action)
        policy.candidates()
        assert (state.phase, state.rounds, state._current_action) == before


def test_features_are_translation_invariant_and_hide_truth():
    env_a = RadioEnv(mode=3, n_sources=10, seed=201, bearing_decimals=2)
    env_b = RadioEnv(mode=3, n_sources=16, seed=202, bearing_decimals=2)
    policy_a, policy_b = JSPPolicy(env_a), JSPPolicy(env_b)
    cand_a, cand_b = policy_a.candidates()[0], policy_b.candidates()[0]
    assert cand_a == cand_b
    row_a = candidate_features(policy_a.public_snapshot(), policy_a._public_candidate(cand_a))
    row_b = candidate_features(policy_b.public_snapshot(), policy_b._public_candidate(cand_b))
    assert np.array_equal(row_a, row_b)
    snap = policy_a.public_snapshot()
    shifted_snap = dataclasses.replace(
        snap, position=(snap.position[0] + 317.0, snap.position[1] - 211.0))
    shifted_cand = dataclasses.replace(
        policy_a._public_candidate(cand_a),
        position=(cand_a.position[0] + 317.0, cand_a.position[1] - 211.0))
    shifted = candidate_features(shifted_snap, shifted_cand)
    assert np.allclose(row_a, shifted)


def test_decision_log_is_machine_readable():
    seed = 303
    sources = random_sources(3, 10, np.random.default_rng(seed))
    env = RadioEnv(mode=3, n_sources=10, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    result = JSPCandidate(3).run(env)
    payload = json.dumps(result, allow_nan=False)
    assert payload and result["decision_log"]
    assert "candidates" in result["decision_log"][0]
    assert "selection" in result["decision_log"][0]


def test_missing_learned_model_disables_learning_and_finishes_baseline(tmp_path):
    seed = 404
    sources = random_sources(4, 10, np.random.default_rng(seed))
    env = RadioEnv(mode=4, n_sources=10, seed=seed, bearing_decimals=2)
    env.reset(seed=seed, sources=sources)
    result = JSPCandidate(4, planner="learned",
                          model_path=str(tmp_path / "missing.joblib")).run(env)
    assert result["success"] and env.cleared_count() == 10
    assert result["planning_disabled_reason"].startswith("model:")


def test_discovered_source_has_interruptible_and_continuous_candidates_without_radius_gate():
    env = RadioEnv(mode=4, n_sources=1, seed=505, bearing_decimals=2)
    env.reset(seed=505, sources=[Source(1, np.array([1400.0, 0.0]), 1500.0)])
    env.error_field.value = lambda *args: 0.0
    policy = JSPPolicy(env)
    policy._scan(0, 1)
    candidates = policy.candidates()
    assert any(c.kind == "locate1" and c.key == 1 for c in candidates)
    assert any(c.kind == "resolve" and c.key == 1 for c in candidates)
