from __future__ import annotations

import numpy as np
import pytest

from brl.bilateral import solve_bilateral
from brl.bilateral_state import BilateralState
from brl.g25o import initial_track_polygon
from brl.local_env import RadioEnv, Source
from brl.protocol import CertificateViolation


def _case():
    source = Source(1, np.array([900.0, 160.0]), 1500.0)
    env = RadioEnv(mode=3, n_sources=1, seed=41, bearing_decimals=2)
    env.reset(seed=41, sources=[source])
    env.error_field.value = lambda *args: 0.0
    first = env.measure([0.0, 0.0], 1, is_refine=True)
    return env, first


def _callbacks(env, actions):
    def measure(q):
        actions.append(("measure", np.asarray(q).copy()))
        return env.measure(q, 1, is_refine=True)
    def clear(q):
        actions.append(("clear", np.asarray(q).copy()))
        return env.clear(q, 1)
    return measure, clear


def test_resumable_continuous_actions_match_original_solver():
    env1, first1 = _case(); actions1 = []
    m1, c1 = _callbacks(env1, actions1)
    solve_bilateral([0.0, 0.0], first1["svd_deg"], env1.pos, m1, c1,
                    initial_region=initial_track_polygon(np.zeros(2), first1["svd_deg"]))

    env2, first2 = _case(); actions2 = []
    m2, c2 = _callbacks(env2, actions2)
    state = BilateralState([0.0, 0.0], first2["svd_deg"], env2.pos,
                           initial_region=initial_track_polygon(np.zeros(2), first2["svd_deg"]))
    state.finish(m2, c2)
    assert state.success and env2.cleared_count() == 1
    assert [a[0] for a in actions2] == [a[0] for a in actions1]
    assert all(np.allclose(a[1], b[1]) for a, b in zip(actions1, actions2))
    assert env2.virtual_time == pytest.approx(env1.virtual_time)


def test_checkpoint_preserves_pending_symmetric_probe():
    state = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    first = state.next_action()
    state.accept_observation(first.action_id,
                             {"accepted": True, "measure_result": "no_signal"})
    checkpoint = state.checkpoint()
    a = state.next_action(); b = checkpoint.next_action()
    assert a == b
    checkpoint.accept_observation(b.action_id,
                                  {"accepted": True, "measure_result": "no_signal"})
    assert checkpoint.hi == pytest.approx(750.0)


def test_duplicate_response_is_idempotent_and_conflict_is_rejected():
    state = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    action = state.next_action()
    response = {"accepted": True, "measure_result": "no_signal"}
    state.accept_observation(action.action_id, response)
    measures = state.extra_measures
    state.accept_observation(action.action_id, response)
    assert state.extra_measures == measures
    with pytest.raises(CertificateViolation, match="conflicting"):
        state.accept_observation(action.action_id,
                                 {"accepted": True, "measure_result": "direction", "svd_deg": 0.0})


def test_checkpoint_preserves_terminal_and_near_phases():
    terminal = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    terminal.hi = 20.0
    action = terminal.next_action()
    assert action.kind == "clear" and terminal.phase == "terminal_clear"
    assert terminal.checkpoint().next_action() == action

    near = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    probe = near.next_action()
    near.accept_observation(probe.action_id,
                            {"accepted": True, "measure_result": "near"})
    clear = near.next_action()
    assert clear.kind == "clear" and near.phase == "near_clear"
    assert near.checkpoint().next_action() == clear


def test_external_bearing_does_not_replace_pending_symmetric_probe():
    state = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    first = state.next_action()
    state.accept_observation(first.action_id,
                             {"accepted": True, "measure_result": "no_signal"})
    pending = state.next_action()
    state.incorporate_bearing([0.0, 0.0], 0.0)
    assert state.next_action() == pending


def test_certified_or_near_clear_failure_is_certificate_violation():
    state = BilateralState([0.0, 0.0], 0.0, [0.0, 0.0])
    probe = state.next_action()
    state.accept_observation(probe.action_id,
                             {"accepted": True, "measure_result": "near"})
    clear = state.next_action()
    with pytest.raises(CertificateViolation, match="contradicted"):
        state.accept_observation(clear.action_id,
                                 {"accepted": True, "clear_result": "no_target_in_range"})
