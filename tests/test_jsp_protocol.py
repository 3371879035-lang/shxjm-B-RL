from __future__ import annotations

import numpy as np
import pytest

from brl.local_env import RadioEnv, Source
from brl.protocol import ActionIOError, DeadlineExceeded
from brl.remote import OfficialClient, RemoteBelief


class StaticClient:
    def __init__(self, response):
        self.response = response

    def measure(self, *args, **kwargs):
        return dict(self.response)

    def clear(self, *args, **kwargs):
        return dict(self.response)


def test_local_coverage_marker_requires_exact_station():
    env = RadioEnv(mode=3, n_sources=1, seed=1, bearing_decimals=2)
    env.reset(seed=1, sources=[Source(1, np.array([500.0, 0.0]), 1500.0)])
    with pytest.raises(ValueError, match="does not match"):
        env.measure(env.coverage_points[0] + np.array([0.01, 0.0]), 1, coverage_idx=0)
    assert env.n_measure == 0


@pytest.mark.parametrize("bad_time", [None, float("nan"), -1.0])
def test_remote_invalid_clock_does_not_mutate_ledger(bad_time):
    response = {"accepted": True, "measure_result": "no_signal"}
    if bad_time is not None:
        response["virtual_time_s"] = bad_time
    belief = RemoteBelief(StaticClient(response), mode=3)
    before = (belief.pos.copy(), belief.current_channel, belief.n_measure,
              belief.virtual_time, list(belief.channels[2].observations))
    with pytest.raises(ActionIOError):
        belief.measure([10.0, 0.0], 2)
    assert np.array_equal(belief.pos, before[0])
    assert (belief.current_channel, belief.n_measure, belief.virtual_time) == before[1:4]
    assert belief.channels[2].observations == before[4]


def test_remote_rejects_mismatched_coverage_before_action():
    client = StaticClient({"accepted": True, "virtual_time_s": 5.0,
                           "measure_result": "no_signal"})
    belief = RemoteBelief(client, mode=3)
    with pytest.raises(ActionIOError, match="does not match"):
        belief.measure(belief.coverage_points[0] + [1.0, 0.0], 1, coverage_idx=0)
    assert belief.n_measure == 0


def test_quantized_public_region_contains_rounding_boundary_target():
    # True direction 1.004 degrees rounds to 1.00.  A one-degree wedge would
    # exclude it by 0.004 degrees; the explicit 1.01 envelope must contain it.
    radius = 1000.0
    theta = np.deg2rad(1.004)
    source = Source(1, radius * np.array([np.cos(theta), np.sin(theta)]), 1500.0)
    env = RadioEnv(mode=3, n_sources=1, seed=2, bearing_decimals=2)
    env.reset(seed=2, sources=[source])
    env.error_field.value = lambda *args: 0.0
    obs = env.measure([0.0, 0.0], 1)
    assert obs["svd_deg"] == 1.0
    # Convex polygon containment without relying on vertex orientation.
    poly = env.channels[1].poly
    edges = np.roll(poly, -1, axis=0) - poly
    rel = source.position - poly
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    assert np.all(cross >= -1e-7) or np.all(cross <= 1e-7)


def test_official_deadline_stops_action_but_keeps_exit_available(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"accepted":true}'

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return Response()

    monkeypatch.setattr("brl.remote.urlopen", fake_urlopen)
    client = OfficialClient("http://example.invalid", "r", max_network_retries=1)
    client.action_deadline_monotonic = 0.0
    with pytest.raises(DeadlineExceeded):
        client.measure([0.0, 0.0], 1)
    assert calls == []
    assert client.exit()["accepted"] is True
    assert calls == ["http://example.invalid/exit"]


@pytest.mark.parametrize("remaining", [None, float("nan"), 0.0])
def test_enter_requires_finite_positive_remaining_duration(monkeypatch, remaining):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            payload = {"accepted": True}
            if remaining is not None:
                payload["remaining_real_duration_s"] = remaining
            import json
            return json.dumps(payload).encode()

    monkeypatch.setattr("brl.remote.urlopen", lambda *args, **kwargs: Response())
    client = OfficialClient("http://example.invalid", "r", max_network_retries=1)
    with pytest.raises(ActionIOError):
        client.enter()
    assert client.remaining_real_duration_s is None
