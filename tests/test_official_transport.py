from __future__ import annotations

import io
import json
from urllib.error import URLError

import pytest

import brl.remote as remote_module
from brl.protocol import ActionIOError
from brl.remote import OfficialClient, RemoteBelief


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def test_network_retry_reuses_identical_request_id(monkeypatch):
    payloads = []
    cache = {}
    executed = {"count": 0}

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        payloads.append(payload)
        request_id = payload["request_id"]
        if request_id not in cache:
            executed["count"] += 1
            cache[request_id] = {"accepted": True, "virtual_time_s": 5.0,
                                 "measure_result": "no_signal"}
        if len(payloads) == 1:
            raise URLError("response lost")
        return FakeResponse(cache[request_id])

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_module.time, "sleep", lambda _: None)
    client = OfficialClient("http://test", "team", max_network_retries=2)
    response = client.measure([0.0, 0.0], 1)
    assert response["accepted"]
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert executed["count"] == 1


def test_clear_response_loss_reuses_identical_request_id(monkeypatch):
    payloads = []
    cached = {}

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        payloads.append(payload)
        cached.setdefault(payload["request_id"], {
            "accepted": True, "virtual_time_s": 3.0,
            "clear_result": "no_target_in_range",
        })
        if len(payloads) == 1:
            raise TimeoutError("executed but response lost")
        return FakeResponse(cached[payload["request_id"]])

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_module.time, "sleep", lambda _: None)
    client = OfficialClient("http://test", "team", max_network_retries=2)
    response = client.clear([10.0, 20.0], 4)
    assert response["clear_result"] == "no_target_in_range"
    assert payloads[0] == payloads[1]


def test_business_rejection_is_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        return FakeResponse({"accepted": False, "error": "rejected"})

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
    client = OfficialClient("http://test", "team", max_network_retries=5)
    assert not client.measure([0.0, 0.0], 1)["accepted"]
    assert len(calls) == 1


def test_remote_belief_wraps_transport_failure():
    class FailingClient:
        def measure(self, *args, **kwargs):
            raise TimeoutError("lost")

    belief = RemoteBelief(FailingClient(), mode=3)
    with pytest.raises(ActionIOError):
        belief.measure([0.0, 0.0], 1)


def test_remote_belief_rejects_missing_measure_result():
    class InvalidClient:
        def measure(self, *args, **kwargs):
            return {"accepted": True, "virtual_time_s": 5.0}

    belief = RemoteBelief(InvalidClient(), mode=3)
    with pytest.raises(ActionIOError):
        belief.measure([0.0, 0.0], 1)


def test_remote_belief_rejects_direction_without_finite_bearing():
    class InvalidClient:
        def measure(self, *args, **kwargs):
            return {"accepted": True, "virtual_time_s": 5.0,
                    "measure_result": "direction", "svd_deg": float("nan")}

    belief = RemoteBelief(InvalidClient(), mode=3)
    with pytest.raises(ActionIOError):
        belief.measure([0.0, 0.0], 1)


@pytest.mark.parametrize("response", [
    {"accepted": False, "error": "rejected"},
    {"accepted": True, "virtual_time_s": 3.0},
    {"accepted": True, "virtual_time_s": 3.0, "clear_result": "unknown"},
])
def test_remote_belief_rejects_invalid_clear_without_updating_ledger(response):
    class InvalidClient:
        def clear(self, *args, **kwargs):
            return response

    belief = RemoteBelief(InvalidClient(), mode=3)
    before = (belief.n_clear, belief.n_clear_fail, belief.virtual_time, belief.pos.copy())
    with pytest.raises(ActionIOError):
        belief.clear([50.0, 0.0], 1)
    after = (belief.n_clear, belief.n_clear_fail, belief.virtual_time, belief.pos.copy())
    assert before[:3] == after[:3]
    assert (before[3] == after[3]).all()


def test_exit_business_rejection_is_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse({"accepted": False, "error": "exit refused"})

    monkeypatch.setattr(remote_module, "urlopen", fake_urlopen)
    client = OfficialClient("http://test", "team", max_network_retries=5)
    response = client.exit()
    assert response["accepted"] is False
    assert len(calls) == 1
