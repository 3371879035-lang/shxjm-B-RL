from __future__ import annotations

"""Durable action journal and strict response validation for CVR experiments."""

import json
import math
import copy
from pathlib import Path
from typing import Callable

import numpy as np

from brl.protocol import ActionIOError


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class EventJournal:
    def __init__(self, path: Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.events: list[dict] = []
        self._sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise FileExistsError(f"event journal already exists: {self.path}")

    def emit(self, event: dict) -> dict:
        self._sequence += 1
        row = _json_value({"sequence": self._sequence, **event})
        self.events.append(row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        return row


class JournaledActionView:
    """Proxy every strategy action through registration, validation, and commit."""

    def __init__(self, env, journal: EventJournal) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "journal", journal)
        object.__setattr__(self, "_request_serial", 0)

    def __getattr__(self, key):
        return getattr(self._env, key)

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        inner = object.__getattribute__(self, "_env")
        try:
            raw = object.__getattribute__(inner, "_env")
            from brl.independent_candidate import CheckedView
            cloned_inner = CheckedView(copy.deepcopy(raw, memo))
        except AttributeError:
            cloned_inner = copy.deepcopy(inner, memo)
        object.__setattr__(clone, "_env", cloned_inner)
        object.__setattr__(clone, "journal", copy.deepcopy(self.journal, memo))
        object.__setattr__(clone, "_request_serial", self._request_serial)
        return clone

    def __setattr__(self, key, value):
        if key in {"_env", "journal", "_request_serial"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self._env, key, value)

    def _request_id(self) -> str:
        serial = self._request_serial + 1
        object.__setattr__(self, "_request_serial", serial)
        return f"action-{serial:05d}"

    def _before(self) -> dict:
        return {
            "virtual_time_s": float(self.virtual_time),
            "distance_m": float(self.move_distance),
            "measure_calls": int(self.n_measure),
            "switches": int(self.n_switch),
            "clear_calls": int(self.n_clear),
            "failed_clear": int(self.n_clear_fail),
        }

    def _execute(self, kind: str, position, channel: int, call: Callable, **metadata):
        request_id = self._request_id()
        point = tuple(map(float, np.asarray(position, dtype=float)))
        before = self._before()
        self.journal.emit({
            "event": "action_registered", "request_id": request_id,
            "action_kind": kind, "channel": int(channel), "position": list(point),
            "before": before, **metadata,
        })
        try:
            response = call()
            after = self._before()
            if not math.isfinite(after["virtual_time_s"]):
                raise ActionIOError("official time is not finite")
            if after["virtual_time_s"] + 1e-12 < before["virtual_time_s"]:
                raise ActionIOError("official time moved backward")
            self.journal.emit({
                "event": "action_confirmed", "request_id": request_id,
                "action_kind": kind, "channel": int(channel), "position": list(point),
                "before": before, "after": after, "response": dict(response), **metadata,
            })
            return response
        except Exception as exc:
            self.journal.emit({
                "event": "action_failed", "request_id": request_id,
                "action_kind": kind, "channel": int(channel), "position": list(point),
                "before": before, "error_type": type(exc).__name__, "error": str(exc),
                **metadata,
            })
            raise

    def measure(self, position, channel, coverage_idx=None, is_refine=False):
        return self._execute(
            "measure", position, channel,
            lambda: self._env.measure(position, channel, coverage_idx=coverage_idx,
                                      is_refine=is_refine),
            coverage_idx=coverage_idx, is_refine=bool(is_refine),
        )

    def clear(self, position, channel):
        return self._execute(
            "clear", position, channel,
            lambda: self._env.clear(position, channel),
            coverage_idx=None, is_refine=False,
        )
