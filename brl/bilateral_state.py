"""Resumable form of the certified bilateral localization procedure.

The state machine exposes one validated action at a time.  Pausing never
rebuilds a round: the two symmetric probes and their evidence provenance are
part of the checkpoint.  It has no access to simulator truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Callable, Optional

import numpy as np

from .bilateral import EPS, K, R_MAX, _ordered_by_travel, bearing_clip, clip
from .geometry import minimum_enclosing_circle
from .protocol import CertificateViolation, GeometryNumericalError


@dataclass(frozen=True)
class BilateralAction:
    action_id: int
    kind: str
    position: tuple[float, float]


class BilateralState:
    """Serializable evidence state for one discovered source."""

    def __init__(self, first_position, first_bearing_deg: float, current_position,
                 *, geometry: bool = True, initial_region: Optional[np.ndarray] = None):
        self.origin = np.asarray(first_position, dtype=float).copy()
        self.first_bearing_deg = float(first_bearing_deg)
        a0 = math.radians(self.first_bearing_deg)
        u = np.array([math.cos(a0), math.sin(a0)], dtype=float)
        v = np.array([-u[1], u[0]], dtype=float)
        self.B = np.stack([u, v], axis=1)
        self.pos = np.asarray(current_position, dtype=float).copy()
        self.geometry = bool(geometry)
        self.lo, self.hi = 0.0, R_MAX
        self.poly = np.array([[0.0, 0.0], [R_MAX, -R_MAX * K],
                              [R_MAX, R_MAX * K]], dtype=float)
        if initial_region is not None:
            supplied = (np.asarray(initial_region, dtype=float) - self.origin) @ self.B
            bounds = ((np.array([-1.0, 0.0]), 0.0),
                      (np.array([1.0, 0.0]), R_MAX),
                      (np.array([-K, 1.0]), 0.0),
                      (np.array([-K, -1.0]), 0.0))
            for normal, offset in bounds:
                supplied = clip(supplied, normal, offset)
            if len(supplied) == 0:
                raise GeometryNumericalError("invalid initial conservative region")
            self.poly = supplied
            self.lo = max(self.lo, float(self.poly[:, 0].min()))
            self.hi = min(self.hi, float(self.poly[:, 0].max()))
        self.rounds = 0
        self.extra_measures = 0
        self.clear_attempts = 0
        self.phase = "ready"
        self.round_mid: Optional[float] = None
        self.round_probes: list[np.ndarray] = []
        self.pending_probe = 0
        self.terminal_targets: list[np.ndarray] = []
        self.pending_terminal = 0
        self.done = False
        self.success = False
        self._next_id = 1
        self._current_action: Optional[BilateralAction] = None
        self._processed: dict[int, tuple] = {}

    def checkpoint(self) -> "BilateralState":
        return copy.deepcopy(self)

    def preview_action(self) -> Optional[BilateralAction]:
        """Return the next action without mutating evidence or caching a token."""
        shadow = copy.copy(self)
        shadow.origin = self.origin.copy()
        shadow.B = self.B.copy()
        shadow.pos = self.pos.copy()
        shadow.poly = self.poly.copy()
        shadow.round_probes = [p.copy() for p in self.round_probes]
        shadow.terminal_targets = [p.copy() for p in self.terminal_targets]
        shadow._processed = dict(self._processed)
        return shadow.next_action()

    @property
    def region(self) -> np.ndarray:
        return self.poly @ self.B.T + self.origin

    def _glob(self, local) -> np.ndarray:
        return self.origin + self.B @ np.asarray(local, dtype=float)

    def incorporate_bearing(self, position, bearing_deg: float) -> None:
        """Intersect an additional accepted public bearing with the hard region."""
        global_poly = bearing_clip(self.region, np.asarray(position, dtype=float),
                                   math.radians(float(bearing_deg)))
        if len(global_poly) == 0:
            raise GeometryNumericalError("external bearing emptied certified region")
        self.poly = (global_poly - self.origin) @ self.B
        self.lo = max(self.lo, float(self.poly[:, 0].min()))
        self.hi = min(self.hi, float(self.poly[:, 0].max()))
        if self.lo > self.hi + 1e-7:
            raise GeometryNumericalError("external bearing inverted interval")
        if self.geometry:
            mec = minimum_enclosing_circle(self.poly)
            if float(mec.radius) <= 19.5:
                # Keep round_mid/round_probes as provenance in the checkpoint,
                # but a newly formed hard clear certificate takes precedence.
                self.phase = "certified_clear"
                self.terminal_targets = [np.asarray(mec.center, dtype=float)]
                self.pending_terminal = 0
                self._current_action = None

    def _prepare(self) -> None:
        if self.done or self.phase != "ready":
            return
        if self.geometry:
            self.lo = max(self.lo, float(self.poly[:, 0].min()))
            self.hi = min(self.hi, float(self.poly[:, 0].max()))
            mec = minimum_enclosing_circle(self.poly)
            if float(mec.radius) <= 19.5:
                self.phase = "certified_clear"
                self.terminal_targets = [np.asarray(mec.center, dtype=float)]
                self.pending_terminal = 0
                return
        if self.hi - self.lo <= 24.0 + 1e-7:
            mid = (self.lo + self.hi) / 2.0
            targets = [np.array([mid, self.hi * K / 2.0]),
                       np.array([mid, -self.hi * K / 2.0])]
            self.terminal_targets = _ordered_by_travel(
                targets, self.B.T @ (self.pos - self.origin))
            self.pending_terminal = 0
            self.phase = "terminal_clear"
            return
        if self.rounds >= 7:
            raise GeometryNumericalError("interval did not contract within seven rounds")
        self.rounds += 1
        self.round_mid = (self.lo + self.hi) / 2.0
        side = self.round_mid * K + 5.0
        probes = [np.array([self.round_mid, side]),
                  np.array([self.round_mid, -side])]
        self.round_probes = _ordered_by_travel(
            probes, self.B.T @ (self.pos - self.origin))
        self.pending_probe = 0
        self.phase = "round"

    def next_action(self) -> Optional[BilateralAction]:
        if self.done:
            return None
        if self._current_action is not None:
            return self._current_action
        self._prepare()
        if self.phase == "round":
            kind = "measure"
            point = self._glob(self.round_probes[self.pending_probe])
        elif self.phase in {"certified_clear", "terminal_clear", "near_clear"}:
            kind = "clear"
            point = self._glob(self.terminal_targets[self.pending_terminal])
        else:
            raise GeometryNumericalError(f"invalid bilateral phase {self.phase}")
        action = BilateralAction(self._next_id, kind,
                                 (float(point[0]), float(point[1])))
        self._next_id += 1
        self._current_action = action
        return action

    @staticmethod
    def _response_key(response: dict) -> tuple:
        return tuple(sorted((str(k), repr(v)) for k, v in response.items()))

    def accept_observation(self, action_id: int, response: dict) -> None:
        key = self._response_key(response)
        if action_id in self._processed:
            if self._processed[action_id] != key:
                raise CertificateViolation("same bilateral action received conflicting responses")
            return
        action = self._current_action
        if action is None or action.action_id != int(action_id):
            raise CertificateViolation("stale or unknown bilateral action response")
        if response.get("accepted", True) is not True:
            raise CertificateViolation("bilateral action was not accepted")
        self.pos = np.asarray(action.position, dtype=float)
        if action.kind == "measure":
            self.extra_measures += 1
            status = response.get("measure_result")
            if status == "near":
                self.phase = "near_clear"
                local = (self.pos - self.origin) @ self.B
                self.terminal_targets = [local]
                self.pending_terminal = 0
            elif status == "direction":
                try:
                    bearing = float(response["svd_deg"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise CertificateViolation("direction response missing bearing") from exc
                if not np.isfinite(bearing):
                    raise CertificateViolation("direction response has non-finite bearing")
                mid = float(self.round_mid)
                side = abs(float(self.round_probes[self.pending_probe][1]))
                theta = math.radians(bearing - self.first_bearing_deg)
                cx = math.cos(theta)
                if cx > math.sin(EPS) + 1e-12:
                    self.lo = max(self.lo, mid)
                elif cx < -math.sin(EPS) - 1e-12:
                    self.hi = min(self.hi, mid)
                else:
                    width = (R_MAX * K + side) * math.tan(2 * EPS)
                    self.lo = max(self.lo, mid - width)
                    self.hi = min(self.hi, mid + width)
                if self.geometry:
                    local = (self.pos - self.origin) @ self.B
                    self.poly = bearing_clip(self.poly, local, theta)
                self._finish_round()
            elif status == "no_signal":
                self.pending_probe += 1
                if self.pending_probe >= len(self.round_probes):
                    self.hi = min(self.hi, float(self.round_mid))
                    self._finish_round()
            else:
                raise CertificateViolation(f"invalid measure response {response}")
        else:
            self.clear_attempts += 1
            result = response.get("clear_result")
            if result == "success":
                self.done = True
                self.success = True
                self.phase = "done"
            elif result != "no_target_in_range":
                raise CertificateViolation(f"invalid clear response {response}")
            elif self.phase in {"certified_clear", "near_clear"}:
                raise CertificateViolation(f"{self.phase} contradicted by failed clear")
            else:
                self.pending_terminal += 1
                if self.pending_terminal >= len(self.terminal_targets):
                    raise CertificateViolation("two-disk terminal cover failed")
        self._processed[action_id] = key
        self._current_action = None

    def _finish_round(self) -> None:
        self.poly = clip(self.poly, np.array([1.0, 0.0]), self.hi)
        self.poly = clip(self.poly, np.array([-1.0, 0.0]), -self.lo)
        if len(self.poly) == 0 or self.lo > self.hi + 1e-7:
            raise GeometryNumericalError("empty certified region")
        self.phase = "ready"
        self.round_mid = None
        self.round_probes = []
        self.pending_probe = 0

    def finish(self, measure: Callable, clear: Callable) -> "BilateralState":
        while not self.done:
            action = self.next_action()
            if action is None:
                break
            point = np.asarray(action.position, dtype=float)
            response = measure(point) if action.kind == "measure" else clear(point)
            self.accept_observation(action.action_id, response)
        return self
