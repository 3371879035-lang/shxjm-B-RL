from __future__ import annotations

"""Observable ISR-compatible executor used by CVR control experiments.

This is an explicit implementation of the frozen ISR scheduling rules.  It
does not call ``IndependentCandidate`` or ``JointShell.run``; differential
tests compare it against that implementation action by action.
"""

import math
import numpy as np

from brl.bilateral import bearing_clip, solve_bilateral
from brl.coverage import s25_points, s3_points
from brl.g25o import (_best_clear_point as best_clear,
                      _clip_target_square as target_clip,
                      _poly_center_radius as center_radius,
                      initial_track_polygon as initial_poly, route_open)
from brl.protocol import ActionIOError, CertificateViolation, GeometryNumericalError


class ISRCompatibilityExecutor:
    def __init__(self, env, *, probe: bool = True) -> None:
        self.env = env
        self.points = s3_points() if env.mode == 3 else s25_points()
        self.env.coverage_points = self.points.copy()
        self.env.n_coverage = len(self.points)
        self.tracks: dict[int, dict] = {}
        self.probe = bool(probe)
        self.max_region_radius = 800.0 if env.mode == 3 else 300.0
        self.outer_first = False
        self.solver_failures = 0

    def observe(self, channel: int, point: np.ndarray, response: dict) -> None:
        result = response.get("measure_result")
        if result == "near":
            if self.env.clear(point, channel)["clear_result"] == "success":
                self.tracks.pop(channel, None)
            return
        if result != "direction" or response.get("svd_deg") is None:
            return
        bearing = float(response["svd_deg"])
        if channel not in self.tracks:
            self.tracks[channel] = {
                "first": point.copy(), "deg": bearing,
                "P": initial_poly(point, bearing), "nobs": 1,
            }
        else:
            track = self.tracks[channel]
            track["P"] = target_clip(
                bearing_clip(track["P"], point, math.radians(bearing)))
            track["nobs"] += 1
        if not len(self.tracks[channel]["P"]):
            first = self.tracks[channel]
            first["P"] = initial_poly(first["first"], first["deg"])

    def scan(self, index: int) -> None:
        point = self.points[index]
        old_tracks = list(self.tracks)
        channels = [
            channel for channel, state in self.env.channels.items()
            if state.status == "unknown" and index not in state.scan_points
        ]
        channels.sort(key=lambda channel: (channel != self.env.current_channel, channel))
        for channel in channels:
            if self.env.done:
                return
            self.observe(channel, point, self.env.measure(point, channel, coverage_idx=index))
        self.refine(old_tracks, point)

    def refine(self, channels, point: np.ndarray) -> None:
        for channel in channels:
            if channel not in self.tracks or self.env.done:
                continue
            track = self.tracks[channel]
            center, radius = center_radius(track["P"])
            if radius <= 19.5 or track["nobs"] >= 4:
                continue
            delta = center - point
            distance = float(np.linalg.norm(delta))
            first_direction = center - track["first"]
            first_length = float(np.linalg.norm(first_direction))
            if first_length < 1e-9:
                continue
            sine = abs(delta[0] * first_direction[1] - delta[1] * first_direction[0]) / (
                distance * first_length + 1e-9)
            if distance <= 1200.0 and sine > 0.15:
                self.observe(channel, point,
                             self.env.measure(point, channel, is_refine=True))

    def resolve(self, channel: int) -> None:
        track = self.tracks[channel]
        if self.probe:
            center, radius = center_radius(track["P"])
            if 19.5 < radius <= 80.0:
                if self.env.clear(center, channel)["clear_result"] == "success":
                    self.tracks.pop(channel, None)
                    return
        try:
            solve_bilateral(
                track["first"], track["deg"], self.env.pos,
                lambda point: self.env.measure(point, channel, is_refine=True),
                lambda point: self.env.clear(point, channel),
                initial_region=track["P"],
            )
        except ActionIOError:
            raise
        except GeometryNumericalError:
            self.solver_failures += 1
            angle = math.radians(track["deg"])
            direction = np.array([math.cos(angle), math.sin(angle)])
            side = np.array([-direction[1], direction[0]])
            for along in np.linspace(0.0, 1500.0, 61):
                for offset in (13.3, -13.3):
                    if self.env.done:
                        break
                    point = track["first"] + along * direction + offset * side
                    if self.env.clear(point, channel)["clear_result"] == "success":
                        break
                if self.env.channels[channel].status == "cleared":
                    break
        if self.env.channels[channel].status != "cleared":
            raise CertificateViolation("geometric fallback failed: not a completion certificate")
        self.tracks.pop(channel, None)

    def _pending(self) -> set[int]:
        return {
            index for index in range(len(self.points))
            if any(state.status == "unknown" and index not in state.scan_points
                   for state in self.env.channels.values())
        }

    def _choice(self, pending: set[int]):
        outer = {index for index in pending if index >= 13}
        active = outer if self.outer_first and outer else pending
        jobs = [("scan", index) for index in sorted(active)]
        positions = [self.points[index] for _, index in jobs]
        for channel, track in self.tracks.items():
            center, radius = center_radius(track["P"])
            if not pending or radius <= self.max_region_radius:
                jobs.append(("resolve", channel))
                positions.append(center)
        if not jobs:
            return None
        return jobs[route_open(np.asarray(positions), self.env.pos)[0]]

    def resume_after_origin(self, *, max_steps: int = 100) -> None:
        steps = 0
        while not self.env.done and steps < max_steps:
            steps += 1
            pending = self._pending()
            if self.env.cleared_count() + self.env.discovered_count() >= 16:
                pending.clear()
            choice = self._choice(pending)
            if choice is None:
                if self.tracks:
                    channels = list(self.tracks)
                    centers = [center_radius(self.tracks[ch]["P"])[0] for ch in channels]
                    self.resolve(channels[route_open(np.asarray(centers), self.env.pos)[0]])
                    continue
                break
            kind, key = choice
            if kind == "scan":
                self.scan(key)
            else:
                self.resolve(key)
        if not self.env.done:
            order = route_open(self.points, np.zeros(2))
            if self.env.mode == 4:
                order = [0, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                         15, 14, 13, 24, 23, 22, 21, 20, 19, 18, 17, 16]
            for index in order:
                if any(state.status == "unknown" and index not in state.scan_points
                       for state in self.env.channels.values()):
                    self.scan(index)
            while self.tracks and not self.env.done:
                channels = list(self.tracks)
                centers = [center_radius(self.tracks[ch]["P"])[0] for ch in channels]
                self.resolve(channels[route_open(np.asarray(centers), self.env.pos)[0]])

    def run(self) -> None:
        self.scan(0)
        self.outer_first = bool(self.env.mode == 4 and self.env.discovered_count() <= 0)
        self.resume_after_origin()
