"""ISR-v2 soft planning layer on top of the certified ISR policy.

The hard polygon, S3/S25 coverage obligations, bilateral solver and completion
certificate remain authoritative.  Samples in this module only rank actions;
an empty sample set never proves that a channel is absent or cleared.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from brl.g25o import _poly_center_radius as center_radius
from brl.g25o import route_open
from brl.independent_candidate import CheckedView, JointShell


ANGLE_ENVELOPE_DEG = 1.01
ORIENTATION_BINS = 72
RADIUS_BINS = np.linspace(1000.0, 1500.0, 6)


def _halton(index: int, base: int) -> float:
    f = 1.0
    out = 0.0
    i = int(index)
    while i:
        f /= base
        out += f * (i % base)
        i //= base
    return out


def _inside_convex(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Orientation-independent containment for a convex polygon."""
    if len(poly) < 3 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)
    edges = np.roll(poly, -1, axis=0) - poly
    rel = points[:, None, :] - poly[None, :, :]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    return np.all(cross >= -1e-8, axis=1) | np.all(cross <= 1e-8, axis=1)


def deterministic_samples(poly: np.ndarray, channel: int, limit: int = 512) -> np.ndarray:
    """Low-discrepancy samples inside a hard polygon, independent of env RNG."""
    limit = min(2048, max(1, int(limit)))
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return np.empty((0, 2), dtype=float)
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    accepted: list[np.ndarray] = []
    # The first bearing region is a narrow triangle.  Batched rejection remains
    # deterministic and caps work even for clipped slivers.
    offset = 257 * int(channel)
    for batch in range(32):
        ids = np.arange(1 + offset + batch * 2048, 1 + offset + (batch + 1) * 2048)
        uv = np.asarray([[_halton(int(i), 2), _halton(int(i), 3)] for i in ids])
        pts = lo + uv * (hi - lo)
        keep = pts[_inside_convex(pts, poly)]
        if len(keep):
            accepted.extend(keep)
        if len(accepted) >= limit:
            break
    if not accepted:
        return np.empty((0, 2), dtype=float)
    return np.asarray(accepted[:limit], dtype=float)


def _bearing_error_deg(points: np.ndarray, origin: np.ndarray, bearing_deg: float) -> np.ndarray:
    angles = np.degrees(np.arctan2(points[:, 1] - origin[1], points[:, 0] - origin[0])) % 360.0
    return np.abs((angles - float(bearing_deg) + 180.0) % 360.0 - 180.0)


class SoftSourceBelief:
    """Deterministic sample posterior used only to rank legal actions."""

    def __init__(self, mode: int, channel: int, hard_poly: np.ndarray):
        self.mode = int(mode)
        self.channel = int(channel)
        self.points = deterministic_samples(hard_poly, channel)
        self.disabled_reason = "" if len(self.points) else "sample_empty"
        self._orient_deg = np.arange(ORIENTATION_BINS, dtype=float) * (360.0 / ORIENTATION_BINS)
        # Last orientation index is the all-direction (omni) hypothesis.
        self.alive = np.ones((len(self.points), ORIENTATION_BINS + 1, len(RADIUS_BINS)), dtype=bool)

    @property
    def active(self) -> bool:
        return not self.disabled_reason and len(self.points) > 0 and bool(np.any(self.alive))

    def _disable_if_empty(self) -> None:
        live_pos = np.any(self.alive, axis=(1, 2)) if len(self.alive) else np.zeros(0, dtype=bool)
        if not np.any(live_pos):
            self.disabled_reason = "soft_posterior_empty"

    def update_direction(self, position: Iterable[float], bearing_deg: float) -> None:
        if not self.active:
            return
        p = np.asarray(position, dtype=float)
        keep = _bearing_error_deg(self.points, p, bearing_deg) <= ANGLE_ENVELOPE_DEG + 1e-9
        dist = np.linalg.norm(self.points - p, axis=1)
        radial = dist[:, None] <= RADIUS_BINS[None, :] + 1e-9
        if self.mode == 4:
            vec = p[None, :] - self.points
            directions = np.column_stack((np.cos(np.radians(self._orient_deg)), np.sin(np.radians(self._orient_deg))))
            facing = vec @ directions.T >= -1e-9
            visible = np.concatenate((facing, np.ones((len(self.points), 1), dtype=bool)), axis=1)
            self.alive &= keep[:, None, None] & visible[:, :, None] & radial[:, None, :]
        else:
            self.alive &= keep[:, None, None] & radial[:, None, :]
        self._disable_if_empty()

    def update_no_signal(self, position: Iterable[float]) -> None:
        if self.mode != 4 or not self.active:
            return
        p = np.asarray(position, dtype=float)
        dist = np.linalg.norm(self.points - p, axis=1)
        radial = dist[:, None] <= RADIUS_BINS[None, :] + 1e-9
        vec = p[None, :] - self.points
        directions = np.column_stack((np.cos(np.radians(self._orient_deg)), np.sin(np.radians(self._orient_deg))))
        facing = vec @ directions.T >= -1e-9
        visible = np.concatenate((facing, np.ones((len(self.points), 1), dtype=bool)), axis=1)
        self.alive &= ~(visible[:, :, None] & radial[:, None, :])
        self._disable_if_empty()

    def update_failed_clear(self, position: Iterable[float]) -> None:
        if not self.active:
            return
        p = np.asarray(position, dtype=float)
        outside = np.linalg.norm(self.points - p, axis=1) > 20.0 + 1e-9
        self.alive &= outside[:, None, None]
        self._disable_if_empty()

    def live_points(self) -> np.ndarray:
        if not self.active:
            return np.empty((0, 2), dtype=float)
        return self.points[np.any(self.alive, axis=(1, 2))]

    def hit_fraction(self, position: Iterable[float], radius: float = 19.0) -> float:
        pts = self.live_points()
        if not len(pts):
            return 0.0
        return float(np.mean(np.linalg.norm(pts - np.asarray(position, dtype=float), axis=1) <= radius))

    def p_signal(self, position: Iterable[float]) -> float:
        if not self.active:
            return 0.0
        p = np.asarray(position, dtype=float)
        dist = np.linalg.norm(self.points - p, axis=1)
        radial = dist[:, None] <= RADIUS_BINS[None, :] + 1e-9
        if self.mode == 3:
            visible = radial[:, None, :]
        else:
            vec = p[None, :] - self.points
            directions = np.column_stack((np.cos(np.radians(self._orient_deg)), np.sin(np.radians(self._orient_deg))))
            facing = vec @ directions.T >= -1e-9
            geom = np.concatenate((facing, np.ones((len(self.points), 1), dtype=bool)), axis=1)
            visible = geom[:, :, None] & radial[:, None, :]
        denom = int(np.count_nonzero(self.alive))
        return float(np.count_nonzero(self.alive & visible) / denom) if denom else 0.0

    def clear_candidates(self, hard_center: np.ndarray, current: np.ndarray, limit: int = 32) -> list[np.ndarray]:
        pts = self.live_points()
        raw = [np.asarray(hard_center, dtype=float), np.asarray(current, dtype=float)]
        if len(pts):
            raw.append(np.mean(pts, axis=0))
            # Deterministic representatives spanning the posterior.
            for idx in np.linspace(0, len(pts) - 1, min(29, len(pts)), dtype=int):
                raw.append(pts[int(idx)])
        unique: list[np.ndarray] = []
        for q in raw:
            if not any(np.linalg.norm(q - z) < 1e-6 for z in unique):
                unique.append(q)
            if len(unique) >= limit:
                break
        return unique


def _route_length(points: np.ndarray, start: np.ndarray, seq: list[int]) -> float:
    if not seq:
        return 0.0
    ordered = points[np.asarray(seq)]
    return float(np.linalg.norm(ordered[0] - start) + np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())


def route_open_multistart(points: np.ndarray, start: Iterable[float], max_starts: int = 8) -> list[int]:
    """Open NN+2-opt routes with up to eight deterministic forced first nodes."""
    pts = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    if len(pts) <= 1:
        return list(range(len(pts)))
    nearest = np.argsort(np.linalg.norm(pts - start, axis=1), kind="stable")[:max_starts]
    candidates: list[list[int]] = [route_open(pts, start)]
    # Force each selected first node by solving the remainder from that node.
    for first in nearest:
        rem = [i for i in range(len(pts)) if i != int(first)]
        tail_local = route_open(pts[rem], pts[int(first)]) if rem else []
        candidates.append([int(first)] + [rem[i] for i in tail_local])
    return min(candidates, key=lambda seq: (_route_length(pts, start, seq), tuple(seq)))


@dataclass(frozen=True)
class ISRV2Config:
    soft_probe: bool = True
    multistart_route: bool = True
    scan_reprice: bool = True
    q4_signal_model: bool = True
    max_soft_actions: int = 32
    max_plan_total_s: float = 10.0
    max_plan_call_s: float = 0.25


class ISRV2Policy(JointShell):
    """Candidate policy; every soft decision has a deterministic ISR fallback."""

    def __init__(self, env, config: ISRV2Config | None = None):
        self.config = config or ISRV2Config()
        # Disable the v1 center probe only when the Q3 soft probe replaces it.
        super().__init__(env, threshold=0, probe=not (env.mode == 3 and self.config.soft_probe), max_fast_steps=100)
        self.max_region_radius = 800.0 if env.mode == 3 else 300.0
        self.soft: dict[int, SoftSourceBelief] = {}
        self.probe_counts: dict[int, int] = {}
        self.decision_log: list[dict] = []
        self.soft_actions = 0
        self.plan_wall_s = 0.0
        self.soft_disabled_reason = ""

    def _guarded(self, fn, fallback):
        if self.soft_disabled_reason:
            return fallback
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        self.plan_wall_s += dt
        if dt > self.config.max_plan_call_s or self.plan_wall_s > self.config.max_plan_total_s:
            self.soft_disabled_reason = "per_call_budget" if dt > self.config.max_plan_call_s else "episode_budget"
            return fallback
        return value

    def observe(self, ch, p, obs):
        super().observe(ch, p, obs)
        typ = obs.get("measure_result")
        if ch not in self.tracks:
            self.soft.pop(ch, None)
            return
        if ch not in self.soft and typ == "direction":
            self.soft[ch] = SoftSourceBelief(self.env.mode, ch, self.tracks[ch]["P"])
        belief = self.soft.get(ch)
        if belief is None:
            return
        if typ == "direction":
            belief.update_direction(p, float(obs["svd_deg"]))
        elif typ == "no_signal":
            belief.update_no_signal(p)

    def _probe_choice(self, ch: int):
        st = self.tracks[ch]
        hard_center, hard_radius = center_radius(st["P"])
        belief = self.soft.get(ch)
        if belief is None or not belief.active or hard_radius > 80.0:
            return None
        candidates = belief.clear_candidates(hard_center, self.env.pos)
        scored = []
        # Conservative continuation estimate by hard-region size.  It is only a
        # ranking baseline; insufficient expected gain suppresses the probe.
        continuation = 35.0 + 0.18 * hard_radius + 0.03 * float(np.linalg.norm(hard_center - self.env.pos))
        for q in candidates:
            p_hit = belief.hit_fraction(q)
            travel = float(np.linalg.norm(q - self.env.pos)) / 5.0
            expected = travel + p_hit * 5.0 + (1.0 - p_hit) * (3.0 + continuation)
            gain = continuation - expected
            scored.append((expected, -p_hit, float(q[0]), float(q[1]), q, p_hit, gain, travel))
        if not scored:
            return None
        best = min(scored)
        return best if best[6] > 1.0 and best[5] >= 0.20 else None

    def resolve(self, ch):
        if self.env.mode == 3 and self.config.soft_probe and not self.soft_disabled_reason:
            while self.probe_counts.get(ch, 0) < 2 and self.soft_actions < self.config.max_soft_actions:
                choice = self._guarded(lambda: self._probe_choice(ch), None)
                if choice is None:
                    break
                _, _, _, _, q, p_hit, gain, travel = choice
                self.probe_counts[ch] = self.probe_counts.get(ch, 0) + 1
                self.soft_actions += 1
                out = self.env.clear(q, ch)
                row = {"event": "early_clear", "channel": ch, "candidate": [float(q[0]), float(q[1])],
                       "estimated_hit_fraction": p_hit, "estimated_gain_s": gain,
                       "travel_cost_s": travel, "feedback": out.get("clear_result")}
                self.decision_log.append(row)
                if out.get("clear_result") == "success":
                    self.tracks.pop(ch, None)
                    self.soft.pop(ch, None)
                    return
                belief = self.soft.get(ch)
                if belief is not None:
                    belief.update_failed_clear(q)
            # Parent has try_clear=False in this configuration and proceeds to
            # the same bilateral/certified fallback path as ISR-NoProbe.
        super().resolve(ch)
        self.soft.pop(ch, None)

    def _job_order(self, jobs, coords):
        points = np.asarray(coords, dtype=float)
        fallback = route_open(points, self.env.pos)
        seq = fallback
        if self.config.multistart_route and fallback and jobs[fallback[0]][0] == "scan":
            # Keep ISR-v1's scan-vs-service decision.  Optimize only which scan
            # is visited next; mixing resolve nodes into an open-tour estimate
            # caused a small mean gain but unacceptable tail detours in v1.
            scan_indices = [i for i, job in enumerate(jobs) if job[0] == "scan"]
            scan_points = points[np.asarray(scan_indices)]
            scan_default = route_open(scan_points, self.env.pos)
            scan_seq = self._guarded(lambda: route_open_multistart(scan_points, self.env.pos), scan_default)
            mapped = [scan_indices[i] for i in scan_seq]
            seq = mapped + [i for i in fallback if i not in set(mapped)]
        if not self.config.scan_reprice or self.soft_disabled_reason or len(seq) <= 1:
            return seq
        def priced():
            # Reprice only nearby scan alternatives and keep the original
            # scan-vs-service decision.  P_sig can break a local route choice;
            # it cannot justify a long detour or postpone a resolve action.
            if jobs[seq[0]][0] != "scan":
                return seq
            base_travel = float(np.linalg.norm(points[seq[0]] - self.env.pos))
            choices = [i for i in seq[:8] if jobs[i][0] == "scan"
                       and float(np.linalg.norm(points[i] - self.env.pos)) <= base_travel + 100.0]
            scored = []
            base_value = sum(b.p_signal(points[seq[0]]) for b in self.soft.values())
            for rank, i in enumerate(choices):
                travel = float(np.linalg.norm(points[i] - self.env.pos)) / 5.0
                value = sum(b.p_signal(points[i]) for b in self.soft.values()) if self.config.q4_signal_model else 0.0
                scored.append((-value, travel, rank, i))
            if not scored:
                return seq
            best = min(scored)
            first = best[3] if -best[0] >= base_value + 0.15 else seq[0]
            return [first] + [i for i in seq if i != first]
        return self._guarded(priced, seq)

    def run(self):
        env = self.env
        unvisited = set(range(len(self.points)))
        self.scan(0)
        unvisited.remove(0)
        outer_first = env.mode == 4 and env.discovered_count() <= self.threshold
        steps = 0
        while not env.done and steps < self.max_fast_steps:
            steps += 1
            if env.cleared_count() + env.discovered_count() >= 16:
                unvisited.clear()
            else:
                unvisited = {j for j in unvisited if any(s.status == "unknown" and j not in s.scan_points for s in env.channels.values())}
            outer = {j for j in unvisited if j >= 13}
            active = outer if outer_first and outer else unvisited
            jobs = [("scan", j) for j in sorted(active)]
            coords = [self.points[j] for _, j in jobs]
            for ch, st in self.tracks.items():
                c, r = center_radius(st["P"])
                if not unvisited or r <= self.max_region_radius:
                    jobs.append(("resolve", ch)); coords.append(c)
            if not jobs:
                if self.tracks:
                    chs = list(self.tracks)
                    cs = np.asarray([center_radius(self.tracks[c]["P"])[0] for c in chs])
                    seq = route_open_multistart(cs, env.pos) if self.config.multistart_route else route_open(cs, env.pos)
                    self.resolve(chs[seq[0]])
                    continue
                break
            typ, key = jobs[self._job_order(jobs, coords)[0]]
            if typ == "scan":
                self.scan(key); unvisited.remove(key)
            else:
                self.resolve(key)
        # Exact inherited terminal obligations, including the 100-step fallback.
        if not env.done:
            for j in self.order:
                if any(s.status == "unknown" and j not in s.scan_points for s in env.channels.values()):
                    self.scan(j)
            while self.tracks and not env.done:
                chs = list(self.tracks)
                cs = np.asarray([center_radius(self.tracks[c]["P"])[0] for c in chs])
                self.resolve(chs[route_open(cs, env.pos)[0]])
        return env


class ISRV2Candidate:
    def __init__(self, mode: int, config: ISRV2Config | None = None):
        self.mode = int(mode)
        if self.mode not in (3, 4):
            raise ValueError("mode must be 3 or 4")
        self.config = config or ISRV2Config()

    def run(self, env):
        if int(env.mode) != self.mode:
            raise ValueError("mode mismatch")
        view = CheckedView(env)
        policy = ISRV2Policy(view, self.config)
        view.coverage_points = policy.points.copy()
        view.n_coverage = len(policy.points)
        policy.run()
        success = bool(view.completion_certificate())
        if not success:
            raise RuntimeError("ISRV2 returned without a completion certificate")
        return {"success": success, "cleared": int(view.cleared_count()),
                "virtual_time_s": float(view.virtual_time), "distance_m": float(view.move_distance),
                "measure_calls": int(view.n_measure), "switches": int(view.n_switch),
                "clear_calls": int(view.n_clear), "failed_clear": int(view.n_clear_fail),
                "solver_failures": int(policy.solver_failures),
                "optical_fallbacks": int(policy.solver_failures),
                "soft_actions": int(policy.soft_actions), "planning_wall_s": float(policy.plan_wall_s),
                "soft_disabled_reason": policy.soft_disabled_reason,
                "decision_log": policy.decision_log,
                "candidate": "ISRV2-safe-soft-20260912"}
