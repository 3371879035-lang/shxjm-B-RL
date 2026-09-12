"""JSP: resumable localization and segmented coverage scheduling."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Callable, Optional

import numpy as np

from brl.bilateral_state import BilateralState
from brl.coverage import s25_points, s3_points
from brl.g25o import initial_track_polygon as initial_poly
from brl.g25o import route_open, _poly_center_radius as center_radius
from brl.independent_candidate import CheckedView
from brl.protocol import (ActionIOError, CertificateViolation,
                          GeometryNumericalError)

from .model import JSPModelError, JSPRanker
from .snapshot import (PublicCandidate, PublicSnapshot, PublicTrack,
                       assert_public_snapshot, candidate_features)


@dataclass(frozen=True)
class JSPConfig:
    planner: str = "analytic"
    model_path: Optional[str] = None
    max_decisions: int = 100
    max_interruptions: int = 32
    max_candidates: int = 8
    scan_segment: int = 4
    max_plan_call_s: float = 0.25
    max_plan_total_s: float = 10.0
    learned_min_gain_s: float = 1.0
    detour_limit_m: float = 100.0


@dataclass(frozen=True)
class CandidateAction:
    kind: str
    key: int
    position: tuple[float, float]
    measure_count: int = 0
    baseline: bool = False
    analytic_score: float = math.inf


class JSPPolicy:
    def __init__(self, env, config: JSPConfig | None = None):
        self.env = env
        self.config = config or JSPConfig()
        if self.config.planner not in {"analytic", "learned"}:
            raise ValueError("planner must be analytic or learned")
        self.points = s3_points() if env.mode == 3 else s25_points()
        self.env.coverage_points = self.points.copy()
        self.env.n_coverage = len(self.points)
        self.tracks: dict[int, dict] = {}
        self.locators: dict[int, BilateralState] = {}
        self.probe_attempted: set[int] = set()
        self.decision_log: list[dict] = []
        self.plan_wall_s = 0.0
        self.planning_disabled_reason = ""
        self.decisions = 0
        self.interruptions = 0
        self.optical_fallbacks = 0
        self.outer_first = False
        self.ranker = None
        self._last_selection: dict = {}
        if self.config.planner == "learned":
            if not self.config.model_path:
                self.planning_disabled_reason = "model:model path is missing"
            else:
                try:
                    self.ranker = JSPRanker(self.config.model_path, env.mode)
                except JSPModelError as exc:
                    self.planning_disabled_reason = f"model:{exc}"

    def _pending_channels(self, idx: int) -> list[int]:
        env = self.env
        chs = [c for c, st in env.channels.items()
               if st.status == "unknown" and idx not in st.scan_points]
        return sorted(chs, key=lambda c: (c != env.current_channel, c))

    def _pending_stations(self) -> list[int]:
        pending = [i for i in range(len(self.points)) if self._pending_channels(i)]
        outer = [i for i in pending if i >= 13]
        return outer if self.outer_first and outer else pending

    def _observe_scan(self, ch: int, p: np.ndarray, response: dict) -> None:
        typ = response.get("measure_result")
        if typ == "near":
            out = self.env.clear(p, ch)
            if out.get("clear_result") != "success":
                raise CertificateViolation("near observation contradicted by failed clear")
            self.tracks.pop(ch, None); self.locators.pop(ch, None)
            return
        if typ != "direction":
            return
        deg = float(response["svd_deg"])
        if ch not in self.tracks:
            poly = initial_poly(p, deg)
            self.tracks[ch] = {"first": p.copy(), "deg": deg, "P": poly, "nobs": 1}
            self.locators[ch] = BilateralState(p, deg, self.env.pos,
                                                initial_region=poly)
        else:
            state = self.locators[ch]
            state.incorporate_bearing(p, deg)
            self.tracks[ch]["P"] = state.region.copy()
            self.tracks[ch]["nobs"] += 1

    def _scan(self, idx: int, limit: Optional[int]) -> None:
        p = self.points[idx]
        old_tracks = list(self.tracks)
        chs = self._pending_channels(idx)
        if limit is not None:
            chs = chs[:limit]
        for ch in chs:
            if self.env.done:
                return
            response = self.env.measure(p, ch, coverage_idx=idx)
            self._observe_scan(ch, p, response)
        # Preserve ISR-v1's useful same-station refinement.  These observations
        # are often what keeps Q4 bilateral localization short.
        # Run the opportunistic pass once, after this station's coverage work is
        # complete.  Repeating it after every four-channel segment would measure
        # the same fixed-error position several times without adding evidence.
        if self._pending_channels(idx):
            return
        for ch in old_tracks:
            if ch not in self.tracks or self.env.done:
                continue
            st = self.tracks[ch]
            c, r = center_radius(st["P"])
            if r <= 19.5 or st["nobs"] >= 4:
                continue
            delta = c - p
            d = float(np.linalg.norm(delta))
            w = c - st["first"]
            w0 = float(np.linalg.norm(w))
            if w0 < 1e-9:
                continue
            sine = abs(delta[0] * w[1] - delta[1] * w[0]) / (d * w0 + 1e-9)
            if d <= 1200.0 and sine > 0.15:
                response = self.env.measure(p, ch, is_refine=True)
                self._observe_scan(ch, p, response)

    def _locator_action(self, ch: int):
        c, r = center_radius(self.tracks[ch]["P"])
        if ch not in self.probe_attempted and 19.5 < r <= 80.0:
            return "probe", (float(c[0]), float(c[1]))
        # Candidate generation is a pure preview.  Preparing an action on the
        # live state would cache stale geometry before later scan bearings arrive.
        action = self.locators[ch].preview_action()
        return action.kind, action.position

    def _execute_locator_one(self, ch: int) -> None:
        state = self.locators[ch]
        c, r = center_radius(self.tracks[ch]["P"])
        if ch not in self.probe_attempted and 19.5 < r <= 80.0:
            self.probe_attempted.add(ch)
            response = self.env.clear(c, ch)
            if response.get("clear_result") == "success":
                self.tracks.pop(ch, None); self.locators.pop(ch, None)
            return
        action = state.next_action()
        if action is None:
            return
        p = np.asarray(action.position, dtype=float)
        if action.kind == "measure":
            response = self.env.measure(p, ch, is_refine=True)
        else:
            response = self.env.clear(p, ch)
        state.accept_observation(action.action_id, response)
        self.tracks[ch]["P"] = state.region.copy()
        if state.success:
            self.tracks.pop(ch, None); self.locators.pop(ch, None)

    def _optical_fallback(self, ch: int) -> None:
        self.optical_fallbacks += 1
        st = self.tracks[ch]
        a = math.radians(st["deg"])
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        for x in np.linspace(0, 1500, 61):
            for y in (13.3, -13.3):
                if self.env.done:
                    break
                if self.env.clear(st["first"] + x * u + y * v, ch).get("clear_result") == "success":
                    self.tracks.pop(ch, None); self.locators.pop(ch, None)
                    return
        raise CertificateViolation("certified optical fallback failed")

    def _resolve(self, ch: int) -> None:
        try:
            while ch in self.locators and not self.env.done:
                self._execute_locator_one(ch)
        except ActionIOError:
            raise
        except GeometryNumericalError:
            self._optical_fallback(ch)

    @staticmethod
    def _switches(current: int, channels: list[int]) -> int:
        switches = 0
        last = current
        for channel in channels:
            if channel != last:
                switches += 1
            last = channel
        return switches

    def public_snapshot(self) -> PublicSnapshot:
        pending = self._pending_stations()
        tracks = []
        for ch in sorted(self.tracks):
            c, r = center_radius(self.tracks[ch]["P"])
            tracks.append(PublicTrack(ch, (float(c[0]), float(c[1])), float(r),
                                      int(self.tracks[ch]["nobs"]), ch in self.locators))
        snap = PublicSnapshot(
            mode=int(self.env.mode), position=(float(self.env.pos[0]), float(self.env.pos[1])),
            current_channel=int(self.env.current_channel),
            cleared=int(self.env.cleared_count()),
            absent=sum(1 for st in self.env.channels.values() if st.status == "absent"),
            discovered=int(self.env.discovered_count()),
            pending_station_count=len(pending),
            pending_measurements=sum(len(self._pending_channels(i)) for i in pending),
            tracks=tuple(tracks))
        assert_public_snapshot(snap)
        return snap

    def _baseline_choice(self, scan_indices: list[int]) -> tuple[str, int]:
        jobs: list[tuple[str, int]] = [("scan", i) for i in scan_indices]
        coords = [self.points[i] for i in scan_indices]
        for ch in sorted(self.tracks):
            c, r = center_radius(self.tracks[ch]["P"])
            threshold = 800.0 if self.env.mode == 3 else 300.0
            if not scan_indices or r <= threshold:
                jobs.append(("resolve", ch)); coords.append(c)
        if not jobs:
            ch = min(self.tracks, key=lambda c: np.linalg.norm(
                center_radius(self.tracks[c]["P"])[0] - self.env.pos))
            return "resolve", ch
        seq = route_open(np.asarray(coords), self.env.pos)
        return jobs[seq[0]]

    def _analytic(self, candidate: CandidateAction,
                  scan_indices: list[int]) -> float:
        p = np.asarray(candidate.position)
        travel = float(np.linalg.norm(p - self.env.pos)) / 5.0
        fixed = 0.0
        if candidate.kind.startswith("scan"):
            channels = self._pending_channels(candidate.key)[:candidate.measure_count]
            fixed = 5.0 * len(channels) + self._switches(self.env.current_channel, channels)
        else:
            action_kind, _ = self._locator_action(candidate.key)
            fixed = ((3.0 if action_kind == "probe" else 5.0)
                     + (1.0 if action_kind == "measure" and candidate.key != self.env.current_channel else 0.0))
            if candidate.kind == "resolve":
                state = self.locators[candidate.key]
                width = max(0.0, state.hi - state.lo)
                remaining = max(0, min(7 - state.rounds,
                                       int(math.ceil(math.log2(max(width, 24.0) / 24.0)))))
                fixed += remaining * 10.0 + 8.0
        # Open-route estimate from the candidate endpoint over remaining station
        # and next localization points.  It values actual bilateral probes rather
        # than polygon centers.
        # Candidate keys are type-dependent: scan keys are station indices,
        # while locate/resolve keys are channel numbers.  A localization action
        # must not accidentally remove a same-numbered pending station.
        completed_station = candidate.key if candidate.kind in {"scan4", "scanall"} else None
        remaining_points = [self.points[i] for i in scan_indices if i != completed_station]
        if (candidate.kind == "scan4"
                and len(self._pending_channels(candidate.key)) > candidate.measure_count):
            # A segment does not complete the station.  Omitting its return leg
            # made the first prototype bounce between tasks and underpriced the
            # largest source of extra movement.
            remaining_points.append(self.points[candidate.key])
        for ch in sorted(self.locators):
            if ch == candidate.key and candidate.kind == "resolve":
                continue
            _, action_position = self._locator_action(ch)
            remaining_points.append(np.asarray(action_position, dtype=float))
        route_cost = 0.0
        if remaining_points:
            pts = np.asarray(remaining_points)
            seq = route_open(pts, p)
            ordered = pts[seq]
            route_cost = float(np.linalg.norm(ordered[0] - p))
            if len(ordered) > 1:
                route_cost += float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
            route_cost /= 5.0
        return travel + fixed + route_cost

    def candidates(self) -> list[CandidateAction]:
        stations = self._pending_stations()
        nearest = sorted(stations, key=lambda i: (round(float(np.linalg.norm(
            self.points[i] - self.env.pos)), 9), i))[:8]
        base_kind, base_key = self._baseline_choice(stations)
        raw: list[CandidateAction] = []
        for idx in nearest:
            n = len(self._pending_channels(idx))
            raw.append(CandidateAction("scan4", idx,
                                       tuple(float(x) for x in self.points[idx]),
                                       min(self.config.scan_segment, n)))
            raw.append(CandidateAction("scanall", idx,
                                       tuple(float(x) for x in self.points[idx]), n))
        for ch in sorted(self.locators):
            action_kind, action_position = self._locator_action(ch)
            if action_position is not None:
                # Every discovered source contributes both interruptible and
                # continuous actions.  Geometry and travel affect ranking, not
                # membership in the legal action pool.
                raw.append(CandidateAction("locate1", ch, action_position))
                raw.append(CandidateAction("resolve", ch, action_position))
        # The frozen ISR-v1 next task is mandatory even when route_open's
        # optimized first node is outside the nearest-eight shortlist.
        if base_kind == "scan" and not any(c.kind == "scanall" and c.key == base_key for c in raw):
            n = len(self._pending_channels(base_key))
            raw.append(CandidateAction("scanall", base_key,
                                       tuple(float(x) for x in self.points[base_key]), n))
        if base_kind == "resolve" and not any(c.kind == "resolve" and c.key == base_key for c in raw):
            _, action_position = self._locator_action(base_key)
            raw.append(CandidateAction("resolve", base_key, action_position))
        scored = []
        for c in raw[:64]:
            baseline = ((base_kind == "scan" and c.kind == "scanall" and c.key == base_key)
                        or (base_kind == "resolve" and c.kind == "resolve" and c.key == base_key))
            scored.append(replace(c, baseline=baseline,
                                  analytic_score=self._analytic(c, stations)))
        unique: dict[tuple, CandidateAction] = {}
        for c in scored:
            key = (c.kind, c.key, round(c.position[0], 6), round(c.position[1], 6))
            old = unique.get(key)
            if old is None or c.analytic_score < old.analytic_score:
                unique[key] = c
        values = list(unique.values())
        baseline = [c for c in values if c.baseline]
        analytic = min(values, key=lambda c: (c.analytic_score, c.kind, c.key)) if values else None
        selected: list[CandidateAction] = []
        for c in baseline + ([analytic] if analytic else []):
            if c is not None and c not in selected:
                selected.append(c)
        for kind in ("scan4", "scanall", "locate1", "resolve"):
            group = [c for c in values if c.kind == kind]
            if group:
                c = min(group, key=lambda z: (z.analytic_score, z.key))
                if c not in selected:
                    selected.append(c)
        for c in sorted(values, key=lambda z: (z.analytic_score, z.kind, z.key)):
            if c not in selected:
                selected.append(c)
            if len(selected) >= self.config.max_candidates:
                break
        return selected[:self.config.max_candidates]

    @staticmethod
    def _public_candidate(c: CandidateAction) -> PublicCandidate:
        return PublicCandidate(c.kind, c.key, c.position, c.measure_count,
                               c.baseline, c.analytic_score)

    def choose(self, candidates: list[CandidateAction]) -> CandidateAction:
        baseline = next((c for c in candidates if c.baseline), None)
        analytic = min(candidates, key=lambda c: (c.analytic_score, c.kind, c.key))
        if self.config.planner != "learned" or self.ranker is None:
            if baseline is None or baseline.kind == "resolve":
                choice = baseline or analytic
                self._last_selection = {
                    "reason": "baseline_resolution" if baseline is not None else "analytic_minimum",
                    "predicted_gain_s": None, "learned_scores_s": None,
                }
                return choice
            # The geometric candidate is mandatory eventually.  Insert only a
            # next localization probe that lies almost on the path to ISR-v1's
            # next coverage station; otherwise preserve the frozen choice.
            b = np.asarray(baseline.position)
            direct = float(np.linalg.norm(b - self.env.pos))
            detours = []
            for c in candidates:
                if c.kind != "locate1":
                    continue
                q = np.asarray(c.position)
                detour = (float(np.linalg.norm(q - self.env.pos))
                          + float(np.linalg.norm(b - q)) - direct)
                if detour <= self.config.detour_limit_m:
                    detours.append((detour, c.key, c))
            if detours:
                detour, _, choice = min(detours)
                self._last_selection = {"reason": "bounded_interruption",
                                        "detour_m": float(detour),
                                        "predicted_gain_s": None,
                                        "learned_scores_s": None}
                return choice
            self._last_selection = {"reason": "baseline_guard",
                                    "predicted_gain_s": None,
                                    "learned_scores_s": None}
            return baseline
        snap = self.public_snapshot()
        rows = np.vstack([candidate_features(snap, self._public_candidate(c)) for c in candidates])
        try:
            pred = self.ranker.predict(rows, timeout_s=self.config.max_plan_call_s)
        except JSPModelError as exc:
            self.planning_disabled_reason = f"model:{exc}"
            self._last_selection = {"reason": "model_fallback", "error": str(exc),
                                    "predicted_gain_s": None, "learned_scores_s": None}
            return baseline or analytic
        best_idx = int(np.argmin(pred))
        base_idx = next((i for i, c in enumerate(candidates) if c.baseline), None)
        if base_idx is not None and pred[base_idx] - pred[best_idx] < self.config.learned_min_gain_s:
            self._last_selection = {
                "reason": "learned_gain_below_threshold",
                "predicted_gain_s": float(pred[base_idx] - pred[best_idx]),
                "learned_scores_s": pred.tolist(),
            }
            return candidates[base_idx]
        self._last_selection = {
            "reason": "learned_predicted_gain",
            "predicted_gain_s": (None if base_idx is None
                                  else float(pred[base_idx] - pred[best_idx])),
            "learned_scores_s": pred.tolist(),
        }
        return candidates[best_idx]

    def _state_for_log(self) -> dict:
        locators = {}
        for ch, state in self.locators.items():
            locators[str(ch)] = {
                "phase": state.phase, "round": int(state.rounds),
                "pending_probe": int(state.pending_probe),
                "lo": float(state.lo), "hi": float(state.hi),
                "clear_attempts": int(state.clear_attempts),
            }
        return {
            "position": [float(self.env.pos[0]), float(self.env.pos[1])],
            "virtual_time_s": float(self.env.virtual_time),
            "distance_m": float(self.env.move_distance),
            "measure_calls": int(self.env.n_measure), "switches": int(self.env.n_switch),
            "clear_calls": int(self.env.n_clear), "failed_clear": int(self.env.n_clear_fail),
            "cleared": int(self.env.cleared_count()),
            "observation_counts": {str(ch): len(st.observations)
                                   for ch, st in self.env.channels.items()},
            "last_results": {str(ch): st.last_result for ch, st in self.env.channels.items()
                             if st.last_result is not None},
            "locators": locators,
        }

    def execute(self, candidate: CandidateAction) -> None:
        if candidate.kind == "scan4":
            self._scan(candidate.key, candidate.measure_count)
            self.interruptions += 1
        elif candidate.kind == "scanall":
            self._scan(candidate.key, None)
        elif candidate.kind == "locate1":
            self._execute_locator_one(candidate.key)
            self.interruptions += 1
        elif candidate.kind == "resolve":
            self._resolve(candidate.key)
        else:
            raise ValueError(candidate.kind)

    def finish_deterministic(self) -> None:
        # Preserve every pending S3/S25 channel obligation, then consume every
        # discovered source from its current resumable state.
        while not self.env.done:
            pending = self._pending_stations()
            if not pending:
                break
            pts = self.points[np.asarray(pending)]
            idx = pending[route_open(pts, self.env.pos)[0]]
            self._scan(idx, None)
        while self.locators and not self.env.done:
            ch = min(self.locators, key=lambda c: float(np.linalg.norm(
                np.asarray(self._locator_action(c)[1]) - self.env.pos)))
            self._resolve(ch)

    def resume_baseline(self) -> None:
        """Continue the frozen ISR scheduling rule from the current JSP state."""
        while not self.env.done:
            stations = self._pending_stations()
            if not stations and not self.locators:
                break
            kind, key = self._baseline_choice(stations)
            if kind == "scan":
                self._scan(key, None)
            else:
                self._resolve(key)

    def run(self, decision_hook: Optional[Callable] = None):
        # ISR-v1 always performs the zero-distance origin survey before enabling
        # Q4's outward-first schedule.
        if self._pending_channels(0):
            self._scan(0, None)
        self.outer_first = bool(self.env.mode == 4 and self.env.discovered_count() <= 0)
        while not self.env.done and self.decisions < self.config.max_decisions:
            forced = []
            for ch in sorted(self.locators):
                action_kind, position = self._locator_action(ch)
                if action_kind == "clear" and position is not None:
                    forced.append((float(np.linalg.norm(
                        np.asarray(position) - self.env.pos)), ch, position))
            if forced:
                _, ch, position = min(forced)
                before = self._state_for_log()
                self._execute_locator_one(ch)
                after = self._state_for_log()
                self.decision_log.append({
                    "decision": self.decisions, "kind": "forced_clear", "key": ch,
                    "position": list(position), "baseline": False,
                    "analytic_score": 0.0,
                    "selection": {"reason": "deterministic_clear_priority",
                                  "predicted_gain_s": None, "learned_scores_s": None},
                    "candidates": [], "feedback": [], "before": before, "after": after,
                    "delta_time_s": float(after["virtual_time_s"] - before["virtual_time_s"]),
                    "delta_distance_m": float(after["distance_m"] - before["distance_m"]),
                    "delta_measure_calls": 0,
                    "delta_switches": int(after["switches"] - before["switches"]),
                    "delta_clear_calls": int(after["clear_calls"] - before["clear_calls"]),
                    "delta_failed_clear": int(after["failed_clear"] - before["failed_clear"]),
                })
                continue
            if self.interruptions >= self.config.max_interruptions or self.planning_disabled_reason:
                break
            started = time.perf_counter()
            candidates = self.candidates()
            if not candidates:
                break
            choice = self.choose(candidates)
            elapsed = time.perf_counter() - started
            self.plan_wall_s += elapsed
            if elapsed > self.config.max_plan_call_s or self.plan_wall_s > self.config.max_plan_total_s:
                self.planning_disabled_reason = ("per_call_budget" if elapsed > self.config.max_plan_call_s
                                                 else "episode_budget")
                break
            if decision_hook is not None:
                decision_hook(self, candidates, choice)
            self.decisions += 1
            before = self._state_for_log()
            self.execute(choice)
            after = self._state_for_log()
            changed_feedback = []
            for channel, count in after["observation_counts"].items():
                if count != before["observation_counts"].get(channel, 0):
                    changed_feedback.append({
                        "channel": int(channel), "new_observations": int(
                            count - before["observation_counts"].get(channel, 0)),
                        "last_result": after["last_results"].get(channel),
                    })
            learned = self._last_selection.get("learned_scores_s")
            self.decision_log.append({
                "decision": self.decisions, "kind": choice.kind, "key": choice.key,
                "position": list(choice.position), "analytic_score": choice.analytic_score,
                "baseline": choice.baseline,
                "selection": self._last_selection,
                "candidates": [{
                    "kind": c.kind, "key": c.key, "position": list(c.position),
                    "measure_count": c.measure_count, "baseline": c.baseline,
                    "analytic_score": float(c.analytic_score),
                    "learned_score_s": (None if learned is None else float(learned[i])),
                } for i, c in enumerate(candidates)],
                "feedback": changed_feedback,
                "before": before, "after": after,
                "delta_time_s": float(after["virtual_time_s"] - before["virtual_time_s"]),
                "delta_distance_m": float(after["distance_m"] - before["distance_m"]),
                "delta_measure_calls": int(after["measure_calls"] - before["measure_calls"]),
                "delta_switches": int(after["switches"] - before["switches"]),
                "delta_clear_calls": int(after["clear_calls"] - before["clear_calls"]),
                "delta_failed_clear": int(after["failed_clear"] - before["failed_clear"]),
            })
        if not self.env.done:
            # Continue the frozen ISR task rule from the exact current state.
            # This preserves all accepted observations, half-finished bilateral
            # rounds, current channel and accrued cost when a planning budget or
            # model guard disables new scheduling.
            self.resume_baseline()
        return self.env


class JSPCandidate:
    def __init__(self, mode: int, planner: str = "analytic",
                 model_path: Optional[str] = None, config: JSPConfig | None = None):
        self.mode = int(mode)
        if self.mode not in (3, 4):
            raise ValueError("mode must be 3 or 4")
        self.config = config or JSPConfig(planner=planner, model_path=model_path)

    def run(self, env, decision_hook: Optional[Callable] = None) -> dict:
        if int(env.mode) != self.mode:
            raise ValueError("mode mismatch")
        view = CheckedView(env)
        policy = JSPPolicy(view, self.config)
        policy.run(decision_hook=decision_hook)
        success = bool(view.completion_certificate())
        if not success:
            raise CertificateViolation("JSP returned without a completion certificate")
        return {
            "success": success, "cleared": int(view.cleared_count()),
            "virtual_time_s": float(view.virtual_time), "distance_m": float(view.move_distance),
            "measure_calls": int(view.n_measure), "switches": int(view.n_switch),
            "clear_calls": int(view.n_clear), "failed_clear": int(view.n_clear_fail),
            "planning_wall_s": float(policy.plan_wall_s),
            "planning_disabled_reason": policy.planning_disabled_reason,
            "decisions": int(policy.decisions), "interruptions": int(policy.interruptions),
            "optical_fallbacks": int(policy.optical_fallbacks),
            "decision_log": policy.decision_log,
            "candidate": f"JSP-{self.config.planner}-20260913",
        }
