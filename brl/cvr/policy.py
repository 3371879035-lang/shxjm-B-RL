from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from brl.coverage import s25_points, s3_points
from brl.g25o import _poly_center_radius as center_radius
from brl.g25o import route_open
from brl.independent_candidate import CheckedView
from brl.jsp.policy import JSPPolicy
from brl.protocol import CertificateViolation

from .certify import CoverageCertificateEngine
from .evidence import CoverageEvidenceLedger, CoverageNode, FutureCoveragePlan
from .ids import ChannelId, PlanNodeId, StationId
from .planner import PlannedSegment, SegmentPlanner, SegmentPlannerConfig
from .waypoints import PublicCVRSnapshot, PublicCVRTrack, WaypointGenerator


@dataclass(frozen=True)
class CVRConfig:
    planner_enabled: bool = True
    variable_waypoints: bool = True
    max_plan_call_s: float = 0.5
    max_plan_total_s: float = 12.0
    max_segments: int = 100
    max_segment_depth: int = 1


class CVRPolicy:
    """Execute certified search mutations while reusing hard localization state."""

    def __init__(self, env, config: CVRConfig | None = None) -> None:
        self.env = env
        self.config = config or CVRConfig()
        self.executor = JSPPolicy(env)
        points = s3_points() if env.mode == 3 else s25_points()
        all_channels = frozenset(ChannelId(channel) for channel in range(1, 21))
        self.plan = FutureCoveragePlan.initial(
            tuple(
                CoverageNode(
                    PlanNodeId(f"fixed-{index}"),
                    tuple(map(float, point)),
                    all_channels,
                    StationId(index),
                )
                for index, point in enumerate(points)
            )
        )
        self.ledger = CoverageEvidenceLedger()
        self.certified_absent: set[ChannelId] = set()
        planner_config = SegmentPlannerConfig(
            max_depth=self.config.max_segment_depth,
            max_call_s=self.config.max_plan_call_s,
            max_total_s=self.config.max_plan_total_s,
        )
        self.engine = CoverageCertificateEngine(env.mode)
        self.planner = SegmentPlanner(self.engine, WaypointGenerator(), planner_config)
        self.events: list[dict] = [
            {
                "event": "plan_initial",
                "mode": int(env.mode),
                "plan_version": self.plan.version,
                "nodes": [
                    {
                        "node_id": node.node_id.value,
                        "position": list(node.position),
                        "channels": [channel.value for channel in sorted(node.channels)],
                        "legacy_station": node.legacy_station.value,
                    }
                    for node in self.plan.nodes
                ],
            }
        ]
        self.fallback_reason = ""
        self._evidence_serial = 0
        self.replans = 0
        self.replacement_candidates = 0
        self.certified_replacements = 0
        self.executed_replacements = 0
        self.extra_measurements = 0
        self.initial_fixed_nodes = len(points)
        self.removed_legacy_stations: set[int] = set()
        self.outer_first = False

    def _complete(self) -> bool:
        if self.env.cleared_count() >= 16:
            return True
        return all(
            self.env.channels[channel].status in {"cleared", "absent"}
            or ChannelId(channel) in self.certified_absent
            for channel in range(1, 21)
        )

    def _public_snapshot(self) -> PublicCVRSnapshot:
        tracks = []
        for channel in sorted(self.executor.tracks):
            center, radius = center_radius(self.executor.tracks[channel]["P"])
            next_kind, next_position = self.executor._locator_action(channel)
            if next_position is None:
                next_position = center
            tracks.append(
                PublicCVRTrack(
                    ChannelId(channel),
                    (float(center[0]), float(center[1])),
                    float(radius),
                    str(next_kind),
                    tuple(map(float, next_position)),
                )
            )
        return PublicCVRSnapshot(
            int(self.env.mode),
            (float(self.env.pos[0]), float(self.env.pos[1])),
            int(self.env.current_channel),
            tuple(tracks),
            self.plan,
        )

    def _remove_discovered_channel(self, channel: ChannelId) -> None:
        if any(channel in node.channels for node in self.plan.nodes):
            self.plan = self.plan.remove_channel(channel)

    def _refresh_absence(self, channel: ChannelId) -> None:
        if self.env.channels[channel.value].status != "unknown":
            return
        points = self.ledger.negative_points(channel)
        result = self.engine.certify(points)
        if result.covered:
            self.certified_absent.add(channel)
            self._remove_discovered_channel(channel)
            self.events.append(
                {
                    "event": "certified_absent",
                    "channel": channel.value,
                    "accepted_negative_points": [list(point) for point in points],
                    "future_points": [],
                    "mode": int(self.env.mode),
                    "plan_version": self.plan.version,
                    "certificate": {
                        "covered": result.covered,
                        "max_gap_m": result.max_gap_m,
                        "reason": result.reason,
                    },
                }
            )

    def _measure_node(self, node: CoverageNode, channel: ChannelId) -> None:
        self._evidence_serial += 1
        evidence_id = f"cvr-observation-{self._evidence_serial}"
        coverage_index = None if node.legacy_station is None else node.legacy_station.value
        response = self.env.measure(node.position, channel.value, coverage_idx=coverage_index)
        point = np.asarray(node.position, dtype=float)
        self.executor._observe_scan(channel.value, point, response)

        result = str(response["measure_result"])
        accepted = response.get("accepted") is True
        self.ledger.accept(
            channel,
            node.position,
            result,
            evidence_id,
            accepted=accepted,
        )
        # The executed obligation disappears from this node. A discovered or
        # cleared channel disappears from the entire future absence plan.
        if node.node_id in {pending.node_id for pending in self.plan.nodes}:
            self.plan = self.plan.complete_node_channel(node.node_id, channel)
        if result in {"direction", "near"} or self.env.channels[channel.value].status == "cleared":
            self._remove_discovered_channel(channel)

        if node.legacy_station is None:
            self.extra_measurements += 1
        self.events.append(
            {
                "event": "measurement",
                "request_id": evidence_id,
                "node_id": node.node_id.value,
                "channel": channel.value,
                "position": list(node.position),
                "legacy_station": None if node.legacy_station is None else node.legacy_station.value,
                "result": result,
                "accepted": accepted,
                "plan_version": self.plan.version,
            }
        )
        if result == "no_signal":
            self._refresh_absence(channel)

    def _apply_segment(self, segment: PlannedSegment) -> None:
        if segment.mutations:
            mutation = segment.mutations[0]
            remaining_ids = {node.node_id for node in mutation.plan_after.nodes}
            removed_fixed = {
                node.legacy_station.value
                for node in mutation.plan_before.nodes
                if node.node_id not in remaining_ids and node.legacy_station is not None
            }
            self.removed_legacy_stations.update(removed_fixed)
            self.events.append(
                {
                    "event": "plan_mutation",
                    "proposal_id": mutation.proposal_id,
                    "plan_version_before": mutation.plan_before.version,
                    "plan_version": mutation.plan_after.version,
                    "removed": [node.node_id.value for node in mutation.plan_before.nodes if node.node_id not in {n.node_id for n in mutation.plan_after.nodes}],
                    "added": [
                        {
                            "node_id": node.node_id.value,
                            "position": list(node.position),
                            "channels": [channel.value for channel in sorted(node.channels)],
                        }
                        for node in mutation.plan_after.nodes
                        if node.node_id not in {n.node_id for n in mutation.plan_before.nodes}
                    ],
                    "channel_certificates": [
                        {
                            "channel": channel,
                            "covered": result.covered,
                            "max_gap_m": result.max_gap_m,
                            "reason": result.reason,
                        }
                        for channel, result in segment.channel_certificates
                    ],
                }
            )
            self.plan = mutation.plan_after
            self.certified_replacements += 1

        completed_variable = False
        for task in segment.tasks:
            node = next((pending for pending in self.plan.nodes if pending.node_id.value == task.key), None)
            if node is None:
                continue
            old_tracks = tuple(self.executor.tracks)
            for raw_channel in task.channels:
                channel = ChannelId(raw_channel)
                if self.env.channels[raw_channel].status != "unknown":
                    self._remove_discovered_channel(channel)
                    continue
                current = next((pending for pending in self.plan.nodes if pending.node_id == node.node_id), None)
                if current is None or channel not in current.channels:
                    continue
                self._measure_node(current, channel)
                if self._complete():
                    break
            if node.legacy_station is None:
                completed_variable = True
            if not self._complete():
                self._refine_at_completed_node(old_tracks, np.asarray(node.position, dtype=float))
            if self._complete():
                break
        if segment.mutations and completed_variable:
            self.executed_replacements += 1

    def _refine_at_completed_node(self, old_tracks: tuple[int, ...], point: np.ndarray) -> None:
        """Preserve ISR's useful one-shot bearing refinement at a scan stop."""
        for channel in old_tracks:
            if channel not in self.executor.tracks or self._complete():
                continue
            track = self.executor.tracks[channel]
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
                distance * first_length + 1e-9
            )
            if distance <= 1200.0 and sine > 0.15:
                response = self.env.measure(point, channel, is_refine=True)
                self.executor._observe_scan(channel, point, response)

    def _next_hard_job(self) -> tuple[str, str | int] | None:
        pending = tuple(self.plan.nodes)
        active = pending
        if self.outer_first:
            outer = tuple(
                node
                for node in pending
                if node.legacy_station is not None and node.legacy_station.value >= 13
            )
            if outer:
                active = outer
        jobs: list[tuple[str, str | int]] = [("scan", node.node_id.value) for node in active]
        positions = [node.position for node in active]
        threshold = 800.0 if self.env.mode == 3 else 300.0
        for channel, track in self.executor.tracks.items():
            center, radius = center_radius(track["P"])
            if not pending or radius <= threshold:
                jobs.append(("resolve", channel))
                positions.append(tuple(map(float, center)))
        if not jobs:
            return None
        index = route_open(np.asarray(positions, dtype=float), self.env.pos)[0]
        return jobs[index]

    def _execute_one_locator(self) -> None:
        if not self.executor.locators:
            return
        forced = []
        regular = []
        for channel in sorted(self.executor.locators):
            kind, position = self.executor._locator_action(channel)
            if position is None:
                continue
            distance = float(np.linalg.norm(np.asarray(position, dtype=float) - self.env.pos))
            row = (distance, channel, kind)
            (forced if kind == "clear" else regular).append(row)
        options = forced or regular
        if not options:
            return
        _, channel, kind = min(options)
        before = float(self.env.virtual_time)
        self.executor._execute_locator_one(channel)
        self.events.append(
            {
                "event": "locator_action",
                "channel": channel,
                "kind": kind,
                "delta_time_s": float(self.env.virtual_time - before),
                "plan_version": self.plan.version,
            }
        )

    def _fallback(self, reason: str) -> None:
        self.fallback_reason = reason
        self.events.append(
            {"event": "fallback", "reason": reason, "plan_version": self.plan.version}
        )
        self.executor.resume_baseline()

    def run(self) -> "CVRPolicy":
        # Preserve the comparable zero-distance origin survey.
        origin = next(node for node in self.plan.nodes if node.legacy_station == StationId(0))
        for channel in tuple(sorted(origin.channels)):
            if self.env.channels[channel.value].status == "unknown":
                current = next((node for node in self.plan.nodes if node.node_id == origin.node_id), None)
                if current is None:
                    break
                self._measure_node(current, channel)
            if self._complete():
                return self

        self.outer_first = bool(self.env.mode == 4 and self.env.discovered_count() <= 0)

        if not self.config.planner_enabled:
            self._fallback("planner_disabled")
            return self

        for _ in range(self.config.max_segments):
            if self._complete():
                return self
            if not self.plan.nodes:
                break
            hard_job = self._next_hard_job()
            if hard_job is None:
                break
            if hard_job[0] == "resolve":
                channel = int(hard_job[1])
                self.executor._resolve(channel)
                self._remove_discovered_channel(ChannelId(channel))
                continue
            snapshot = self._public_snapshot()
            if snapshot.tracks:
                exit_position = min(
                    (track.next_position for track in snapshot.tracks),
                    key=lambda point: float(np.linalg.norm(np.asarray(point) - self.env.pos)),
                )
            else:
                exit_position = snapshot.position
            started = time.perf_counter()
            segment = self.planner.plan(
                snapshot,
                self.ledger,
                exit_position,
                allow_replacements=self.config.variable_waypoints,
                forced_first_node=str(hard_job[1]),
            )
            elapsed = time.perf_counter() - started
            self.replans += 1
            self.replacement_candidates += int(self.planner.last_proposal_count)
            self.events.append(
                {
                    "event": "segment",
                    "reason": segment.reason,
                    "estimated_cost_s": segment.estimated_cost_s,
                    "baseline_cost_s": segment.baseline_cost_s,
                    "planning_wall_s": elapsed,
                    "plan_version": self.plan.version,
                }
            )
            if segment.reason == "planning_budget_fallback":
                reason = "per_call_budget" if elapsed > self.config.max_plan_call_s else "episode_budget"
                self._fallback(reason)
                return self
            self._apply_segment(segment)

        if not self._complete():
            self._fallback("segment_budget" if self.plan.nodes else "unfinished_localization")
        return self


class CVRCandidate:
    def __init__(self, mode: int, config: CVRConfig | None = None) -> None:
        self.mode = int(mode)
        if self.mode not in (3, 4):
            raise ValueError("mode must be 3 or 4")
        self.config = config or CVRConfig()

    def run(self, env, decision_hook: Callable | None = None) -> dict:
        if int(env.mode) != self.mode:
            raise ValueError("mode mismatch")
        view = CheckedView(env)
        policy = CVRPolicy(view, self.config)
        policy.run()
        success = bool(policy._complete())
        if not success:
            raise CertificateViolation("CVR returned without a completion certificate")
        return {
            "success": success,
            "cleared": int(view.cleared_count()),
            "virtual_time_s": float(view.virtual_time),
            "distance_m": float(view.move_distance),
            "measure_calls": int(view.n_measure),
            "switches": int(view.n_switch),
            "clear_calls": int(view.n_clear),
            "failed_clear": int(view.n_clear_fail),
            "planning_wall_s": float(policy.planner.total_wall_s),
            "fallback_reason": policy.fallback_reason,
            "replans": int(policy.replans),
            "replacement_candidates": int(policy.replacement_candidates),
            "certified_replacements": int(policy.certified_replacements),
            "executed_replacements": int(policy.executed_replacements),
            "restored_legacy_stations": int(
                len(policy.removed_legacy_stations) if policy.fallback_reason else 0
            ),
            "physical_stations_removed": int(len(policy.removed_legacy_stations)),
            "extra_measurements": int(policy.extra_measurements),
            "optical_fallbacks": int(policy.executor.optical_fallbacks),
            "plan_versions": int(policy.plan.version),
            "decision_log": policy.events,
            "candidate": "CVR-analytic-20260913",
        }
