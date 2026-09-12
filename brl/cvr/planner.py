from __future__ import annotations

from dataclasses import dataclass
import itertools
import time

import numpy as np

from .certify import CertificateResult, CoverageCertificateEngine
from .evidence import CoverageEvidenceLedger, FutureCoveragePlan
from .waypoints import PublicCVRSnapshot, WaypointGenerator


@dataclass(frozen=True)
class SegmentPlannerConfig:
    beam_width: int = 16
    max_depth: int = 4
    max_proposals: int = 96
    min_predicted_gain_s: float = 1.0
    max_call_s: float = 0.5
    max_total_s: float = 12.0


@dataclass(frozen=True)
class MacroTask:
    kind: str
    key: str
    position: tuple[float, float]
    channels: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlanMutation:
    proposal_id: str
    plan_before: FutureCoveragePlan
    plan_after: FutureCoveragePlan


@dataclass(frozen=True)
class PlannedSegment:
    tasks: tuple[MacroTask, ...]
    mutations: tuple[PlanMutation, ...]
    certificate: CertificateResult
    channel_certificates: tuple[tuple[int, CertificateResult], ...]
    estimated_cost_s: float
    baseline_cost_s: float
    reason: str


class SegmentPlanner:
    def __init__(
        self,
        engine: CoverageCertificateEngine,
        generator: WaypointGenerator,
        config: SegmentPlannerConfig | None = None,
    ) -> None:
        self.engine = engine
        self.generator = generator
        self.config = config or SegmentPlannerConfig()
        self.total_wall_s = 0.0
        self.last_proposal_count = 0

    @staticmethod
    def _switches(current: int, channels: tuple[int, ...]) -> int:
        total, last = 0, current
        for channel in channels:
            total += int(channel != last)
            last = channel
        return total

    def segment_cost(
        self,
        snapshot: PublicCVRSnapshot,
        tasks: tuple[MacroTask, ...],
        exit_position: tuple[float, float],
    ) -> float:
        position = np.asarray(snapshot.position, dtype=float)
        channel = snapshot.current_channel
        total = 0.0
        for task in tasks:
            target = np.asarray(task.position, dtype=float)
            total += float(np.linalg.norm(target - position)) / 5.0
            if task.kind == "measure":
                total += 5.0 * len(task.channels) + self._switches(channel, task.channels)
                if task.channels:
                    channel = task.channels[-1]
            position = target
        total += float(np.linalg.norm(np.asarray(exit_position, dtype=float) - position)) / 5.0
        return total

    @staticmethod
    def _ordered_channels(current: int, channels) -> tuple[int, ...]:
        values = sorted(channel.value for channel in channels)
        if current in values:
            values.remove(current)
            values.insert(0, current)
        return tuple(values)

    def _tasks(
        self,
        snapshot: PublicCVRSnapshot,
        plan: FutureCoveragePlan,
        forced_first_node: str | None = None,
    ) -> tuple[MacroTask, ...]:
        current = np.asarray(snapshot.position, dtype=float)
        ordered = sorted(
            plan.nodes,
            key=lambda node: (
                float(np.linalg.norm(np.asarray(node.position, dtype=float) - current)),
                node.node_id.value,
            ),
        )
        if forced_first_node is not None:
            forced = next((node for node in ordered if node.node_id.value == forced_first_node), None)
            if forced is not None:
                ordered.remove(forced)
                ordered.insert(0, forced)
        ordered = ordered[: self.config.max_depth]
        return tuple(
            MacroTask(
                "measure",
                node.node_id.value,
                node.position,
                self._ordered_channels(snapshot.current_channel, node.channels),
            )
            for node in ordered
        )

    def _best_order(
        self,
        snapshot: PublicCVRSnapshot,
        tasks: tuple[MacroTask, ...],
        exit_position: tuple[float, float],
    ) -> tuple[tuple[MacroTask, ...], float]:
        if len(tasks) <= 1:
            return tasks, self.segment_cost(snapshot, tasks, exit_position)
        scored = []
        for order in itertools.permutations(tasks):
            scored.append((self.segment_cost(snapshot, order, exit_position), tuple(task.key for task in order), order))
        cost, _, order = min(scored, key=lambda item: (item[0], item[1]))
        return tuple(order), float(cost)

    def _certificates(
        self,
        plan: FutureCoveragePlan,
        ledger: CoverageEvidenceLedger,
    ) -> tuple[tuple[int, CertificateResult], ...]:
        channels = sorted({channel for node in plan.nodes for channel in node.channels})
        results = []
        cache: dict[tuple[tuple[float, float], ...], CertificateResult] = {}
        for channel in channels:
            points = ledger.negative_points(channel) + tuple(
                node.position for node in plan.nodes if channel in node.channels
            )
            key = tuple((round(float(x), 9), round(float(y), 9)) for x, y in points)
            result = cache.get(key)
            if result is None:
                result = self.engine.certify(points)
                cache[key] = result
            results.append((channel.value, result))
        return tuple(results)

    def _aggregate(
        self, results: tuple[tuple[int, CertificateResult], ...]
    ) -> CertificateResult:
        if not results:
            return CertificateResult(True, self.engine.mode, -1000.0, 0, "no_unknown_channels")
        worst = max((result for _, result in results), key=lambda result: result.max_gap_m)
        return CertificateResult(
            all(result.covered for _, result in results),
            self.engine.mode,
            worst.max_gap_m,
            sum(result.witness_count for _, result in results),
            "covered" if all(result.covered for _, result in results) else "channel_geometric_gap",
        )

    def _baseline(
        self,
        snapshot: PublicCVRSnapshot,
        ledger: CoverageEvidenceLedger,
        exit_position: tuple[float, float],
        reason: str,
        forced_first_node: str | None = None,
    ) -> PlannedSegment:
        tasks = self._tasks(snapshot, snapshot.plan, forced_first_node)
        tasks, cost = self._best_order(snapshot, tasks, exit_position)
        certificates = self._certificates(snapshot.plan, ledger)
        return PlannedSegment(tasks, (), self._aggregate(certificates), certificates, cost, cost, reason)

    def plan(
        self,
        snapshot: PublicCVRSnapshot,
        ledger: CoverageEvidenceLedger,
        exit_position: tuple[float, float],
        *,
        allow_replacements: bool = True,
        forced_first_node: str | None = None,
    ) -> PlannedSegment:
        started = time.perf_counter()
        if self.config.max_call_s <= 0.0 or self.total_wall_s >= self.config.max_total_s:
            return self._baseline(snapshot, ledger, exit_position, "planning_budget_fallback", forced_first_node)
        baseline = self._baseline(snapshot, ledger, exit_position, "frozen_baseline", forced_first_node)
        if not allow_replacements:
            self.last_proposal_count = 0
            elapsed = time.perf_counter() - started
            self.total_wall_s += elapsed
            return baseline

        chosen = baseline
        proposals = self.generator.replacement_proposals(snapshot)[: self.config.max_proposals]
        if forced_first_node is not None:
            proposals = tuple(
                proposal
                for proposal in proposals
                if any(node.value == forced_first_node for node in proposal.removed)
            )
        self.last_proposal_count = len(proposals)
        for proposal in proposals:
            if time.perf_counter() - started > self.config.max_call_s:
                elapsed = time.perf_counter() - started
                self.total_wall_s += elapsed
                return self._baseline(snapshot, ledger, exit_position, "planning_budget_fallback", forced_first_node)
            changed = snapshot.plan.replace(proposal.removed, proposal.added)
            certificates = self._certificates(changed, ledger)
            aggregate = self._aggregate(certificates)
            if not aggregate.covered:
                continue
            replacement_first = proposal.added[0].node_id.value
            tasks = self._tasks(snapshot, changed, replacement_first)
            tasks, cost = self._best_order(snapshot, tasks, exit_position)
            if cost <= baseline.baseline_cost_s - self.config.min_predicted_gain_s:
                mutation = PlanMutation(proposal.proposal_id, snapshot.plan, changed)
                candidate = PlannedSegment(
                    tasks,
                    (mutation,),
                    aggregate,
                    certificates,
                    cost,
                    baseline.baseline_cost_s,
                    "certified_replacement",
                )
                if (candidate.estimated_cost_s, tuple(task.key for task in candidate.tasks)) < (
                    chosen.estimated_cost_s,
                    tuple(task.key for task in chosen.tasks),
                ):
                    chosen = candidate
        elapsed = time.perf_counter() - started
        self.total_wall_s += elapsed
        if self.total_wall_s > self.config.max_total_s:
            return self._baseline(snapshot, ledger, exit_position, "planning_budget_fallback", forced_first_node)
        return chosen
