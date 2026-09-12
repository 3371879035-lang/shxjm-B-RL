from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .ids import ChannelId, PlanNodeId, StationId, require_channel

Point = tuple[float, float]


@dataclass(frozen=True)
class NegativeObservation:
    channel: ChannelId
    position: Point
    request_id: str


@dataclass(frozen=True)
class CoverageNode:
    node_id: PlanNodeId
    position: Point
    channels: frozenset[ChannelId]
    legacy_station: StationId | None

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise ValueError("coverage-node position must be 2D")
        if any(not isinstance(channel, ChannelId) for channel in self.channels):
            raise TypeError("coverage-node channels must be ChannelId values")

    def without(self, channel: ChannelId) -> "CoverageNode":
        require_channel(channel)
        return replace(self, channels=frozenset(c for c in self.channels if c != channel))


@dataclass(frozen=True)
class FutureCoveragePlan:
    version: int
    nodes: tuple[CoverageNode, ...]

    def __post_init__(self) -> None:
        ids = tuple(node.node_id for node in self.nodes)
        if len(ids) != len(set(ids)):
            raise ValueError("plan node ids must be unique")
        if self.version < 1:
            raise ValueError("plan version must be positive")

    @classmethod
    def initial(cls, nodes: Iterable[CoverageNode]) -> "FutureCoveragePlan":
        return cls(1, tuple(nodes))

    def replace(
        self,
        removed: tuple[PlanNodeId, ...],
        added: tuple[CoverageNode, ...],
    ) -> "FutureCoveragePlan":
        remove_set = frozenset(removed)
        existing = frozenset(node.node_id for node in self.nodes)
        if not remove_set or len(remove_set) != len(removed) or not remove_set.issubset(existing):
            raise ValueError("replacement must name unique pending plan nodes")
        kept = tuple(node for node in self.nodes if node.node_id not in remove_set)
        return FutureCoveragePlan(self.version + 1, kept + tuple(added))

    def remove_channel(self, channel: ChannelId) -> "FutureCoveragePlan":
        require_channel(channel)
        return FutureCoveragePlan(
            self.version + 1,
            tuple(node.without(channel) for node in self.nodes if node.without(channel).channels),
        )

    def complete_node_channel(self, node_id: PlanNodeId, channel: ChannelId) -> "FutureCoveragePlan":
        require_channel(channel)
        if node_id not in {node.node_id for node in self.nodes}:
            raise ValueError("completed node is not pending")
        updated = []
        for node in self.nodes:
            changed = node.without(channel) if node.node_id == node_id else node
            if changed.channels:
                updated.append(changed)
        return FutureCoveragePlan(self.version + 1, tuple(updated))


class CoverageEvidenceLedger:
    def __init__(self) -> None:
        self._negative: dict[ChannelId, list[NegativeObservation]] = {}
        self._request_ids: set[str] = set()

    def accept(
        self,
        channel: ChannelId,
        position: Point,
        result: str,
        request_id: str,
        *,
        accepted: bool,
    ) -> bool:
        require_channel(channel)
        if not accepted or result != "no_signal" or request_id in self._request_ids:
            return False
        if not request_id:
            raise ValueError("request id must be non-empty")
        self._request_ids.add(request_id)
        observation = NegativeObservation(channel, tuple(map(float, position)), request_id)
        self._negative.setdefault(channel, []).append(observation)
        return True

    def negative_points(self, channel: ChannelId) -> tuple[Point, ...]:
        require_channel(channel)
        return tuple(observation.position for observation in self._negative.get(channel, ()))

    def snapshot(self) -> tuple[NegativeObservation, ...]:
        return tuple(
            observation
            for channel in sorted(self._negative)
            for observation in self._negative[channel]
        )
