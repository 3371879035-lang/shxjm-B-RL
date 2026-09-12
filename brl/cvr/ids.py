from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ChannelId:
    value: int

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 20:
            raise ValueError("channel must be in 1..20")


@dataclass(frozen=True, order=True)
class StationId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("station index must be non-negative")


@dataclass(frozen=True, order=True)
class PlanNodeId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("plan node id must be non-empty")


def require_channel(value: ChannelId) -> ChannelId:
    if not isinstance(value, ChannelId):
        raise TypeError("expected ChannelId")
    return value


def require_station(value: StationId) -> StationId:
    if not isinstance(value, StationId):
        raise TypeError("expected StationId")
    return value
