from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evidence import CoverageNode, FutureCoveragePlan
from .ids import ChannelId, PlanNodeId


@dataclass(frozen=True)
class PublicCVRTrack:
    channel: ChannelId
    center: tuple[float, float]
    radius: float
    next_kind: str
    next_position: tuple[float, float]


@dataclass(frozen=True)
class PublicCVRSnapshot:
    mode: int
    position: tuple[float, float]
    current_channel: int
    tracks: tuple[PublicCVRTrack, ...]
    plan: FutureCoveragePlan


@dataclass(frozen=True)
class ReplacementProposal:
    proposal_id: str
    removed: tuple[PlanNodeId, ...]
    added: tuple[CoverageNode, ...]


def _inside_domain(point: np.ndarray) -> tuple[float, float]:
    radius = float(np.linalg.norm(point))
    # Keep a millimetre of numerical slack because candidate coordinates are
    # serialized to six decimals before certification and replay.
    if radius > 1799.999:
        point = point * (1799.999 / radius)
    return float(point[0]), float(point[1])


class WaypointGenerator:
    """Generate a bounded, deterministic set from public policy state only."""

    def generate(self, snapshot: PublicCVRSnapshot) -> tuple[tuple[float, float], ...]:
        points: list[tuple[float, float]] = []
        current = np.asarray(snapshot.position, dtype=float)
        pending = tuple(snapshot.plan.nodes)

        # Fixed witnesses are legal candidates too. Including them ensures the
        # proposal space always contains a no-op geometric reference.
        for node in pending[:8]:
            points.append(node.position)

        for track in snapshot.tracks:
            center = np.asarray(track.center, dtype=float)
            next_position = np.asarray(track.next_position, dtype=float)
            points.extend((_inside_domain(center), _inside_domain(next_position)))
            direction = center - current
            length = float(np.linalg.norm(direction))
            if length > 1e-9:
                forward = direction / length
                lateral = np.array([-forward[1], forward[0]])
                for fraction in (0.35, 0.55, 0.75, 1.0):
                    for offset in (-300.0, 0.0, 300.0):
                        points.append(_inside_domain(current + fraction * direction + offset * lateral))

            # Interpolate between a pending coverage witness and the next hard
            # locator point. These points are where search and localization can
            # share physical travel if the certificate accepts the replacement.
            for node in pending[:4]:
                fixed = np.asarray(node.position, dtype=float)
                for fraction in (0.25, 0.5, 0.75):
                    points.append(_inside_domain((1.0 - fraction) * fixed + fraction * next_position))

        return tuple(
            sorted(
                {(round(x, 6), round(y, 6)) for x, y in points},
                key=lambda point: (point[0], point[1]),
            )
        )

    def replacement_proposals(
        self, snapshot: PublicCVRSnapshot
    ) -> tuple[ReplacementProposal, ...]:
        candidates = self.generate(snapshot)
        nodes = snapshot.plan.nodes
        proposals: list[ReplacementProposal] = []
        serial = 0

        # The closest pending witnesses are the only plausible short-horizon
        # substitutions. Generate replace-1 and consecutive replace-2 options,
        # then let the independent certificate reject unsafe geometry.
        current = np.asarray(snapshot.position, dtype=float)
        ordered = tuple(
            sorted(nodes, key=lambda node: (float(np.linalg.norm(np.asarray(node.position) - current)), node.node_id.value))
        )
        blocks: list[tuple[CoverageNode, ...]] = [(node,) for node in ordered[:8]]
        blocks.extend((ordered[i], ordered[i + 1]) for i in range(min(7, max(0, len(ordered) - 1))))

        for block in blocks:
            removed = tuple(node.node_id for node in block)
            channels = frozenset(channel for node in block for channel in node.channels)
            for point in candidates:
                if len(proposals) >= 96:
                    return tuple(proposals)
                serial += 1
                added = CoverageNode(
                    PlanNodeId(f"var-v{snapshot.plan.version}-{serial}"),
                    point,
                    channels,
                    None,
                )
                proposals.append(
                    ReplacementProposal(f"replace-{serial}", removed, (added,))
                )
        return tuple(proposals)
