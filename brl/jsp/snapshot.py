"""Immutable public-only state passed to JSP scorers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FEATURE_VERSION = "jsp-public-relative-v2"


@dataclass(frozen=True)
class PublicTrack:
    channel: int
    center: tuple[float, float]
    radius: float
    observations: int
    pending_locator: bool


@dataclass(frozen=True)
class PublicSnapshot:
    mode: int
    position: tuple[float, float]
    current_channel: int
    cleared: int
    absent: int
    discovered: int
    pending_station_count: int
    pending_measurements: int
    tracks: tuple[PublicTrack, ...]


@dataclass(frozen=True)
class PublicCandidate:
    kind: str
    key: int
    position: tuple[float, float]
    measure_count: int
    baseline: bool
    analytic_score: float


KINDS = ("scan4", "scanall", "locate1", "resolve")
FEATURE_NAMES = (
    "mode_scaled", "travel_unit_x", "travel_unit_y", "travel_dx_scaled",
    "travel_dy_scaled", "travel_distance_scaled", "cleared_fraction",
    "absent_fraction", "discovered_fraction", "pending_station_fraction",
    "pending_measurement_fraction", "active_track_fraction",
    "candidate_measure_fraction", "is_baseline", "analytic_cost_scaled",
    "is_localization", "same_channel_localization", "track_radius_scaled",
    "track_observations_scaled", "kind_scan4", "kind_scanall",
    "kind_locate1", "kind_resolve",
)


def candidate_features(snapshot: PublicSnapshot, candidate: PublicCandidate) -> np.ndarray:
    """Fixed public feature vector; no environment or mutable references."""
    p = np.asarray(snapshot.position, dtype=float)
    q = np.asarray(candidate.position, dtype=float)
    delta = q - p
    distance = float(np.linalg.norm(delta))
    unit = delta / distance if distance > 1e-12 else np.zeros(2, dtype=float)
    track = next((t for t in snapshot.tracks if t.channel == candidate.key), None)
    kind = [1.0 if candidate.kind == name else 0.0 for name in KINDS]
    values = [
        snapshot.mode / 4.0,
        unit[0], unit[1],
        delta[0] / 1800.0, delta[1] / 1800.0,
        distance / 3600.0,
        snapshot.cleared / 16.0, snapshot.absent / 20.0,
        snapshot.discovered / 16.0,
        snapshot.pending_station_count / 25.0,
        snapshot.pending_measurements / 500.0,
        len(snapshot.tracks) / 16.0,
        candidate.measure_count / 20.0,
        1.0 if candidate.baseline else 0.0,
        candidate.analytic_score / 10000.0,
        1.0 if candidate.kind.startswith("locate") or candidate.kind == "resolve" else 0.0,
        1.0 if candidate.kind.startswith("locate") and candidate.key == snapshot.current_channel else 0.0,
        0.0 if track is None else track.radius / 1800.0,
        0.0 if track is None else track.observations / 12.0,
        *kind,
    ]
    out = np.asarray(values, dtype=np.float64)
    if out.shape != (len(FEATURE_NAMES),):
        raise AssertionError("JSP feature schema length mismatch")
    out.setflags(write=False)
    return out


def assert_public_snapshot(value: PublicSnapshot) -> None:
    """Reject accidental live objects before they reach a scorer."""
    if not isinstance(value, PublicSnapshot):
        raise TypeError("JSP scorer requires PublicSnapshot")
    for item in (value.position, value.tracks):
        if not isinstance(item, tuple):
            raise TypeError("public snapshot contains a mutable container")
