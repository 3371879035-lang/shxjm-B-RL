from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, Voronoi

DOMAIN_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0


@dataclass(frozen=True)
class CertificateResult:
    covered: bool
    mode: int
    max_gap_m: float
    witness_count: int
    reason: str


def _nearest_distance(point: np.ndarray, points: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points - point[None, :], axis=1)))


def _boundary_candidate_angles(points: np.ndarray) -> list[float]:
    # A reference boundary point is required for rotationally symmetric and
    # otherwise degenerate point sets (for example, the origin alone).
    angles: list[float] = [0.0]
    for point in points:
        if np.linalg.norm(point) > 1e-12:
            angles.append(math.atan2(float(point[1]), float(point[0])) + math.pi)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            normal = points[j] - points[i]
            rhs = (float(points[j] @ points[j]) - float(points[i] @ points[i])) / 2.0
            a, b = float(normal[0]), float(normal[1])
            amplitude = math.hypot(a, b) * DOMAIN_RADIUS
            if amplitude <= 1e-12 or abs(rhs) > amplitude + 1e-9:
                continue
            base = math.atan2(b, a)
            offset = math.acos(max(-1.0, min(1.0, rhs / amplitude)))
            angles.extend((base - offset, base + offset))
    return angles


def q3_required_radius(points: np.ndarray) -> float:
    candidates: list[np.ndarray] = [np.zeros(2)]
    unique = np.unique(points, axis=0)
    if len(unique) >= 3:
        try:
            voronoi = Voronoi(unique)
            candidates.extend(
                vertex
                for vertex in voronoi.vertices
                if np.linalg.norm(vertex) <= DOMAIN_RADIUS + 1e-9
            )
        except QhullError:
            pass
    for angle in _boundary_candidate_angles(unique):
        candidates.append(DOMAIN_RADIUS * np.array([math.cos(angle), math.sin(angle)]))
    return max(_nearest_distance(np.asarray(candidate), unique) for candidate in candidates)


def _hull_inradius_at_origin(hull: ConvexHull) -> float:
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    if np.any(offsets > 1e-9):
        return -math.inf
    return float(np.min(-offsets / np.linalg.norm(normals, axis=1)))


def _origin_to_segment(a: np.ndarray, b: np.ndarray) -> float:
    edge = b - a
    denominator = float(edge @ edge)
    if denominator <= 1e-18:
        return float(np.linalg.norm(a))
    t = float(np.clip(float(-a @ edge) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(a + t * edge))


def _triangle_intersects_domain(vertices: np.ndarray) -> bool:
    if np.any(np.linalg.norm(vertices, axis=1) <= DOMAIN_RADIUS + 1e-9):
        return True
    crosses = []
    for i in range(3):
        a, b = vertices[i], vertices[(i + 1) % 3]
        edge = b - a
        crosses.append(float(edge[0] * (-a[1]) - edge[1] * (-a[0])))
    origin_inside = all(value >= -1e-9 for value in crosses) or all(
        value <= 1e-9 for value in crosses
    )
    if origin_inside:
        return True
    return min(
        _origin_to_segment(vertices[i], vertices[(i + 1) % 3]) for i in range(3)
    ) <= DOMAIN_RADIUS + 1e-9


def q4_max_relevant_edge(points: np.ndarray) -> tuple[float, bool, int]:
    hull = ConvexHull(points)
    if _hull_inradius_at_origin(hull) < DOMAIN_RADIUS - 1e-9:
        return math.inf, False, 0
    triangulation = Delaunay(points)
    maximum = 0.0
    relevant = 0
    for simplex in triangulation.simplices:
        vertices = points[simplex]
        if not _triangle_intersects_domain(vertices):
            continue
        relevant += 1
        maximum = max(
            maximum,
            max(
                float(np.linalg.norm(vertices[i] - vertices[j]))
                for i in range(3)
                for j in range(i)
            ),
        )
    return maximum, True, relevant


class CoverageCertificateEngine:
    def __init__(self, mode: int, margin_m: float = 1e-6):
        if mode not in (3, 4):
            raise ValueError("mode must be 3 or 4")
        self.mode = mode
        self.margin_m = float(margin_m)

    def certify(self, points: Iterable[tuple[float, float]]) -> CertificateResult:
        array = np.asarray(tuple(points), dtype=float)
        if array.ndim != 2 or array.shape[1:] != (2,) or len(array) == 0:
            raise ValueError("certificate points must be a non-empty 2D coordinate array")
        if not np.isfinite(array).all():
            raise ValueError("certificate points must be finite 2D coordinates")
        array = np.unique(array, axis=0)
        try:
            if self.mode == 3:
                required = q3_required_radius(array)
                witness_count = len(array)
            else:
                required, hull_ok, witness_count = q4_max_relevant_edge(array)
                if not hull_ok:
                    return CertificateResult(
                        False, self.mode, math.inf, witness_count, "domain_not_in_hull"
                    )
            covered = required <= MIN_RECEIVE_RADIUS - self.margin_m
            return CertificateResult(
                covered,
                self.mode,
                float(required - MIN_RECEIVE_RADIUS),
                witness_count,
                "covered" if covered else "geometric_gap",
            )
        except QhullError:
            return CertificateResult(False, self.mode, math.inf, len(array), "degenerate_hull")
