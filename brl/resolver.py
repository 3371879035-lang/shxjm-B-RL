"""单源定位、第二测点与有限步骤光学保底。"""
from __future__ import annotations

from math import cos, pi, sin
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (HalfPlane, conservative_clear_certificate, intersect_halfplanes,
                       minimum_enclosing_circle, perpendicular, polygon_centroid,
                       point_in_convex_polygon, unit_from_deg, wedge_halfplanes)


def second_point_candidates(s1: Sequence[float], theta_deg: float,
                            forward: float = 750.0, side: float = 300.0) -> np.ndarray:
    """第2问的可靠接收候选点 q± = s1 + forward*u ± side*v。"""
    s1 = np.asarray(s1, dtype=float)
    u = unit_from_deg(theta_deg)
    v = perpendicular(u)
    return np.vstack([s1 + forward * u + side * v, s1 + forward * u - side * v])


def optical_fallback_points(s: Sequence[float], theta_deg: float,
                            n_half: int = 60, step: float = 25.0, offset: float = 13.1) -> np.ndarray:
    """一次有效示向后，覆盖窄矩形的最多 2*(n_half+1) 个清除点。

    返回按蛇形顺序排列的点。
    """
    s = np.asarray(s, dtype=float)
    u = unit_from_deg(theta_deg)
    v = perpendicular(u)
    plus = []
    minus = []
    for k in range(n_half + 1):
        plus.append(s + k * step * u + offset * v)
        minus.append(s + k * step * u - offset * v)
    # 蛇形：a0+ a0- a1- a1+ a2+ a2- ...
    out = []
    for k in range(n_half + 1):
        if k % 2 == 0:
            out.append(plus[k]); out.append(minus[k])
        else:
            out.append(minus[k]); out.append(plus[k])
    return np.asarray(out, dtype=float)


def refine_candidates(observations: Sequence[dict], poly: np.ndarray,
                      current_pos: Sequence[float], max_candidates: int = 3) -> List[np.ndarray]:
    """给出继续交会的测点。

    对尚未失联的源使用第2问的 q± 与小侧偏点；
    一旦出现 no_signal（可能为定向源背向），把沿首测示向的前向点提前，
    以尽量保持在可见半平面内。
    """
    cands: List[np.ndarray] = []
    first_dir = None
    first_dir_idx = -1
    for i, ob in enumerate(observations):
        if ob.get("result") == "direction" and "svd_deg" in ob:
            first_dir = ob
            first_dir_idx = i
            break
    robust = []
    short_lateral = []
    forward_visible = []
    if first_dir is not None:
        s1 = np.asarray(first_dir["position"], dtype=float)
        th = float(first_dir["svd_deg"])
        u = unit_from_deg(th)
        v = perpendicular(u)
        robust = [s1 + 750.0 * u + 300.0 * v,
                  s1 + 750.0 * u - 300.0 * v]
        short_lateral = [s1 + 375.0 * u + 150.0 * v,
                         s1 + 375.0 * u - 150.0 * v]
        offsets = [
            (300.0, 0.0),
            (500.0, 100.0),
            (500.0, -100.0),
            (700.0, 0.0),
            (700.0, 180.0),
            (700.0, -180.0),
            (900.0, 0.0),
            (900.0, 260.0),
            (900.0, -260.0),
            (1100.0, 0.0),
        ]
        for f, side in offsets:
            forward_visible.append(s1 + f * u + side * v)
    centroid_fallback = []
    if poly is not None and len(poly) >= 1:
        c = polygon_centroid(poly)
        d = c - np.asarray(current_pos, dtype=float)
        nd = np.linalg.norm(d)
        if nd > 1.0:
            d = d / nd
            v = perpendicular(d)
            centroid_fallback.append(c - 250.0 * d + 200.0 * v)
            centroid_fallback.append(c - 250.0 * d - 200.0 * v)
    ns_after = 0
    if first_dir is not None:
        for ob in observations[first_dir_idx + 1:]:
            if ob.get("result") == "no_signal":
                ns_after += 1
    if ns_after == 0:
        cands.extend(robust)
        cands.extend(short_lateral)
        cands.extend(centroid_fallback)
        cands.extend(forward_visible)
    else:
        cands.extend(forward_visible)
        cands.extend(short_lateral)
        cands.extend(robust)
        cands.extend(centroid_fallback)
    # 去重、排除已测位置
    obs_pos = [np.asarray(o["position"], dtype=float) for o in observations]
    out = []
    for p in cands:
        if any(np.linalg.norm(p - q) < 1.0 for q in obs_pos):
            continue
        if any(np.linalg.norm(p - q) < 1.0 for q in out):
            continue
        out.append(p)
        if len(out) >= max_candidates:
            break
    return out


def reliable_clear(poly: np.ndarray, margin: float = 19.5) -> Tuple[bool, Optional[np.ndarray], float]:
    return conservative_clear_certificate(poly, margin=margin)


def estimate_visible_direction(poly: np.ndarray, detector_pos: Sequence[float]) -> Optional[np.ndarray]:
    """辅助用：从可行域估计指向源的期望方向。"""
    if poly is None or len(poly) == 0:
        return None
    c = polygon_centroid(poly)
    d = c - np.asarray(detector_pos, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return d / n


def bearing_fim(q: Sequence[float], z: Sequence[float], sigma_deg: float = 1.0) -> np.ndarray:
    """单次方位测量的 Fisher 信息矩阵（2x2）。

    方位角 beta=atan2(z_y-q_y,z_x-q_x)，对目标位置 z 的雅可比为
    J=[-sin beta/r, cos beta/r]，FIM = J^T J / sigma_rad^2。
    """
    q = np.asarray(q, dtype=float)
    z = np.asarray(z, dtype=float)
    d = z - q
    r = float(np.linalg.norm(d))
    if r < 1e-9:
        return np.zeros((2, 2), dtype=float)
    beta = np.arctan2(d[1], d[0])
    J = np.array([[-np.sin(beta) / r, np.cos(beta) / r]], dtype=float)
    sigma = max(np.deg2rad(float(sigma_deg)), 1e-12)
    return (J.T @ J) / (sigma ** 2)


def expected_fim_det(q: Sequence[float], poly: np.ndarray,
                     n_samples: int = 64, seed: int = 0) -> float:
    """在当前可行域上估计方位 FIM 行列式，用于第二测点/局部测点评分。"""
    from .geometry import sample_convex_polygon
    rng = np.random.default_rng(seed)
    pts = sample_convex_polygon(poly, n_samples, rng)
    if len(pts) == 0:
        return 0.0
    vals = [float(np.linalg.det(bearing_fim(q, z))) for z in pts]
    return float(np.mean(vals))
