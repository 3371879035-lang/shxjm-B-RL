"""保守动态不存在证书：Q3 圆盘并集与 Q4 局部凸包。

只使用 no_signal 观测和题面硬约束；所有证书都是充分条件，不删除任何
未被完整覆盖的单元。
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple
import numpy as np


def circle_rect_min_dist(center: Sequence[float], half: float, radius: float = 1800.0) -> float:
    c = np.asarray(center, dtype=float)
    dx = max(abs(float(c[0])) - float(half), 0.0)
    dy = max(abs(float(c[1])) - float(half), 0.0)
    return math.hypot(dx, dy)


def grid_cells_intersecting_disk(spacing: float = 200.0, radius: float = 1800.0) -> Tuple[np.ndarray, float]:
    """返回与目标圆相交的闭正方形中心与半边长。"""
    half = float(spacing) / 2.0
    k = int(float(radius) / float(spacing)) + 2
    cells = []
    for i in range(-k, k + 1):
        x = i * float(spacing)
        for j in range(-k, k + 1):
            y = j * float(spacing)
            if circle_rect_min_dist((x, y), half, float(radius)) <= float(radius) + 1e-9:
                cells.append((x, y))
    return np.asarray(cells, dtype=float), half


def convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points, dtype=float)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def point_in_convex_polygon_strict(p: Sequence[float], hull: np.ndarray, tol: float = 1e-9) -> bool:
    hull = np.asarray(hull, dtype=float)
    if len(hull) < 3:
        return False
    p = np.asarray(p, dtype=float)
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross <= tol:
            return False
    return True


def _corner_distances(points: np.ndarray, cells: np.ndarray, half: float) -> np.ndarray:
    """每个负测点到每个单元四个角点的最大距离，形状 (n_cells, n_points)。"""
    pts = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if pts.size == 0 or cells.size == 0:
        return np.zeros((len(cells), len(pts)), dtype=float)
    corners = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    cc = cells[:, None, :] + corners[None, :, :]  # (nc,4,2)
    d = np.linalg.norm(pts[None, :, None, :] - cc[:, None, :, :], axis=3)  # (nc,np,4)
    return np.max(d, axis=2)


def q3_absent_cells(neg_points: Sequence[Sequence[float]], cells: np.ndarray, half: float,
                    margin: float = 1e-6) -> np.ndarray:
    """全向源 Q3：单个负测点覆盖单元四角即可排除整个单元。"""
    dmax = _corner_distances(np.asarray(neg_points, dtype=float), cells, half)
    if dmax.size == 0:
        return np.zeros(len(cells), dtype=bool)
    return np.any(dmax <= 1000.0 - margin, axis=1)


def q4_absent_cells(neg_points: Sequence[Sequence[float]], cells: np.ndarray, half: float,
                    margin: float = 1e-6) -> np.ndarray:
    """定向源 Q4：局部凸包证书。

    对单元保留所有到四角最大距离 <= 1000-margin 的负观测点，若它们
    的凸包严格包含单元四角，则任意源位置 g∈B 和任意 180 度发射方向，
    都存在一个负观测点位于发射半平面内且在 1000m 内，与 no_signal
    矛盾。因此单元可排除。
    """
    pts = np.asarray(neg_points, dtype=float)
    cells = np.asarray(cells, dtype=float)
    if len(pts) < 3 or len(cells) == 0:
        return np.zeros(len(cells), dtype=bool)
    dmax = _corner_distances(pts, cells, half)  # (nc,np)
    corners_off = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    out = np.zeros(len(cells), dtype=bool)
    safe_r = 1000.0 - margin
    for i, c in enumerate(cells):
        keep = pts[dmax[i] <= safe_r]
        if len(keep) < 3:
            continue
        hull = convex_hull(keep)
        if len(hull) < 3:
            continue
        corners = c[None, :] + corners_off
        if all(point_in_convex_polygon_strict(q, hull, tol=1e-7) for q in corners):
            out[i] = True
    return out


def cell_index_of_point(p: Sequence[float], cells: np.ndarray, half: float) -> int:
    """返回点是否落在某个已保留单元内；不在任何单元时返回 -1。"""
    if cells is None or len(cells) == 0:
        return -1
    p = np.asarray(p, dtype=float)
    d = cells - p[None, :]
    inside = np.all(np.abs(d) <= half + 1e-9, axis=1)
    if not np.any(inside):
        return -1
    # 边界可能同时落在多个单元，取最近中心
    idx = np.where(inside)[0]
    return int(idx[np.argmin(np.linalg.norm(d[idx], axis=1))])