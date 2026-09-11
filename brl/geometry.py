"""保守几何定位核心模块。

所有多边形均为凸多边形，顺序为逆时针（Sutherland-Hodgman 保持输入方向）。
本模块只使用题目硬约束：目标圆域、最大有效接收半径、示向误差界。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, pi, tan
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9
DOMAIN_RADIUS = 1800.0
MAX_RECEIVE_RADIUS = 1500.0
SVD_ERROR_DEG = 1.0


@dataclass(frozen=True)
class HalfPlane:
    """a*x + b*y <= c."""
    a: float
    b: float
    c: float

    def normalized(self) -> "HalfPlane":
        n = (self.a * self.a + self.b * self.b) ** 0.5
        if n < EPS:
            raise ValueError("退化半平面")
        return HalfPlane(self.a / n, self.b / n, self.c / n)

    def signed_distance(self, p: np.ndarray) -> float:
        return float(self.a * p[0] + self.b * p[1] - self.c)


def unit_from_deg(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    return np.array([cos(t), sin(t)], dtype=float)


def perpendicular(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


def wedge_halfplanes(s: Sequence[float], theta_deg: float, eps_deg: float = SVD_ERROR_DEG) -> List[HalfPlane]:
    """由检测点 s 与示向度 theta 构造误差楔形的半平面表示。

    u^T (z-s) >= 0
    |v^T (z-s)| <= tan(eps) u^T (z-s)
    """
    s = np.asarray(s, dtype=float)
    u = unit_from_deg(theta_deg)
    v = perpendicular(u)
    t = tan(np.deg2rad(eps_deg))
    # forward: u^T z >= u^T s  =>  (-u)^T z <= -u^T s
    hps = [HalfPlane(-u[0], -u[1], -float(np.dot(u, s)))]
    # v^T (z-s) <= t u^T(z-s)
    n1 = v - t * u
    hps.append(HalfPlane(n1[0], n1[1], float(np.dot(n1, s))))
    # -v^T(z-s) <= t u^T(z-s)
    n2 = -v - t * u
    hps.append(HalfPlane(n2[0], n2[1], float(np.dot(n2, s))))
    return [hp.normalized() for hp in hps]


def circle_outer_polygon(center: Sequence[float], radius: float, n: int = 96) -> np.ndarray:
    """圆的外切正 n 边形（包含圆），用于保守近似圆形约束。"""
    c = np.asarray(center, dtype=float)
    k = np.arange(n, dtype=float)
    # 外切正 n 边形顶点半径 r / cos(pi/n)
    R = radius / cos(pi / n)
    ang = 2.0 * pi * k / n
    return c[None, :] + R * np.column_stack([np.cos(ang), np.sin(ang)])


def circle_outer_halfplanes(center: Sequence[float], radius: float, n: int = 96) -> List[HalfPlane]:
    """圆的外切正 n 边形约束，作为半平面集合。"""
    c = np.asarray(center, dtype=float)
    hps: List[HalfPlane] = []
    for k in range(n):
        ang = 2.0 * pi * k / n
        normal = np.array([cos(ang), sin(ang)])
        hps.append(HalfPlane(normal[0], normal[1], float(np.dot(normal, c) + radius)).normalized())
    return hps


def domain_polygon(radius: float = DOMAIN_RADIUS, n: int = 128) -> np.ndarray:
    return circle_outer_polygon((0.0, 0.0), radius, n=n)


def poly_signed_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def clip_convex_polygon(poly: np.ndarray, hp: HalfPlane) -> np.ndarray:
    """Sutherland-Hodgman 单半平面裁剪。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    hp = hp.normalized()
    n = len(poly)
    if n == 1:
        return poly if hp.signed_distance(poly[0]) <= EPS else np.empty((0, 2), dtype=float)
    out: List[np.ndarray] = []
    for i in range(n):
        p = poly[i]
        q = poly[(i + 1) % n]
        fp = hp.signed_distance(p)
        fq = hp.signed_distance(q)
        pin = fp <= EPS
        qin = fq <= EPS
        if pin:
            out.append(p)
        if pin != qin:
            denom = fp - fq
            if abs(denom) > 1e-15:
                t = fp / denom
                out.append(p + t * (q - p))
    if not out:
        return np.empty((0, 2), dtype=float)
    # 去重
    res = []
    for p in out:
        if not res or np.linalg.norm(p - res[-1]) > 1e-9:
            res.append(p)
    if len(res) > 1 and np.linalg.norm(res[0] - res[-1]) <= 1e-9:
        res.pop()
    return np.asarray(res, dtype=float)


def intersect_halfplanes(halfplanes: Sequence[HalfPlane], initial: Optional[np.ndarray] = None,
                         add_domain: bool = True, domain_radius: float = DOMAIN_RADIUS) -> np.ndarray:
    """求半平面交。默认先与目标圆域的外切多边形求交，保证有界。

    返回凸多边形顶点；无交集时返回空数组。
    """
    if initial is None:
        if add_domain:
            poly = domain_polygon(domain_radius)
        else:
            # 大框仅用于最后防数值溢出；调用方不应把它误当题目约束。
            B = 1_000_000.0
            poly = np.array([[-B, -B], [B, -B], [B, B], [-B, B]], dtype=float)
    else:
        poly = np.asarray(initial, dtype=float)
    for hp in halfplanes:
        poly = clip_convex_polygon(poly, hp)
        if len(poly) == 0:
            break
    return poly


def polygon_diameter(poly: np.ndarray) -> float:
    """区域直径，第一版枚举顶点对。"""
    if poly is None or len(poly) < 2:
        return 0.0
    pts = np.asarray(poly)
    dmax = 0.0
    n = len(pts)
    for i in range(n):
        d = np.linalg.norm(pts[i + 1:] - pts[i], axis=1)
        if len(d):
            dmax = max(dmax, float(np.max(d)))
    return dmax


def polygon_diameter_rotating_calipers(poly: np.ndarray) -> float:
    """旋转卡壳求凸多边形直径，用于与暴力顶点对结果交叉验证。

    输入多边形应为逆时针凸多边形。算法在凸多边形上维护对踵点，
    时间复杂度 O(n)，n 为顶点数。
    """
    if poly is None or len(poly) < 2:
        return 0.0
    pts = np.asarray(poly, dtype=float)
    n = len(pts)
    if n == 2:
        return float(np.linalg.norm(pts[0] - pts[1]))
    k = 1
    best = 0.0
    for i in range(n):
        if k == i:
            k = (k + 1) % n
        while True:
            nk = (k + 1) % n
            if nk == i:
                break
            cur = float(np.linalg.norm(pts[i] - pts[k]))
            nxt = float(np.linalg.norm(pts[i] - pts[nk]))
            if nxt > cur:
                k = nk
            else:
                break
        best = max(best, float(np.linalg.norm(pts[i] - pts[k])))
    return best


def verify_diameter_crosscheck(poly: np.ndarray, tol: float = 1e-6) -> Tuple[float, float, float]:
    """返回 (暴力直径, 旋转卡壳直径, 绝对差)。用于质量 Gate。"""
    d1 = polygon_diameter(poly)
    d2 = polygon_diameter_rotating_calipers(poly)
    return d1, d2, abs(d1 - d2)


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.zeros(2, dtype=float)
    if len(poly) == 1:
        return poly[0].copy()
    if len(poly) == 2:
        return 0.5 * (poly[0] + poly[1])
    area = poly_signed_area(poly)
    if abs(area) < 1e-12:
        return poly.mean(axis=0)
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return np.array([cx, cy], dtype=float) / (6.0 * area)


def point_in_convex_polygon(p: Sequence[float], poly: np.ndarray, tol: float = 1e-7) -> bool:
    if poly is None or len(poly) == 0:
        return False
    p = np.asarray(p, dtype=float)
    n = len(poly)
    if n == 1:
        return np.linalg.norm(p - poly[0]) <= tol
    if n == 2:
        a, b = poly
        ab = b - a
        t = np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-15)
        if t < -tol or t > 1 + tol:
            return False
        return np.linalg.norm(a + np.clip(t, 0, 1) * ab - p) <= tol
    sign = 0.0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) <= tol:
            continue
        s = 1.0 if cross > 0 else -1.0
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True


# ---------------- 最小包围圆 -----------------
@dataclass
class Circle:
    center: np.ndarray
    radius: float


def circle_from_two(a: np.ndarray, b: np.ndarray) -> Circle:
    c = 0.5 * (a + b)
    return Circle(c, float(np.linalg.norm(a - b) * 0.5))


def circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Circle:
    # 外接圆
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-15:
        # 共线：取最远两点
        pts = [a, b, c]
        best = Circle(a.copy(), 0.0)
        for i in range(3):
            for j in range(i + 1, 3):
                cc = circle_from_two(pts[i], pts[j])
                if cc.radius > best.radius:
                    best = cc
        return best
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    center = np.array([ux, uy], dtype=float)
    return Circle(center, float(np.linalg.norm(center - a)))


def _welzl(points: np.ndarray, boundary: List[np.ndarray], n: int) -> Circle:
    if n == 0 or len(boundary) == 3:
        if len(boundary) == 0:
            return Circle(np.zeros(2), 0.0)
        if len(boundary) == 1:
            return Circle(boundary[0].copy(), 0.0)
        if len(boundary) == 2:
            return circle_from_two(boundary[0], boundary[1])
        return circle_from_three(boundary[0], boundary[1], boundary[2])
    p = points[n - 1]
    d = _welzl(points, boundary, n - 1)
    if np.linalg.norm(p - d.center) <= d.radius + 1e-8:
        return d
    return _welzl(points, boundary + [p.copy()], n - 1)


def minimum_enclosing_circle(points: Sequence[Sequence[float]]) -> Circle:
    """Welzl 随机增量算法，返回覆盖点集的最小圆近似。"""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return Circle(np.zeros(2), 0.0)
    if pts.shape[0] == 1:
        return Circle(pts[0].copy(), 0.0)
    # 去重
    uniq = []
    for p in pts:
        if not uniq or np.linalg.norm(p - uniq[-1]) > 1e-9:
            uniq.append(p)
    if len(uniq) < len(pts):
        # 不只比较相邻，简单做一轮去重
        seen = []
        for p in pts:
            if all(np.linalg.norm(p - q) > 1e-9 for q in seen):
                seen.append(p)
        pts = np.asarray(seen, dtype=float)
    rng = np.random.default_rng(20260911)
    pts = pts[rng.permutation(len(pts))]
    c = _welzl(pts, [], len(pts))
    # 数值微调半径
    if len(pts):
        c.radius = max(c.radius, float(np.max(np.linalg.norm(pts - c.center, axis=1))))
    return c


def conservative_clear_certificate(poly: np.ndarray, margin: float = 19.5) -> Tuple[bool, Optional[np.ndarray], float]:
    """检查是否可一次 clear 成功；返回 (是否, 清除点, 最小包围圆半径)。"""
    if poly is None or len(poly) == 0:
        return False, None, float("inf")
    c = minimum_enclosing_circle(poly)
    return c.radius <= margin, c.center, float(c.radius)


def sample_convex_polygon(poly: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """在凸多边形内均匀采样（用于信念/场景样本）。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2))
    if len(poly) == 1:
        return np.repeat(poly, k, axis=0)
    if len(poly) == 2:
        t = rng.random(k)[:, None]
        return poly[0][None, :] * t + poly[1][None, :] * (1 - t)
    # 使用三角形扇：按面积加权
    c = polygon_centroid(poly)
    tri = []
    areas = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        tri.append((c, a, b))
        areas.append(abs(0.5 * np.cross(a - c, b - c)))
    areas = np.asarray(areas, dtype=float)
    if areas.sum() <= 1e-15:
        return np.repeat(c[None, :], k, axis=0)
    probs = areas / areas.sum()
    idx = rng.choice(len(tri), size=k, p=probs)
    out = np.empty((k, 2), dtype=float)
    for j, ti in enumerate(idx):
        a, b, cc = tri[ti]
        r1 = np.sqrt(rng.random())
        r2 = rng.random()
        out[j] = (1 - r1) * a + r1 * (1 - r2) * b + r1 * r2 * cc
    return out


def polygon_area(poly: np.ndarray) -> float:
    return abs(poly_signed_area(poly))

def bearing_cross_sine(first: Sequence[float], source_estimate: Sequence[float],
                       candidate: Sequence[float]) -> float:
    """候选源估计位置处两条观测方向的交叉正弦。

    0 表示共线，1 表示正交。旧实现使用候选测点处的夹角，会把共线
    观测误奖励为 180 度；本函数用于替代。
    """
    a = np.asarray(first, dtype=float)
    g = np.asarray(source_estimate, dtype=float)
    p = np.asarray(candidate, dtype=float)
    v1 = a - g
    v2 = p - g
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2))

