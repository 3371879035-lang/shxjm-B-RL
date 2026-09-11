"""保证发现覆盖的测点集：问题3的 S3 与问题4的 S4。"""
from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import List

import numpy as np

DOMAIN_RADIUS = 1800.0
MIN_RECEIVE_RADIUS = 1000.0


def s3_points(hex_radius: float = 1200.0) -> np.ndarray:
    """原点 + 半径 1200 的正六边形六个顶点，共7点。"""
    pts = [np.array([0.0, 0.0])]
    for k in range(6):
        a = k * pi / 3.0
        pts.append(np.array([hex_radius * cos(a), hex_radius * sin(a)]))
    return np.asarray(pts, dtype=float)


def triangular_lattice(h: float = 950.0, limit: float = DOMAIN_RADIUS + 950.0) -> np.ndarray:
    """三角网格点 s_ij=(h(i+j/2), sqrt(3)/2*h*j)，保留 ||s|| <= limit。"""
    # 先估计整数范围
    kmax = int(limit / max(h, 1e-9)) + 4
    pts = []
    for j in range(-kmax, kmax + 1):
        for i in range(-kmax, kmax + 1):
            x = h * (i + 0.5 * j)
            y = (sqrt(3.0) / 2.0) * h * j
            if x * x + y * y <= limit * limit + 1e-9:
                pts.append((x, y))
    # 去重并排序（原点优先，便于固定槽位）
    uniq = []
    for p in pts:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 1e-12 for q in uniq):
            uniq.append(p)
    return np.asarray(uniq, dtype=float)


def s4_points(h: float = 950.0) -> np.ndarray:
    return triangular_lattice(h=h, limit=DOMAIN_RADIUS + h)


def coverage_verify_s3(samples: int = 200000, seed: int = 1) -> dict:
    """数值核验 S3 对全向源的发现覆盖。"""
    rng = np.random.default_rng(seed)
    pts = s3_points()
    # 圆内均匀采样 + 边界采样
    n1 = samples * 3 // 4
    r = DOMAIN_RADIUS * np.sqrt(rng.random(n1))
    th = rng.random(n1) * 2 * pi
    q = np.column_stack([r * np.cos(th), r * np.sin(th)])
    n2 = samples - n1
    th2 = rng.random(n2) * 2 * pi
    q2 = np.column_stack([DOMAIN_RADIUS * np.cos(th2), DOMAIN_RADIUS * np.sin(th2)])
    q = np.vstack([q, q2])
    worst = 0.0
    miss = 0
    for p in q:
        d = np.linalg.norm(pts - p, axis=1).min()
        worst = max(worst, float(d))
        if d > MIN_RECEIVE_RADIUS:
            miss += 1
    return {"points": len(pts), "worst_distance": worst, "misses": miss, "samples": len(q)}


def coverage_verify_s4(samples: int = 200000, seed: int = 2) -> dict:
    """数值核验 S4 对任意 180 度定向源的发现覆盖（每点随机一个方向）。"""
    rng = np.random.default_rng(seed)
    pts = s4_points()
    n1 = samples * 3 // 4
    r = DOMAIN_RADIUS * np.sqrt(rng.random(n1))
    th = rng.random(n1) * 2 * pi
    q = np.column_stack([r * np.cos(th), r * np.sin(th)])
    n2 = samples - n1
    th2 = rng.random(n2) * 2 * pi
    q2 = np.column_stack([DOMAIN_RADIUS * np.cos(th2), DOMAIN_RADIUS * np.sin(th2)])
    q = np.vstack([q, q2])
    dirs = rng.random((len(q), 2)) * 2 - 1
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-15
    worst_visible = 0.0
    miss = 0
    for p, n in zip(q, dirs):
        rel = pts - p
        dist = np.linalg.norm(rel, axis=1)
        visible = (rel @ n) >= -1e-9
        if not np.any(visible & (dist <= MIN_RECEIVE_RADIUS)):
            miss += 1
        else:
            # 记录在可见点中的最小距离最大值
            dv = dist[visible]
            if len(dv):
                worst_visible = max(worst_visible, float(dv.min()))
    return {"points": len(pts), "misses": miss, "samples": len(q), "worst_visible_min": worst_visible}


if __name__ == "__main__":
    p3 = s3_points()
    p4 = s4_points()
    print("S3", len(p3), p3.tolist())
    print("S4", len(p4))
    print(coverage_verify_s3(samples=50000))
    print(coverage_verify_s4(samples=50000))


def s25_points() -> np.ndarray:
    """问题4新覆盖构造：原点 + 内环12点 + 外环12点，共25点。

    内环半径970m，角度15+30k度；外环半径1880m，角度30k度。
    外正十二边形内切圆半径 1880*cos15° = 1815.94m > 1800m，覆盖目标圆。
    36个三角形最大边长约975.90m < 1000m，保证任意180度定向源至少一点可见。
    """
    k = np.arange(12, dtype=float)
    origin = np.zeros((1, 2), dtype=float)
    a_in = np.deg2rad(15.0 + 30.0 * k)
    inner = 970.0 * np.column_stack([np.cos(a_in), np.sin(a_in)])
    a_out = np.deg2rad(30.0 * k)
    outer = 1880.0 * np.column_stack([np.cos(a_out), np.sin(a_out)])
    return np.vstack([origin, inner, outer])


def s25_max_triangle_edge() -> float:
    """返回36个覆盖三角形的最大边长。"""
    pts = s25_points()
    O = pts[0]
    inner = pts[1:13]
    outer = pts[13:25]
    edges = []
    for k in range(12):
        k2 = (k + 1) % 12
        tri1 = [O, inner[k], inner[k2]]
        tri2 = [outer[k], outer[k2], inner[k]]
        tri3 = [inner[k], inner[k2], outer[k2]]
        for tri in (tri1, tri2, tri3):
            for i in range(3):
                edges.append(float(np.linalg.norm(tri[i] - tri[(i + 1) % 3])))
    return max(edges)
