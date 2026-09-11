"""对第1、2问和覆盖/保底构造进行数值核验，输出 JSON。"""
from __future__ import annotations

import json
import sys
from math import cos, pi, radians, sin
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.coverage import coverage_verify_s3, coverage_verify_s4, s3_points, s4_points
from brl.geometry import (conservative_clear_certificate, intersect_halfplanes,
                          minimum_enclosing_circle, unit_from_deg, wedge_halfplanes,
                          polygon_diameter, polygon_diameter_rotating_calipers)
from brl.resolver import optical_fallback_points, second_point_candidates


def s3_analytic_bound():
    # 对 r∈[1000,1800]，d^2(r)=r^2+1200^2-2400r cos30；凸函数最大值在端点。
    c = cos(radians(30))
    vals = []
    for r in (1000.0, 1800.0):
        vals.append(r * r + 1200.0 ** 2 - 2400.0 * r * c)
    return float(np.sqrt(max(vals)))


def q_pm_worst(seed=42, n=200000):
    rng = np.random.default_rng(seed)
    # 首测扇区：距离(0,1500)、角度误差±1°
    dist = 1500.0 * np.sqrt(rng.random(n))
    theta_true = rng.uniform(0, 2 * pi, n)
    # 真实位置相对首测示向的方向误差均匀在 ±1°
    err = np.deg2rad(rng.uniform(-1, 1, n))
    # 为覆盖最坏情形，也可在 0° 方向附近扫描；这里用随机目标再取max，结果下界。
    pos = np.column_stack([dist * np.cos(err), dist * np.sin(err)])
    q = second_point_candidates((0, 0), 0.0)
    d = np.min(np.linalg.norm(pos[:, None, :] - q[None, :, :], axis=2), axis=1)
    return float(d.max())


def fallback_worst(seed=43, n=200000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1500, n)
    y = rng.uniform(-1500 * sin(radians(1)), 1500 * sin(radians(1)), n)
    pos = np.column_stack([x, y])
    pts = optical_fallback_points((0, 0), 0.0)
    d = np.min(np.linalg.norm(pos[:, None, :] - pts[None, :, :], axis=2), axis=1)
    return float(d.max())


def geometry_test(seed=44, n=200):
    rng = np.random.default_rng(seed)
    bad = 0
    max_diam_err = 0.0
    circle_bad = 0
    for _ in range(n):
        s = rng.uniform(-800, 800, 2)
        g = rng.uniform(-1800, 1800, 2)
        if np.linalg.norm(g) > 1800:
            continue
        # 生成一个使 g 相对 s 落在楔形内的示向度（加微小误差）
        true_ang = np.degrees(np.arctan2(g[1] - s[1], g[0] - s[0]))
        for err in (-0.9, 0.0, 0.9):
            theta = true_ang + err
            p = intersect_halfplanes(wedge_halfplanes(s, theta))
            if len(p) == 0 or not _point_in(p, g):
                bad += 1
            c = minimum_enclosing_circle(p)
            if c.radius + 1e-6 < np.max(np.linalg.norm(p - c.center, axis=1)):
                circle_bad += 1
            max_diam_err = max(max_diam_err, 0.0)
    return {"cases": n * 3, "bad_containment": bad, "bad_circle": circle_bad}


def _point_in(poly, p):
    # 简单射线法（凸/非凸均可）
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def diameter_crosscheck(seed=45, n=300):
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    worst = 0.0
    bad = 0
    for _ in range(n):
        pts = rng.normal(0, 1, size=(rng.integers(5, 30), 2))
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        poly = pts[hull.vertices]
        d1 = polygon_diameter(poly)
        d2 = polygon_diameter_rotating_calipers(poly)
        e = abs(d1 - d2)
        worst = max(worst, e)
        if e > 1e-6:
            bad += 1
    return {"cases": n, "worst_abs_diff": float(worst), "mismatches": bad}


def main():
    out = {
        "S3": {"points": len(s3_points()), "analytic_worst_distance": s3_analytic_bound(),
               "numeric": coverage_verify_s3(samples=200000, seed=1)},
        "S4": {"points": len(s4_points()), "numeric": coverage_verify_s4(samples=200000, seed=2)},
        "q_pm_random_sector_max": q_pm_worst(),
        "optical_fallback_random_max": fallback_worst(),
        "geometry": geometry_test(),
        "diameter_crosscheck": diameter_crosscheck(),
        "min_enclosing_triangle_40": minimum_enclosing_circle(
            [(0, 0), (40, 0), (20, 20 * np.sqrt(3))]).radius,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    path = ROOT / "results" / "math_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
