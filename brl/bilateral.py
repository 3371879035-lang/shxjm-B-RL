"""双侧区间定位：把定向源的成对 no_signal 转化为距离区间收缩。

该模块只使用 measure/clear 回调，不访问真实源。证明与上界来自新方案论证：
首次有效示向后最多 12 次追加测向，末端最多 2 次光学 clear。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .geometry import minimum_enclosing_circle
from .protocol import (ActionIOError, CertificateViolation,
                       GeometryNumericalError, OFFICIAL_BEARING_ENVELOPE_DEG)

EPS_DEG = OFFICIAL_BEARING_ENVELOPE_DEG
EPS = math.radians(EPS_DEG)
K = math.tan(EPS)
R_MAX = 1500.0


def _ordered_by_travel(points, current_position):
    """Order symmetric candidates reproducibly across platforms.

    Several bilateral probes are theoretically equidistant.  Raw libm/BLAS
    rounding used to reverse those ties on Windows and Linux, which changed the
    observation sequence.  Nine decimal places are far below the simulator's
    coordinate precision while making the documented list order the tie-break.
    """
    current = np.asarray(current_position, dtype=float)
    indexed = list(enumerate(points))
    indexed.sort(key=lambda item: (
        round(float(np.linalg.norm(np.asarray(item[1], dtype=float) - current)), 9),
        item[0],
    ))
    return [point for _, point in indexed]


def clip(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """半平面 normal @ z <= offset 的凸多边形裁剪。"""
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=float)
    poly = np.asarray(poly, dtype=float)
    vals = poly @ normal - offset
    out = []
    n = len(poly)
    for i, a in enumerate(poly):
        b = poly[(i + 1) % n]
        da = float(vals[i]); db = float(vals[(i + 1) % n])
        if da <= 1e-9:
            out.append(a)
        if (da < -1e-9 and db > 1e-9) or (da > 1e-9 and db < -1e-9):
            out.append(a + (b - a) * (da / (da - db)))
    if not out:
        return np.empty((0, 2), dtype=float)
    return np.asarray(out, dtype=float).reshape(-1, 2)


def bearing_clip(poly: np.ndarray, p: np.ndarray, theta: float) -> np.ndarray:
    u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    v = np.array([-u[1], u[0]], dtype=float)
    for normal in (-u, v - K * u, -v - K * u):
        poly = clip(poly, normal, float(normal @ p))
    return poly


@dataclass
class SolveResult:
    success: bool
    extra_measures: int
    clear_attempts: int
    rounds: int
    position: np.ndarray
    region: np.ndarray


def solve_bilateral(first_position, first_bearing_deg: float, current_position,
                    measure: Callable, clear: Callable, *,
                    geometry: bool = True,
                    initial_region: Optional[np.ndarray] = None) -> SolveResult:
    """用双侧检测解决一个已确认源。

    measure(q)->{'measure_result':..., 'svd_deg':...}
    clear(q)->{'clear_result':...}
    回调必须已绑定频道，并校验 accepted 与结果字段。
    """
    origin = np.asarray(first_position, dtype=float)
    a0 = math.radians(float(first_bearing_deg))
    u = np.array([math.cos(a0), math.sin(a0)], dtype=float)
    v = np.array([-u[1], u[0]], dtype=float)
    B = np.stack([u, v], axis=1)
    pos = np.asarray(current_position, dtype=float).copy()
    lo, hi = 0.0, R_MAX
    poly = np.array([[0.0, 0.0], [R_MAX, -R_MAX * K], [R_MAX, R_MAX * K]], dtype=float)
    if initial_region is not None:
        supplied = (np.asarray(initial_region, dtype=float) - origin) @ B
        for normal, c in ((np.array([-1.0, 0.0]), 0.0), (np.array([1.0, 0.0]), R_MAX),
                          (np.array([-K, 1.0]), 0.0), (np.array([-K, -1.0]), 0.0)):
            supplied = clip(supplied, normal, c)
        if len(supplied) == 0:
            raise GeometryNumericalError("invalid initial conservative region")
        poly = supplied
        lo = max(lo, float(poly[:, 0].min()))
        hi = min(hi, float(poly[:, 0].max()))
    nm = nc = rounds = 0

    def glob(z):
        return origin + B @ np.asarray(z, dtype=float)

    def ret(ok):
        return SolveResult(ok, nm, nc, rounds, pos.copy(), poly @ B.T + origin)

    for _ in range(7):
        if geometry:
            lo = max(lo, float(poly[:, 0].min()))
            hi = min(hi, float(poly[:, 0].max()))
            mec = minimum_enclosing_circle(poly)
            if float(mec.radius) <= 19.5:
                pos = glob(mec.center)
                nc += 1
                ans = clear(pos)
                if ans.get("clear_result") == "success":
                    return ret(True)
                raise CertificateViolation("certified optical clear failed")
        if hi - lo <= 24.0 + 1e-7:
            break
        rounds += 1
        mid = (lo + hi) / 2.0
        side = mid * K + 5.0
        probes = [np.array([mid, side]), np.array([mid, -side])]
        probes = _ordered_by_travel(probes, B.T @ (pos - origin))
        observed = False
        for q in probes:
            pos = glob(q)
            nm += 1
            ans = measure(pos)
            status = ans.get("measure_result")
            if status == "near":
                nc += 1
                if clear(pos).get("clear_result") == "success":
                    return ret(True)
                raise CertificateViolation("near observation contradicted by failed clear")
            if status == "no_signal":
                continue
            if status != "direction" or "svd_deg" not in ans:
                raise ActionIOError(f"invalid bilateral measure response: {ans}")
            observed = True
            theta = math.radians(float(ans["svd_deg"]) - first_bearing_deg)
            cx = math.cos(theta)
            if cx > math.sin(EPS) + 1e-12:
                lo = max(lo, mid)
            elif cx < -math.sin(EPS) - 1e-12:
                hi = min(hi, mid)
            else:
                w = (R_MAX * K + side) * math.tan(2 * EPS)
                lo = max(lo, mid - w)
                hi = min(hi, mid + w)
            if geometry:
                poly = bearing_clip(poly, q, theta)
            break
        if not observed:
            # 双侧 no_signal：若 x>=mid，则两探测点连线上存在首次可见点与源之间的点，
            # 且两探测点都比首次测点更近，不可能同时不可见，故 x<mid。
            hi = min(hi, mid)
        poly = clip(poly, np.array([1.0, 0.0]), hi)
        poly = clip(poly, np.array([-1.0, 0.0]), -lo)
        if len(poly) == 0 or lo > hi + 1e-7:
            raise GeometryNumericalError("empty certified region")
    if hi - lo > 24.0 + 1e-6:
        raise GeometryNumericalError("interval did not contract within seven rounds")
    targets = [np.array([(lo + hi) / 2.0, hi * K / 2.0]),
               np.array([(lo + hi) / 2.0, -hi * K / 2.0])]
    targets = _ordered_by_travel(targets, B.T @ (pos - origin))
    for q in targets:
        pos = glob(q)
        nc += 1
        if clear(pos).get("clear_result") == "success":
            return ret(True)
    raise CertificateViolation("two-disk terminal cover failed")
