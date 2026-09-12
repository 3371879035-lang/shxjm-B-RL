"""Oracle Ladder：只在本地RadioEnv上运行，绝不接入RemoteBelief/官方runner。

层级：
  O1: 知道哪些频道有源
  O2: 首次发现后知道精确位置
  O3: 知道有源频道 + 首次发现后知道精确位置
  O0: 知道数量、频道、位置（不清除方向/半径未知；clear与方向无关）
"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from .coverage import s3_points, s25_points
from .g25o import G25OPolicy, route_open
from .local_env import CHANNELS, CLEAR_DIST, RadioEnv


def _assert_local(env):
    assert isinstance(env, RadioEnv), "Oracle Ladder只允许在本地RadioEnv上读取真值"
    return env


def _actual_channels(env: RadioEnv):
    return {int(s.channel) for s in env.sources}


def pre_mark_empty(env: RadioEnv) -> None:
    """O1/O3/O0：把无源频道直接标记为absent（overwrite oracle knowledge）。"""
    actual = _actual_channels(env)
    for ch, st in env.channels.items():
        if ch not in actual:
            st.status = "absent"


def run_o1(env: RadioEnv) -> dict:
    """O1：知道有源频道，位置/方向/半径仍未知；其余运行G25OR-S25。"""
    _assert_local(env)
    pre_mark_empty(env)
    out = G25OPolicy("G25OR", coverage="S25").run(env)
    return _finalize(env, out, "O1-Channel", env.sources)


def _run_discovery_clear(env: RadioEnv) -> dict:
    """O2/O3共用：正常覆盖扫描发现，首次direction/near后用真值直接清除。"""
    mode = env.mode
    points = s3_points() if mode == 3 else s25_points()
    env.coverage_points = points
    env.n_coverage = len(points)
    order = route_open(points, env.pos)
    found: Dict[int, np.ndarray] = {}
    cleared = set()

    def clear_at(ch, q):
        if ch in cleared or env.channels[ch].status == "cleared" or env.done:
            return
        out = env.clear(np.asarray(q, dtype=float), ch)
        if out.get("clear_result") == "success":
            cleared.add(ch)

    def scan_idx(idx):
        p = points[int(idx)]
        chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
               and idx not in env.channels[ch].scan_points]
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                return
            out = env.measure(p, ch, coverage_idx=int(idx), is_refine=False)
            if out.get("measure_result") in ("direction", "near"):
                found[ch] = np.asarray(env.source_by_channel[ch].position, dtype=float).copy()
                if env.channels[ch].status == "unknown":
                    env.channels[ch].status = "discovered"
            if out.get("measure_result") == "near":
                clear_at(ch, p)
        # 当前位置附近可零绕行清除已发现源
        for ch in list(found.keys()):
            if ch in cleared:
                continue
            g = found[ch]
            if float(np.linalg.norm(env.pos - g)) <= CLEAR_DIST - 0.5:
                clear_at(ch, env.pos)

    def leg_clear(B):
        if env.done:
            return
        # 优先清除当前位置20m内的源
        for ch in list(found.keys()):
            if ch in cleared:
                continue
            g = found[ch]
            if float(np.linalg.norm(env.pos - g)) <= CLEAR_DIST - 0.5:
                clear_at(ch, env.pos)
        # 再尝试低绕行顺路清除
        best = None
        for ch in list(found.keys()):
            if ch in cleared:
                continue
            g = found[ch]
            det = float(np.linalg.norm(env.pos - g) + np.linalg.norm(g - B) - np.linalg.norm(env.pos - B))
            if det <= 250.0 and (best is None or det < best[0]):
                best = (det, ch, g)
        if best is not None and not env.done:
            _, ch, g = best
            clear_at(ch, g)

    scan_idx(order[0])
    for oi in range(1, len(order)):
        if env.done or env.cleared_count() + env.discovered_count() >= 16:
            break
        B = points[order[oi]]
        leg_clear(B)
        if env.done:
            break
        scan_idx(order[oi])
    # 清掉剩余已发现源，按最近邻顺序到源中心清除
    while not env.done:
        remaining = [ch for ch in found.keys() if ch not in cleared and env.channels[ch].status != "cleared"]
        if not remaining:
            break
        ch = min(remaining, key=lambda c: float(np.linalg.norm(found[c] - env.pos)))
        # 先移动到源中心再clear；一次clear会包含移动时间
        clear_at(ch, found[ch])
    return _finalize(env, {}, "O2-PositionAfterFirstHit", env.sources)


def run_o2(env: RadioEnv) -> dict:
    """O2：不知道有源频道；首次命中后用真值直接清除。"""
    _assert_local(env)
    out = _run_discovery_clear(env)
    out["method"] = "O2-PositionAfterFirstHit"
    return out


def run_o3(env: RadioEnv) -> dict:
    """O3：知道有源频道；首次命中后用真值直接清除。"""
    _assert_local(env)
    pre_mark_empty(env)
    out = _run_discovery_clear(env)
    out["method"] = "O3-Channel+PositionAfterHit"
    return out


def _held_karp_open(points: Sequence[Sequence[float]]) -> Tuple[float, List[int]]:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return 0.0, []
    if n == 1:
        return float(np.linalg.norm(pts[0])), [0]
    coords = np.vstack([np.zeros((1, 2)), pts])
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    n = len(pts)
    dp = np.full((1 << n, n), np.inf)
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    for j in range(n):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            prev = mask ^ (1 << j)
            if prev == 0:
                continue
            best = np.inf; pred = -1
            for k in range(n):
                if prev & (1 << k):
                    val = dp[prev, k] + D[k + 1, j + 1]
                    if val < best:
                        best = val; pred = k
            dp[mask, j] = best; parent[mask, j] = pred
    mask = (1 << n) - 1
    j = int(np.argmin(dp[mask])); cost = float(dp[mask, j])
    order = []
    for _ in range(n):
        order.append(j)
        k = int(parent[mask, j]); mask ^= (1 << j); j = k
    order.reverse()
    return cost, order


def _optimize_disk_route(sources: Sequence) -> Tuple[np.ndarray, List[int]]:
    """固定中心TSP顺序，在20m清除圆内优化访问点；返回访问点顺序和频道顺序。"""
    pts = np.asarray([s.position for s in sources], dtype=float)
    _, order = _held_karp_open(pts)
    g = pts[order]
    n = len(g)
    if n == 1:
        # 单源：在20m清除圆内选离原点最近的点
        r = float(np.linalg.norm(g[0]))
        if r <= CLEAR_DIST:
            q = np.zeros((1, 2), dtype=float)
        else:
            q = (g[0] - g[0] / r * CLEAR_DIST)[None, :].copy()
        return q, [int(sources[order[0]].channel)]
    x0 = g.reshape(-1)
    def obj(x):
        q = x.reshape(n, 2)
        total = float(np.linalg.norm(q[0]))
        for i in range(1, n):
            total += float(np.linalg.norm(q[i] - q[i - 1]))
        return total
    def cons(x):
        q = x.reshape(n, 2)
        return (CLEAR_DIST - 1e-6) ** 2 - np.sum((q - g) ** 2, axis=1)
    res = minimize(obj, x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": 300, "ftol": 1e-10, "disp": False})
    q = res.x.reshape(n, 2) if res.success else g.copy()
    # 重新投影，保证严格在圆内
    for i in range(n):
        v = q[i] - g[i]; nv = float(np.linalg.norm(v))
        if nv > CLEAR_DIST - 1e-6:
            q[i] = g[i] + v / nv * (CLEAR_DIST - 1e-6)
    return q, [int(sources[order[i]].channel) for i in range(n)]


def run_o0(env: RadioEnv) -> dict:
    """O0：数量、频道、位置已知；不做任何measure，直接走清除圆路线并clear。"""
    _assert_local(env)
    pre_mark_empty(env)
    sources = list(env.sources)
    if not sources:
        return _finalize(env, {}, "O0-FullOracle", sources)
    q, channels = _optimize_disk_route(sources)
    for qi, ch in zip(q, channels):
        if env.done:
            break
        env.clear(qi, ch)
    return _finalize(env, {}, "O0-FullOracle", sources)


def _finalize(env: RadioEnv, out: dict, method: str, sources: Sequence) -> dict:
    result = {
        "method": method,
        "success": bool(env.success or env.completion_certificate()),
        "cleared": int(env.cleared_count()),
        "sources": int(len(sources)),
        "virtual_time_s": float(env.virtual_time),
        "distance_m": float(env.move_distance),
        "measure_calls": int(env.n_measure),
        "clear_calls": int(env.n_clear),
        "failed_clear": int(env.n_clear_fail),
    }
    if isinstance(out, dict):
        for k in ("solver_failures", "optical_fallbacks", "extra_measurements"):
            if k in out:
                result[k] = out[k]
    return result