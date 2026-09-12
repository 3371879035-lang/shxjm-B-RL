"""三项因果拆分：A发现成本、B发现+最优清除、C路线Oracle。"""
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from .bilateral import bearing_clip, solve_bilateral
from .coverage import s3_points, s25_points
from .g25o import (_best_clear_point, _clip_target_square, _poly_center_radius,
                   initial_track_polygon, route_open)
from .local_env import CHANNELS, CLEAR_DIST, RadioEnv
from .oracle_policy import _assert_local, _held_karp_open, pre_mark_empty
from .resolver import optical_fallback_points


def _visible(src, p) -> bool:
    p = np.asarray(p, dtype=float)
    d = float(np.linalg.norm(p - src.position))
    if d > src.radius + 1e-9:
        return False
    if src.kind == "directional":
        n = np.array([math.cos(math.radians(src.direction_deg)), math.sin(math.radians(src.direction_deg))])
        if float(np.dot(n, p - src.position)) < -1e-9:
            return False
    return True


def _measure_unknown_at(env: RadioEnv, idx: int, points: np.ndarray):
    p = points[int(idx)]
    chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
           and idx not in env.channels[ch].scan_points]
    chs.sort(key=lambda c: (c != env.current_channel, c))
    for ch in chs:
        if env.done:
            return
        env.measure(p, ch, coverage_idx=int(idx), is_refine=False)


def _all_true_sources_found(env: RadioEnv) -> bool:
    return all(env.channels[int(s.channel)].status in ("discovered", "cleared") for s in env.sources)


def run_discovery_only(env: RadioEnv) -> dict:
    """A：固定S3/S25路线正常measure，但不定位、不绕路clear，到所有真实源首次发现为止。"""
    _assert_local(env)
    mode = env.mode
    points = s3_points() if mode == 3 else s25_points()
    env.coverage_points = points; env.n_coverage = len(points)
    order = route_open(points, env.pos)
    t0 = env.virtual_time; d0 = env.move_distance; m0 = env.n_measure
    stop_idx = None
    for oi, idx in enumerate(order):
        _measure_unknown_at(env, idx, points)
        if _all_true_sources_found(env):
            stop_idx = int(idx); break
    return {"method": "A-DiscoveryOnly", "all_found": bool(_all_true_sources_found(env)),
            "stop_coverage_idx": stop_idx,
            "virtual_time_s": float(env.virtual_time),
            "discovery_time_s": float(env.virtual_time - t0),
            "discovery_distance_m": float(env.move_distance - d0),
            "discovery_measures": int(env.n_measure - m0),
            "sources": int(len(env.sources)), "cleared": int(env.cleared_count())}

def _optimize_disk_route_from_start(start: Sequence[float], sources: Sequence) -> Tuple[np.ndarray, List[int]]:
    start = np.asarray(start, dtype=float)
    pts = np.asarray([s.position for s in sources], dtype=float)
    n = len(pts)
    if n == 0:
        return np.empty((0, 2)), []
    coords = np.vstack([start[None, :], pts])
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    dp = np.full((1 << n, n), np.inf); parent = np.full((1 << n, n), -1, dtype=np.int32)
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
    mask = (1 << n) - 1; j = int(np.argmin(dp[mask])); order = []
    for _ in range(n):
        order.append(j); k = int(parent[mask, j]); mask ^= (1 << j); j = k
    order.reverse()
    g = pts[order]
    if n == 1:
        v = g[0] - start; nv = float(np.linalg.norm(v))
        if nv <= CLEAR_DIST:
            q = start[None, :].copy()
        else:
            q = (start + v / nv * (nv - CLEAR_DIST))[None, :].copy()
        return q, [int(sources[order[0]].channel)]
    x0 = g.reshape(-1)
    def obj(x):
        q = x.reshape(n, 2)
        total = float(np.linalg.norm(q[0] - start))
        for i in range(1, n):
            total += float(np.linalg.norm(q[i] - q[i - 1]))
        return total
    def cons(x):
        q = x.reshape(n, 2)
        return (CLEAR_DIST - 1e-6) ** 2 - np.sum((q - g) ** 2, axis=1)
    res = minimize(obj, x0, method="SLSQP", constraints={"type": "ineq", "fun": cons},
                   options={"maxiter": 300, "ftol": 1e-10, "disp": False})
    q = res.x.reshape(n, 2) if res.success else g.copy()
    for i in range(n):
        v = q[i] - g[i]; nv = float(np.linalg.norm(v))
        if nv > CLEAR_DIST - 1e-6:
            q[i] = g[i] + v / nv * (CLEAR_DIST - 1e-6)
    return q, [int(sources[order[i]].channel) for i in range(n)]


def run_discovery_optimal_cleanup(env: RadioEnv) -> dict:
    """B：A到全发现后，用真值最佳路线清掉剩余源。"""
    _assert_local(env)
    a = run_discovery_only(env)
    if not a["all_found"]:
        return {**a, "method": "B-Discovery+OptimalCleanup", "cleanup_time_s": float("inf"),
                "virtual_time_s": float("inf")}
    pre_mark_empty(env)
    t1 = env.virtual_time; d1 = env.move_distance
    remaining = [s for s in env.sources if env.channels[int(s.channel)].status != "cleared"]
    qs, channels = _optimize_disk_route_from_start(env.pos, remaining)
    for qi, ch in zip(qs, channels):
        if env.done:
            break
        env.clear(qi, ch)
    return {"method": "B-Discovery+OptimalCleanup", "discovery_time_s": a["discovery_time_s"],
            "discovery_distance_m": a["discovery_distance_m"], "discovery_measures": a["discovery_measures"],
            "cleanup_time_s": float(env.virtual_time - t1), "cleanup_distance_m": float(env.move_distance - d1),
            "virtual_time_s": float(env.virtual_time), "distance_m": float(env.move_distance),
            "measure_calls": int(env.n_measure), "clear_calls": int(env.n_clear),
            "sources": int(len(env.sources)), "cleared": int(env.cleared_count()),
            "all_found": True, "success": bool(env.success or env.completion_certificate())}

def run_route_oracle(env: RadioEnv, seq=None, resolve_key: str = "true") -> dict:
    """C：保持合法信息和证明义务，只用真值优化搜索点、定位、清除顺序。"""
    _assert_local(env)
    mode = env.mode
    points = s3_points() if mode == 3 else s25_points()
    env.coverage_points = points; env.n_coverage = len(points)
    base_order = route_open(points, np.zeros(2, dtype=float))
    seq = list(base_order) if seq is None else [int(x) for x in seq]
    tracks: Dict[int, dict] = {}
    t0 = env.virtual_time; d0 = env.move_distance; m0 = env.n_measure

    def track_observe(ch, p, out):
        typ = out.get("measure_result")
        if typ == "direction" and out.get("svd_deg") is not None:
            if ch not in tracks:
                tracks[ch] = {"first": np.asarray(p, dtype=float).copy(), "deg": float(out["svd_deg"]),
                              "P": initial_track_polygon(np.asarray(p, dtype=float), float(out["svd_deg"])),
                              "nobs": 1}
            else:
                st = tracks[ch]
                st["P"] = _clip_target_square(bearing_clip(st["P"], np.asarray(p, dtype=float),
                                                            math.radians(float(out["svd_deg"]))))
                st["nobs"] = int(st.get("nobs", 1)) + 1
                if len(st["P"]) == 0:
                    st["P"] = initial_track_polygon(st["first"], st["deg"])
        elif typ == "near":
            if env.channels[ch].status != "cleared":
                env.clear(p, ch)
            tracks.pop(ch, None)

    def resolve_track(ch):
        st = tracks[ch]
        first = np.asarray(st["first"], dtype=float).copy(); deg = float(st["deg"])
        P = st["P"] if len(st["P"]) > 0 else None
        def measure_cb(q, ch=ch):
            out = env.measure(q, ch, is_refine=True)
            return {"measure_result": out.get("measure_result"), "svd_deg": out.get("svd_deg")}
        def clear_cb(q, ch=ch):
            return {"clear_result": env.clear(q, ch).get("clear_result")}
        try:
            solve_bilateral(first, deg, env.pos, measure_cb, clear_cb, initial_region=P)
        except Exception:
            for q in optical_fallback_points(first, deg):
                if env.channels[ch].status == "cleared" or env.done:
                    break
                if env.clear(q, ch).get("clear_result") == "success":
                    break
        tracks.pop(ch, None)

    for pos_i, nxt in enumerate(seq):
        if env.done:
            break
        B = points[nxt]
        chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
               and nxt not in env.channels[ch].scan_points]
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                break
            out = env.measure(B, ch, coverage_idx=int(nxt), is_refine=False)
            track_observe(ch, B, out)
        # 与G25OR一致的顺路机会测向，避免C因缺少该机制而整体变慢。
        for ch in list(tracks.keys()):
            st = tracks.get(ch)
            if st is None:
                continue
            c, r = _poly_center_radius(st["P"])
            if not np.isfinite(r) or r <= 19.5 or int(st.get("nobs", 0)) >= 4:
                continue
            delta = c - env.pos; w = c - st["first"]
            d = float(np.linalg.norm(delta)); w0 = float(np.linalg.norm(w))
            if d < 1e-9 or w0 < 1e-9:
                continue
            sine = abs(delta[0] * w[1] - delta[1] * w[0]) / (d * w0)
            if d <= 1200.0 and sine > 0.15:
                out = env.measure(env.pos, ch, is_refine=True)
                track_observe(ch, env.pos, out)
        # 与G25OR一致：扫描点结束后再尝试顺路清除。
        if pos_i + 1 < len(seq):
            Bn = points[seq[pos_i + 1]]
            best = None
            for ch, st in list(tracks.items()):
                qc = _best_clear_point(st["P"], env.pos, Bn)
                if qc is None:
                    continue
                det = float(np.linalg.norm(env.pos - qc) + np.linalg.norm(qc - Bn) - np.linalg.norm(env.pos - Bn))
                if det <= 250.0 and (best is None or det < best[0]):
                    best = (det, ch, qc)
            if best is not None and not env.done:
                _, ch, qc = best
                if env.clear(qc, ch).get("clear_result") == "success":
                    tracks.pop(ch, None)
    d_scan = float(env.move_distance)
    while tracks and not env.done:
        if resolve_key == "mec":
            def order_key(c):
                return _poly_center_radius(tracks[c]["P"])[0]
        else:
            def order_key(c):
                src = env.source_by_channel.get(int(c))
                if src is not None:
                    return np.asarray(src.position, dtype=float)
                return _poly_center_radius(tracks[c]["P"])[0]
        ch = min(tracks.keys(), key=lambda c: float(np.linalg.norm(order_key(c) - env.pos)))
        resolve_track(ch)
    return {"method": "C-RouteOracle", "virtual_time_s": float(env.virtual_time - t0),
            "discovery_time_s": float(env.virtual_time - t0),
            "distance_m": float(env.move_distance - d0), "scan_distance_m": float(d_scan - d0),
            "resolve_distance_m": float(env.move_distance - d_scan),
            "measure_calls": int(env.n_measure - m0),
            "clear_calls": int(env.n_clear), "sources": int(len(env.sources)),
            "cleared": int(env.cleared_count()), "success": bool(env.success or env.completion_certificate())}

def _candidate_orders(env: RadioEnv, points: np.ndarray):
    """用真值构造若干合法覆盖顺序；动作仍为measure/clear/双边求解。"""
    base = [int(i) for i in route_open(points, env.pos)]
    orders = {"baseline": list(base), "baseline_rev": list(reversed(base))}
    n = len(points)
    remaining = set(range(n))
    cur = np.asarray(env.pos, dtype=float)
    targets = []
    for src in list(env.sources):
        visible = []
        for i in range(n):
            p = points[i]
            if _visible(src, p):
                visible.append((float(np.linalg.norm(p - src.position)), i))
        if not visible:
            for i in range(n):
                visible.append((float(np.linalg.norm(points[i] - src.position)) + 500.0, i))
        visible.sort(key=lambda x: (x[0], x[1]))
        targets.append(int(visible[0][1]))
    seq = []
    while remaining:
        cand = [i for i in set(targets) if i in remaining]
        if cand:
            i = min(cand, key=lambda k: (float(np.linalg.norm(points[k] - cur)), k))
        else:
            i = min(remaining, key=lambda k: (float(np.linalg.norm(points[k] - cur)), k))
        seq.append(int(i)); remaining.remove(i); cur = points[i]
    orders["source_first"] = seq
    if targets:
        first = min(set(targets), key=lambda k: (float(np.linalg.norm(points[k] - env.pos)), k))
        tail = [i for i in base if i != first]
        orders["source_first_then_base"] = [int(first)] + tail
    return orders


def run_route_oracle_best(env: RadioEnv) -> dict:
    """C：在若干合法覆盖顺序与求解顺序中，用真值选最优。"""
    import copy
    _assert_local(env)
    points = s3_points() if env.mode == 3 else s25_points()
    orders = _candidate_orders(env, points)
    base_env = copy.deepcopy(env)
    results = []
    for name, seq in orders.items():
        for rk in ("mec", "true"):
            try:
                res = run_route_oracle(copy.deepcopy(base_env), seq=seq, resolve_key=rk)
            except Exception:
                continue
            res["candidate"] = name
            res["resolve_key"] = rk
            results.append(res)
    if not results:
        res = run_route_oracle(copy.deepcopy(base_env), seq=None, resolve_key="mec")
        res["candidate"] = "baseline_fallback"
        return res
    results.sort(key=lambda r: (0 if r.get("success") else 1, float(r["virtual_time_s"])))
    best = dict(results[0])
    best["oracle_candidates"] = [
        {"candidate": r.get("candidate"), "resolve_key": r.get("resolve_key"),
         "virtual_time_s": float(r["virtual_time_s"]), "distance_m": float(r.get("distance_m", 0.0)),
         "success": bool(r.get("success"))} for r in results]
    return best
