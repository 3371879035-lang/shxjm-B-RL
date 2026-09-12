"""G21A：S21保证覆盖 + 零绕行主动补测 + 顺路清除 + 双侧确定性保底。

本模块只在本地/官方 runner 中作为挑战者使用；最终完成证书仍由S21覆盖、
双侧区间定位、光学保底和频道账本共同保证。零绕行补测失败不会删除
已发现源的保守可行域。
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .bilateral import EPS_DEG, K as TAN_EPS, bearing_clip, solve_bilateral
from .coverage import s21_points
from .geometry import bearing_cross_sine, minimum_enclosing_circle
from .g25o import _best_clear_point, _clip_target_square, _poly_center_radius, initial_track_polygon, route_open
from .local_env import CHANNELS, RadioEnv
from .resolver import optical_fallback_points

Q3_SAFE_DIST = 995.0
Q4_MAX_DIST = 1200.0
CLEAR_READY_RADIUS = 19.5
DETOUR_CLEAR_LIMIT = 150.0
OPPORTUNISTIC_BUDGET_Q3 = 2
OPPORTUNISTIC_BUDGET_Q4 = 1
MAX_COMMON_TRACKS = 3
MEASURE_EQ_METERS = 30.0


def _representative_points(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P is None or len(P) == 0:
        return np.empty((0, 2))
    if len(P) == 1:
        return P.copy()
    pts = [P]
    c = minimum_enclosing_circle(P).center
    pts.append(np.asarray(c, dtype=float)[None, :])
    pts.append(0.5 * (P + np.roll(P, -1, axis=0)))
    return np.vstack(pts)


def expected_worst_radius(P: np.ndarray, q: Sequence[float]) -> float:
    """对P的代表点模拟±1.01度示向，返回裁剪后MEC半径的保守上界。"""
    q = np.asarray(q, dtype=float)
    reps = _representative_points(P)
    if len(reps) == 0:
        return float("inf")
    worst = 0.0
    for z in reps:
        base = math.atan2(float(z[1] - q[1]), float(z[0] - q[0]))
        for e in (-EPS_DEG, 0.0, EPS_DEG):
            P2 = bearing_clip(P, q, base + math.radians(e))
            if P2 is None or len(P2) == 0:
                continue
            if len(P2) == 1:
                r = 0.0
            elif len(P2) == 2:
                r = 0.5 * float(np.linalg.norm(P2[0] - P2[1]))
            else:
                r = float(minimum_enclosing_circle(P2).radius)
            if r > worst:
                worst = r
    return worst


class G21APolicy:
    def __init__(self, mode: int = 4, opportunistic_budget: Optional[int] = None,
                 detour_clear_limit: float = DETOUR_CLEAR_LIMIT):
        self.mode = int(mode)
        assert self.mode in (3, 4)
        self.budget = opportunistic_budget if opportunistic_budget is not None else (
            OPPORTUNISTIC_BUDGET_Q3 if self.mode == 3 else OPPORTUNISTIC_BUDGET_Q4)
        self.detour_clear_limit = float(detour_clear_limit)
        self.fallback_count = 0
        self.solver_failures = 0
        self.extra_measurements = 0
        self.opportunistic_measure = 0
        self.opportunistic_direction = 0
        self.opportunistic_no_signal = 0
        self.opportunistic_near = 0
        self.legs_evaluated = 0
        self.legs_selected = 0
        self.clear_ready = 0
        self.became_clear_ready = 0
        self.zero_detour_clear = 0
        self.coverage_move_m = 0.0
        self.opportunity_move_m = 0.0
        self.resolver_move_m = 0.0
        self.clear_move_m = 0.0
        self.coverage_measure_count = 0
        self.resolver_measure_count = 0
        self.clear_calls = 0
        self.per_source = {}

    def _source_log(self, ch):
        if ch not in self.per_source:
            self.per_source[ch] = {"first_seen_time": None, "first_seen_position": None,
                                   "radius_after_first": None, "radius_after_opportunistic": None,
                                   "n_opportunistic": 0, "n_bilateral": 0,
                                   "resolver_move_distance": 0.0,
                                   "clear_certificate_time": None, "clear_time": None}
        return self.per_source[ch]

    def run(self, env: RadioEnv) -> dict:
        points = s21_points() if self.mode == 4 else __import__("brl.coverage", fromlist=["s3_points"]).s3_points()
        if hasattr(env, "coverage_points"):
            env.coverage_points = points; env.n_coverage = len(points)
        order = route_open(points, env.pos)
        tracks: Dict[int, dict] = {}

        def _measure_env(ch, q, coverage_idx, is_refine, phase):
            d0 = float(env.move_distance)
            out = env.measure(q, ch, coverage_idx=coverage_idx, is_refine=is_refine)
            dm = float(env.move_distance) - d0
            if phase == "coverage":
                self.coverage_move_m += dm; self.coverage_measure_count += 1
            elif phase == "opportunity":
                self.opportunity_move_m += dm
            else:
                self.resolver_move_m += dm; self.resolver_measure_count += 1
            return out

        def _clear_env(ch, q):
            d0 = float(env.move_distance)
            out = env.clear(q, ch)
            self.clear_move_m += float(env.move_distance) - d0
            self.clear_calls += 1
            return out

        def observe(ch, p, obs):
            typ = obs.get("measure_result")
            if typ == "near":
                out = _clear_env(ch, p)
                if out.get("clear_result") == "success":
                    self._source_log(ch)["clear_time"] = float(env.virtual_time)
                    tracks.pop(ch, None)
                else:
                    self.fallback_count += 1
                return
            if typ == "direction" and obs.get("svd_deg") is not None:
                if ch not in tracks:
                    tracks[ch] = {"first": np.asarray(p, dtype=float).copy(),
                                  "deg": float(obs["svd_deg"]),
                                  "P": initial_track_polygon(np.asarray(p, dtype=float), float(obs["svd_deg"])),
                                  "nobs": 1, "opp": 0}
                    sl = self._source_log(ch)
                    if sl["first_seen_time"] is None:
                        sl["first_seen_time"] = float(env.virtual_time)
                        sl["first_seen_position"] = np.asarray(p, dtype=float).tolist()
                else:
                    st = tracks[ch]
                    st["P"] = _clip_target_square(bearing_clip(st["P"], np.asarray(p, dtype=float),
                                                               math.radians(float(obs["svd_deg"]))))
                    st["nobs"] += 1
                if len(tracks[ch]["P"]) == 0:
                    tracks[ch]["P"] = initial_track_polygon(tracks[ch]["first"], tracks[ch]["deg"])
                c, r = _poly_center_radius(tracks[ch]["P"])
                self._source_log(ch)["radius_after_first"] = float(r)
                if r <= CLEAR_READY_RADIUS:
                    self._source_log(ch)["clear_certificate_time"] = float(env.virtual_time)

        def measure_at(ch, q, opportunistic=False, coverage_idx=None, is_refine=False):
            before_move = float(np.linalg.norm(np.asarray(q, dtype=float) - env.pos))
            phase = "opportunity" if opportunistic else ("coverage" if coverage_idx is not None else "resolver")
            out = _measure_env(ch, q, coverage_idx, is_refine, phase)
            if opportunistic:
                self.extra_measurements += 1
                self.opportunistic_measure += 1
                sl = self._source_log(ch); sl["n_opportunistic"] += 1
                tracks.get(ch, {}).setdefault("opp", 0)
                if ch in tracks:
                    tracks[ch]["opp"] += 1
                typ = out.get("measure_result")
                if typ == "direction":
                    self.opportunistic_direction += 1
                elif typ == "near":
                    self.opportunistic_near += 1
                else:
                    self.opportunistic_no_signal += 1
            observe(ch, np.asarray(q, dtype=float), out)
            if opportunistic and out.get("measure_result") == "direction" and ch in tracks:
                c2, r2 = _poly_center_radius(tracks[ch]["P"])
                self._source_log(ch)["radius_after_opportunistic"] = float(r2)
            return out, before_move

        def scan_point(idx):
            p = points[idx]
            chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
                   and idx not in env.channels[ch].scan_points]
            chs.sort(key=lambda c: (c != env.current_channel, c))
            for ch in chs:
                if env.done:
                    return
                measure_at(ch, p, opportunistic=False, coverage_idx=idx, is_refine=False)

        def leg_opportunities(B):
            A = np.asarray(env.pos, dtype=float).copy()
            B = np.asarray(B, dtype=float)

            # 保留 G25OR 的顺路滚动清除能力：已获得证书的源低绕行插入。
            best = None
            for ch, st in list(tracks.items()):
                qc0 = _best_clear_point(st["P"], env.pos, B)
                if qc0 is None:
                    continue
                det = float(np.linalg.norm(env.pos - qc0) + np.linalg.norm(qc0 - B) - np.linalg.norm(env.pos - B))
                if det <= 250.0 and (best is None or det < best[0]):
                    best = (det, ch, qc0)
            if best is not None and not env.done:
                _, ch, qc0 = best
                out0 = _clear_env(ch, qc0)
                if out0.get("clear_result") == "success":
                    self._source_log(ch)["clear_time"] = float(env.virtual_time)
                    tracks.pop(ch, None)
                else:
                    self.fallback_count += 1

            AB = B - np.asarray(env.pos, dtype=float); L = float(np.linalg.norm(AB))
            if L < 1.0:
                return
            ts = [0.1 * k for k in range(1, 10)]
            for ch, st in tracks.items():
                c, r = _poly_center_radius(st["P"])
                if np.isfinite(r) and r > CLEAR_READY_RADIUS:
                    t = float(np.dot(c - A, AB) / (L * L)); ts.append(float(min(0.95, max(0.05, t))))
            uniq = []
            for t in sorted(set(round(t, 6) for t in ts)):
                q = A + t * AB
                if not uniq or float(np.linalg.norm(q - uniq[-1])) > 1.0:
                    uniq.append(q)
            candidates = []
            for q in uniq:
                items = []
                for ch, st in tracks.items():
                    if st.get("opp", 0) >= self.budget:
                        continue
                    P = st["P"]
                    if P is None or len(P) < 1:
                        continue
                    r0 = float(_poly_center_radius(P)[1])
                    if not np.isfinite(r0) or r0 <= CLEAR_READY_RADIUS:
                        continue
                    dmax = float(np.max(np.linalg.norm(np.asarray(P, dtype=float) - q[None, :], axis=1)))
                    limit = Q3_SAFE_DIST if self.mode == 3 else Q4_MAX_DIST
                    if dmax > limit:
                        continue
                    c0, _ = _poly_center_radius(P)
                    gamma = bearing_cross_sine(st["first"], c0, q)
                    if gamma < 0.20:
                        continue
                    rw = expected_worst_radius(P, q)
                    if not np.isfinite(rw) or rw >= r0:
                        continue
                    delta = r0 - rw
                    clear_pred = rw <= CLEAR_READY_RADIUS + 1.5
                    if not clear_pred and delta < 200.0:
                        continue
                    val = (5000.0 + delta) if clear_pred else delta
                    items.append((val, ch, r0, rw))
                if not items:
                    continue
                items.sort(reverse=True, key=lambda x: x[0])
                chosen = items[:MAX_COMMON_TRACKS]
                score = sum(x[0] for x in chosen) - MEASURE_EQ_METERS * len(chosen)
                if score > 0:
                    candidates.append((score, q, chosen))
            self.legs_evaluated += 1
            if not candidates:
                return
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, q, chosen = candidates[0]
            self.legs_selected += 1
            for _, ch, _, _ in chosen:
                if env.done:
                    return
                measure_at(ch, q, opportunistic=True)
                if ch in tracks:
                    c2, r2 = _poly_center_radius(tracks[ch]["P"])
                    if np.isfinite(r2) and r2 <= CLEAR_READY_RADIUS:
                        self.became_clear_ready += 1
            if env.done:
                return
            # 当前停车点本身就在清除圆内时，零绕行立即清除。
            for ch in list(tracks.keys()):
                st = tracks.get(ch)
                if st is None:
                    continue
                c, r = _poly_center_radius(st["P"])
                if not np.isfinite(r) or r > CLEAR_READY_RADIUS:
                    continue
                dmax_q = float(np.max(np.linalg.norm(np.asarray(st["P"], dtype=float) - env.pos[None, :], axis=1)))
                if dmax_q <= 20.0 - 0.35:
                    self.zero_detour_clear += 1
                    out0 = _clear_env(ch, env.pos)
                    if out0.get("clear_result") == "success":
                        self._source_log(ch)["clear_time"] = float(env.virtual_time)
                        tracks.pop(ch, None)
                    else:
                        self.fallback_count += 1
            # 顺路清除已经获得证书的源
            for ch in list(tracks.keys()):
                st = tracks.get(ch)
                if st is None:
                    continue
                c, r = _poly_center_radius(st["P"])
                if not np.isfinite(r) or r > CLEAR_READY_RADIUS:
                    continue
                qc = _best_clear_point(st["P"], env.pos, B)
                if qc is None:
                    continue
                detour = float(np.linalg.norm(env.pos - qc) + np.linalg.norm(qc - B) - np.linalg.norm(env.pos - B))
                if detour <= self.detour_clear_limit:
                    self.clear_ready += 1
                    out = _clear_env(ch, qc)
                    if out.get("clear_result") == "success":
                        self._source_log(ch)["clear_time"] = float(env.virtual_time)
                        tracks.pop(ch, None)
                    else:
                        self.fallback_count += 1

        # 主循环：S21点依次访问，点间执行零绕行补测
        scan_point(order[0])
        for oi in range(1, len(order)):
            if env.done or env.cleared_count() + env.discovered_count() >= 16:
                break
            B = points[order[oi]]
            leg_opportunities(B)
            if env.done:
                break
            scan_point(order[oi])

        def resolve_track(ch):
            st = tracks[ch]
            first = np.asarray(st["first"], dtype=float).copy()
            deg = float(st["deg"]); P = st["P"] if len(st["P"]) > 0 else None
            sl = self._source_log(ch); sl["n_bilateral"] += 1
            start_pos = env.pos.copy()

            def measure_cb(q):
                out = _measure_env(ch, q, None, True, "resolver")
                self.extra_measurements += 1
                ans = {"measure_result": out.get("measure_result")}
                if out.get("svd_deg") is not None:
                    ans["svd_deg"] = out["svd_deg"]
                return ans

            def clear_cb(q):
                return {"clear_result": _clear_env(ch, q).get("clear_result")}

            try:
                solve_bilateral(first, deg, start_pos, measure_cb, clear_cb, initial_region=P)
            except Exception:
                self.solver_failures += 1
                for q in optical_fallback_points(first, deg):
                    if env.channels[ch].status == "cleared" or env.done:
                        break
                    if _clear_env(ch, q).get("clear_result") == "success":
                        break
                self.fallback_count += 1
            sl["resolver_move_distance"] += float(np.linalg.norm(env.pos - start_pos))
            if env.channels[ch].status == "cleared":
                sl["clear_time"] = float(env.virtual_time)
            tracks.pop(ch, None)

        while tracks and not env.done:
            ch = min(tracks.keys(), key=lambda c: float(np.linalg.norm(_poly_center_radius(tracks[c]["P"])[0] - env.pos)))
            resolve_track(ch)

        guard = 0
        while not env.done and guard < 5:
            guard += 1
            need = None
            for idx in order:
                if any(env.channels[ch].status == "unknown" and idx not in env.channels[ch].scan_points for ch in CHANNELS):
                    need = idx; break
            if need is None:
                break
            scan_point(need)
            while tracks and not env.done:
                resolve_track(next(iter(tracks)))
        return {
            "success": bool(env.success or env.completion_certificate()),
            "cleared": int(env.cleared_count()),
            "virtual_time_s": float(env.virtual_time),
            "distance_m": float(env.move_distance),
            "measure_calls": int(env.n_measure),
            "switches": int(env.n_switch),
            "clear_calls": int(env.n_clear),
            "failed_clear": int(env.n_clear_fail),
            "solver_failures": int(self.solver_failures),
            "optical_fallbacks": int(self.fallback_count),
            "extra_measurements": int(self.extra_measurements),
            "opportunistic_measure_count": int(self.opportunistic_measure),
            "opportunistic_direction_count": int(self.opportunistic_direction),
            "opportunistic_no_signal_count": int(self.opportunistic_no_signal),
            "opportunistic_near_count": int(self.opportunistic_near),
            "legs_evaluated": int(self.legs_evaluated),
            "legs_selected": int(self.legs_selected),
            "clear_ready_count": int(self.clear_ready),
            "became_clear_ready_count": int(self.became_clear_ready),
            "zero_detour_clear_count": int(self.zero_detour_clear),
            "coverage_move_m": float(self.coverage_move_m),
            "opportunity_move_m": float(self.opportunity_move_m),
            "resolver_move_m": float(self.resolver_move_m),
            "clear_move_m": float(self.clear_move_m),
            "coverage_measure_count": int(self.coverage_measure_count),
            "resolver_measure_count": int(self.resolver_measure_count),
            "clear_calls": int(self.clear_calls),
            "per_source": self.per_source,
        }