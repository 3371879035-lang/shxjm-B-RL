"""几何主动搜索策略 G25O：S25覆盖 + 双侧区间定位 + 机会式补测 + 开路路线。

这是本地候选策略，不是 RL 网络。它可以直接在 RadioEnv 上运行，用于和
现有 A0/PPO 做严格配对比较，也可以作为正式 runner 的候选底层。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bilateral import K, bearing_clip, clip, solve_bilateral
from .coverage import s25_points, s3_points, s4_points
from .geometry import minimum_enclosing_circle
from .local_env import CHANNELS, RadioEnv
from .resolver import optical_fallback_points


def route_open(points: np.ndarray, start: Sequence[float]) -> List[int]:
    pts = np.asarray(points, dtype=float)
    ps = np.vstack([np.asarray(start, dtype=float).reshape(1, 2), pts])
    D = np.linalg.norm(ps[:, None, :] - ps[None, :, :], axis=-1)
    rem = set(range(1, len(ps)))
    seq = [0]
    while rem:
        i = min(rem, key=lambda k: (D[seq[-1], k], k))
        seq.append(i)
        rem.remove(i)
    for _ in range(30):
        improved = False
        for i in range(1, len(seq) - 1):
            for j in range(i + 1, len(seq)):
                old = D[seq[i - 1], seq[i]]
                new = D[seq[i - 1], seq[j]]
                if j + 1 < len(seq):
                    old += D[seq[j], seq[j + 1]]
                    new += D[seq[i], seq[j + 1]]
                if new < old - 1e-8:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return [i - 1 for i in seq[1:]]


def _clip_target_square(poly: np.ndarray) -> np.ndarray:
    for n, c in ((np.array([1.0, 0.0]), 1800.0), (np.array([-1.0, 0.0]), 1800.0),
                 (np.array([0.0, 1.0]), 1800.0), (np.array([0.0, -1.0]), 1800.0)):
        poly = clip(poly, n, c)
    return poly


def initial_track_polygon(p: np.ndarray, theta_deg: float) -> np.ndarray:
    # 严格外包半径1500m的扇形：用远边弦在中央方向投影为1500m。
    # 两个边界点取 p + (1500/cos eps) * u(theta±eps)，三角形包含整个扇形。
    eps = math.radians(1.01)
    th = math.radians(theta_deg)
    L = 1500.0 / math.cos(eps)
    u1 = np.array([math.cos(th - eps), math.sin(th - eps)])
    u2 = np.array([math.cos(th + eps), math.sin(th + eps)])
    poly = np.asarray([p, p + L * u1, p + L * u2], dtype=float)
    return _clip_target_square(poly)


def _poly_center_radius(P: np.ndarray) -> Tuple[np.ndarray, float]:
    if P is None or len(P) == 0:
        return np.zeros(2), float("inf")
    mec = minimum_enclosing_circle(P)
    return np.asarray(mec.center, dtype=float), float(mec.radius)


def _best_clear_point(P: np.ndarray, A: Sequence[float], B: Sequence[float],
                      margin: float = 0.35) -> Optional[np.ndarray]:
    """在保证一次clear成功的区域内，找使 A->q->B 绕行最小的清除点。

    若可行域最小包围圆半径 r<=19.5，则圆 B(c, 20-r-margin) 内任意 q
    都保证与 P 中任意点距离 <=20。返回 None 表示当前没有清除证书。
    """
    c, r = _poly_center_radius(P)
    if r > 19.5:
        return None
    slack = 20.0 - r - margin
    if slack <= 1e-9:
        return None
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    AB = B - A
    L = float(np.linalg.norm(AB))
    cands = [c]
    if L > 1e-9:
        t = float(np.dot(c - A, AB) / (L * L))
        t = min(1.0, max(0.0, t))
        proj = A + t * AB
        v = proj - c
        nv = float(np.linalg.norm(v))
        if nv <= slack:
            cands.append(proj)
        else:
            cands.append(c + v / nv * slack)
    valid = [q for q in cands if float(np.linalg.norm(q - c)) <= slack + 1e-9]
    if not valid:
        return None
    return min(valid, key=lambda q: float(np.linalg.norm(A - q) + np.linalg.norm(q - B)))


class G25OPolicy:
    """机会式几何主动搜索策略。"""

    def __init__(self, variant: str = "G25O"):
        self.variant = variant
        self.full_rolling = variant in ("G25O-R", "G25ORFull", "G25ORF", "G25OR+")
        self.rolling = variant in ("G25OR", "G25O-R", "G25OR+") or self.full_rolling
        self.fallback_count = 0
        self.solver_failures = 0
        self.extra_measurements = 0

    def run(self, env: RadioEnv) -> dict:
        if self.full_rolling:
            return _run_full(self, env)
        mode = env.mode
        if mode == 3:
            points = s3_points()
        else:
            points = s25_points() if self.variant in ("G25", "G25O", "G25OR", "G25O-R", "G25OR+") else s4_points()
        order = route_open(points, env.pos)
        tracks: Dict[int, dict] = {}

        def observe(ch: int, p: np.ndarray, obs: dict) -> None:
            typ = obs.get("measure_result")
            if typ == "near":
                out = env.clear(p, ch)
                if out.get("clear_result") == "success":
                    tracks.pop(ch, None)
                else:
                    self.fallback_count += 1
                return
            if typ == "direction" and obs.get("svd_deg") is not None:
                if ch not in tracks:
                    tracks[ch] = {
                        "first": p.copy(),
                        "deg": float(obs["svd_deg"]),
                        "P": initial_track_polygon(p, float(obs["svd_deg"])),
                        "nobs": 1,
                    }
                else:
                    st = tracks[ch]
                    st["P"] = _clip_target_square(
                        bearing_clip(st["P"], p, math.radians(float(obs["svd_deg"]))))
                    st["nobs"] += 1
                if len(tracks[ch]["P"]) == 0:
                    tracks[ch]["P"] = initial_track_polygon(tracks[ch]["first"], tracks[ch]["deg"])

        def measure_and_observe(ch: int, p: np.ndarray, coverage_idx: Optional[int] = None,
                                is_refine: bool = False) -> dict:
            out = env.measure(p, ch, coverage_idx=coverage_idx, is_refine=is_refine)
            observe(ch, p, out)
            if is_refine:
                self.extra_measurements += 1
            return out

        def scan_point(idx: int, allow_opportunistic: bool) -> None:
            p = points[idx]
            old_tracks = list(tracks)
            chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
                   and idx not in env.channels[ch].scan_points]
            chs.sort(key=lambda c: (c != env.current_channel, c))
            for ch in chs:
                if env.done:
                    return
                measure_and_observe(ch, p, coverage_idx=idx, is_refine=False)
            if not allow_opportunistic or env.done:
                return
            for ch in old_tracks:
                if ch not in tracks:
                    continue
                st = tracks[ch]
                c, rad = _poly_center_radius(st["P"])
                if rad <= 19.5 or st["nobs"] >= 4:
                    continue
                delta = c - p
                d = float(np.linalg.norm(delta))
                w = c - st["first"]
                w0 = float(np.linalg.norm(w))
                if w0 < 1e-9:
                    continue
                sine = abs(delta[0] * w[1] - delta[1] * w[0]) / (d * w0 + 1e-9)
                if d <= 1200.0 and sine > 0.15:
                    measure_and_observe(ch, p, coverage_idx=None, is_refine=True)

        for oi, idx in enumerate(order):
            if env.done:
                break
            if env.cleared_count() + env.discovered_count() >= 16:
                break
            scan_point(idx, allow_opportunistic=(self.variant in ("G25O", "G25OR", "G25O-R", "G25OR+")))
            if env.done:
                break
            if self.rolling and tracks and oi + 1 < len(order):
                next_p = points[order[oi + 1]]
                # 当前已能保证清除的源，若插入路线绕行不大，就地清除。
                best = None
                for ch, st in list(tracks.items()):
                    q = _best_clear_point(st["P"], env.pos, next_p)
                    if q is None:
                        continue
                    detour = float(np.linalg.norm(env.pos - q) + np.linalg.norm(q - next_p)
                                   - np.linalg.norm(env.pos - next_p))
                    if detour <= 250.0 and (best is None or detour < best[0]):
                        best = (detour, ch, q)
                if best is not None:
                    _, ch, q = best
                    out = env.clear(q, ch)
                    if out.get("clear_result") == "success":
                        tracks.pop(ch, None)
                    else:
                        self.fallback_count += 1
            if env.done:
                break

        def resolve_track(ch: int) -> None:
            st = tracks[ch]
            first = st["first"].copy()
            first_deg = float(st["deg"])
            P = st["P"] if len(st["P"]) > 0 else None

            def measure_cb(q, ch=ch):
                out = env.measure(q, ch, is_refine=True)
                self.extra_measurements += 1
                ans = {"measure_result": out.get("measure_result")}
                if out.get("svd_deg") is not None:
                    ans["svd_deg"] = out["svd_deg"]
                return ans

            def clear_cb(q, ch=ch):
                out = env.clear(q, ch)
                return {"clear_result": out.get("clear_result")}

            try:
                solve_bilateral(first, first_deg, env.pos, measure_cb, clear_cb, initial_region=P)
            except Exception:
                self.solver_failures += 1
                for q in optical_fallback_points(first, first_deg):
                    if env.channels[ch].status == "cleared":
                        break
                    if env.clear(q, ch).get("clear_result") == "success":
                        break
                self.fallback_count += 1
            tracks.pop(ch, None)

        while tracks and not env.done:
            ch = min(tracks.keys(),
                     key=lambda c: float(np.linalg.norm(_poly_center_radius(tracks[c]["P"])[0] - env.pos)))
            resolve_track(ch)

        guard = 0
        while not env.done and guard < 5:
            guard += 1
            need_idx = None
            for idx in order:
                if any(env.channels[ch].status == "unknown" and idx not in env.channels[ch].scan_points
                       for ch in CHANNELS):
                    need_idx = idx
                    break
            if need_idx is None:
                break
            scan_point(need_idx, allow_opportunistic=(self.variant in ("G25O", "G25OR", "G25O-R", "G25OR+")))
            while tracks and not env.done:
                ch = next(iter(tracks))
                resolve_track(ch)

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
        }

# ---------------- 完整滚动版本 G25O-R ----------------
_Q3_CELLS = None
_Q3_CELL_R = None


def _q3_absence_grid(spacing: float = 200.0):
    global _Q3_CELLS, _Q3_CELL_R
    if _Q3_CELLS is None:
        cells = []
        limit = 1800.0
        k = int(limit / spacing) + 2
        for i in range(-k, k + 1):
            x = i * spacing
            for j in range(-k, k + 1):
                y = j * spacing
                if x * x + y * y <= limit * limit + 1e-9:
                    cells.append((x, y))
        _Q3_CELLS = np.asarray(cells, dtype=float)
        _Q3_CELL_R = spacing * math.sqrt(2.0) / 2.0
    return _Q3_CELLS, _Q3_CELL_R


def _q3_proven_absent(neg_points, cells, cell_radius, margin: float = 5.0) -> bool:
    pts = np.asarray(neg_points, dtype=float)
    if len(pts) == 0:
        return False
    safe_r = 1000.0 - cell_radius - margin
    if safe_r <= 0:
        return False
    for c in cells:
        if not np.any(np.linalg.norm(pts - c, axis=1) <= safe_r):
            return False
    return True

def _run_full(policy: G25OPolicy, env: RadioEnv) -> dict:
    """完整滚动 G25O-R：动态选点 + 机会测向 + 滚动清除 + Q3动态无源证书。

    所有启发式只影响顺序和是否提前试清，不会绕过S3/S25覆盖证明、
    双侧区间定位和最终光学保底。
    """
    mode = env.mode
    points = s3_points() if mode == 3 else s25_points()
    unvisited = set(range(len(points)))
    tracks: Dict[int, dict] = {}
    neg_points: Dict[int, list] = {ch: [] for ch in CHANNELS}
    proven_absent: set = set()
    if mode == 3:
        cells, cell_radius = _q3_absence_grid()
    else:
        cells, cell_radius = None, None

    def observe(ch: int, p: np.ndarray, obs: dict) -> None:
        typ = obs.get("measure_result")
        if typ == "near":
            out = env.clear(p, ch)
            if out.get("clear_result") == "success":
                tracks.pop(ch, None)
            else:
                policy.fallback_count += 1
            return
        if typ == "direction" and obs.get("svd_deg") is not None:
            if ch not in tracks:
                tracks[ch] = {
                    "first": p.copy(),
                    "deg": float(obs["svd_deg"]),
                    "P": initial_track_polygon(p, float(obs["svd_deg"])),
                    "nobs": 1,
                }
            else:
                st = tracks[ch]
                st["P"] = _clip_target_square(
                    bearing_clip(st["P"], p, math.radians(float(obs["svd_deg"]))))
                st["nobs"] += 1
            if len(tracks[ch]["P"]) == 0:
                tracks[ch]["P"] = initial_track_polygon(tracks[ch]["first"], tracks[ch]["deg"])

    def measure_and_observe(ch: int, p: np.ndarray, coverage_idx=None, is_refine=False) -> dict:
        out = env.measure(p, ch, coverage_idx=coverage_idx, is_refine=is_refine)
        observe(ch, p, out)
        if is_refine:
            policy.extra_measurements += 1
        if out.get("measure_result") == "no_signal" and env.channels[ch].status == "unknown":
            neg_points[ch].append(p.copy())
        return out

    def update_absence() -> None:
        if mode != 3 or cells is None:
            return
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if _q3_proven_absent(neg_points[ch], cells, cell_radius):
                env.channels[ch].status = "absent"
                proven_absent.add(ch)

    def scan_point(idx: int) -> None:
        p = points[idx]
        chs = [ch for ch in CHANNELS if env.channels[ch].status == "unknown"
               and ch not in proven_absent and idx not in env.channels[ch].scan_points]
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                return
            measure_and_observe(ch, p, coverage_idx=idx, is_refine=False)
        update_absence()

    def geom_angle_bonus(idx: int) -> float:
        p = points[idx]
        bonus = 0.0
        for ch, st in tracks.items():
            c, r = _poly_center_radius(st["P"])
            if not np.isfinite(r):
                continue
            a = st["first"]
            v1 = a - p
            v2 = c - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1.0 or n2 < 1.0:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = min(1.0, max(-1.0, cosang))
            ang = math.acos(cosang)
            d = float(np.linalg.norm(p - c))
            if mode == 3:
                if d <= 1300.0:
                    bonus = max(bonus, 450.0 * ang)
            else:
                if d <= 1400.0:
                    bonus = max(bonus, 220.0 * ang)
        return bonus

    def choose_next(active):
        if not active:
            return None
        best = None
        for idx in active:
            p = points[idx]
            travel = float(np.linalg.norm(p - env.pos))
            score = travel - geom_angle_bonus(idx)
            if best is None or score < best[0]:
                best = (score, idx)
        return best[1]

    def try_rolling_clear(next_idx: int) -> None:
        if not tracks or next_idx is None:
            return
        best = None
        nxt = points[next_idx]
        for ch, st in list(tracks.items()):
            q = _best_clear_point(st["P"], env.pos, nxt)
            if q is None:
                continue
            detour = float(np.linalg.norm(env.pos - q) + np.linalg.norm(q - nxt)
                           - np.linalg.norm(env.pos - nxt))
            if detour <= 300.0 and (best is None or detour < best[0]):
                best = (detour, ch, q)
        if best is not None:
            _, ch, q = best
            out = env.clear(q, ch)
            if out.get("clear_result") == "success":
                tracks.pop(ch, None)
            else:
                policy.fallback_count += 1

    def resolve_track(ch: int) -> None:
        st = tracks[ch]
        first = st["first"].copy()
        first_deg = float(st["deg"])
        P = st["P"] if len(st["P"]) > 0 else None

        def measure_cb(q, ch=ch):
            out = env.measure(q, ch, is_refine=True)
            policy.extra_measurements += 1
            ans = {"measure_result": out.get("measure_result")}
            if out.get("svd_deg") is not None:
                ans["svd_deg"] = out["svd_deg"]
            return ans

        def clear_cb(q, ch=ch):
            out = env.clear(q, ch)
            return {"clear_result": out.get("clear_result")}

        try:
            solve_bilateral(first, first_deg, env.pos, measure_cb, clear_cb, initial_region=P)
        except Exception:
            policy.solver_failures += 1
            for q in optical_fallback_points(first, first_deg):
                if env.channels[ch].status == "cleared":
                    break
                if env.clear(q, ch).get("clear_result") == "success":
                    break
            policy.fallback_count += 1
        tracks.pop(ch, None)

    # 主滚动循环
    while not env.done:
        if env.cleared_count() + env.discovered_count() >= 16:
            break
        active = [idx for idx in unvisited
                  if any(env.channels[ch].status == "unknown" and ch not in proven_absent
                         and idx not in env.channels[ch].scan_points for ch in CHANNELS)]
        if not active:
            break
        idx = choose_next(active)
        if idx is None:
            break
        scan_point(idx)
        unvisited.discard(idx)
        if env.done:
            break
        # 当前停点的机会式补测：对仍未清除的 tracks 尽量顺手多测一次。
        p = points[idx]
        for ch in list(tracks.keys()):
            st = tracks.get(ch)
            if st is None:
                continue
            c, r = _poly_center_radius(st["P"])
            if r <= 19.5 or st["nobs"] >= 4:
                continue
            a = st["first"]
            v1 = a - p
            v2 = c - p
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1.0 or n2 < 1.0:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = min(1.0, max(-1.0, cosang))
            ang = math.acos(cosang)
            d = float(np.linalg.norm(p - c))
            thr = 0.15 if mode == 3 else 0.12
            if d <= 1200.0 and ang > thr:
                measure_and_observe(ch, p, coverage_idx=None, is_refine=True)
                update_absence()
        # 若已有清除证书，尝试低绕行插入清除。
        active2 = [i for i in unvisited
                   if any(env.channels[ch].status == "unknown" and ch not in proven_absent
                          and i not in env.channels[ch].scan_points for ch in CHANNELS)]
        if active2 and tracks:
            try_rolling_clear(choose_next(active2))
        if not active2 and tracks:
            # 没有扫描义务时，优先解决最近的 track，避免空转。
            break

    while tracks and not env.done:
        ch = min(tracks.keys(),
                 key=lambda c: float(np.linalg.norm(_poly_center_radius(tracks[c]["P"])[0] - env.pos)))
        resolve_track(ch)

    # 补齐剩余未知频道（Q3可能已被动态证书标记；Q4仍需走完S25）
    guard = 0
    while not env.done and guard < 5:
        guard += 1
        need = None
        for idx in unvisited:
            if any(env.channels[ch].status == "unknown" and ch not in proven_absent
                   and idx not in env.channels[ch].scan_points for ch in CHANNELS):
                need = idx
                break
        if need is None:
            break
        scan_point(need)
        unvisited.discard(need)
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
        "solver_failures": int(policy.solver_failures),
        "optical_fallbacks": int(policy.fallback_count),
        "extra_measurements": int(policy.extra_measurements),
    }