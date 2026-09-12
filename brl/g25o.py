"""几何主动搜索策略 G25O：S25覆盖 + 双侧区间定位 + 机会式补测 + 开路路线。

这是本地候选策略，不是 RL 网络。它可以直接在 RadioEnv 上运行，用于和
现有 A0/PPO 做严格配对比较，也可以作为正式 runner 的候选底层。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bilateral import K, bearing_clip, clip, solve_bilateral
from .certificates import (cell_index_of_point, grid_cells_intersecting_disk,
                           q3_absent_cells, q4_absent_cells)
from .coverage import s25_points, s21_points, s3_points, s4_points
from .geometry import bearing_cross_sine, minimum_enclosing_circle
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

    def __init__(self, variant: str = "G25O", coverage: str = "S25", enable_q4_cert: bool = True):
        self.variant = variant
        self.enable_q4_cert = bool(enable_q4_cert)
        self.coverage = str(coverage).upper()
        if self.coverage not in ("S25", "S21", "S4"):
            raise ValueError("coverage must be S25, S21 or S4")
        self.full_rolling = variant in ("G25O-R", "G25ORFull", "G25ORF", "G25OR+")
        self.rolling = variant in ("G25OR", "G25O-R", "G25OR+") or self.full_rolling
        self.fallback_count = 0
        self.solver_failures = 0
        self.extra_measurements = 0
        self.dynamic_absent_channels = 0
        self.dynamic_absent_cells = 0

    def coverage_set(self, mode: int) -> np.ndarray:
        if int(mode) == 3:
            return s3_points()
        if self.coverage == "S21":
            return s21_points()
        if self.coverage == "S4":
            return s4_points()
        return s25_points()

    def run(self, env: RadioEnv, resume: bool = False) -> dict:
        if self.full_rolling:
            return _run_full(self, env)
        mode = env.mode
        points = self.coverage_set(mode)
        if hasattr(env, "coverage_points"):
            env.coverage_points = points
            env.n_coverage = len(points)
        order = route_open(points, env.pos)
        tracks: Dict[int, dict] = {}
        if resume:
            # 从已有环境状态恢复：把已发现但未清除的频道转成 track，
            # 供续行基策略继续定位清除，而不是从空 tracks 重新开始。
            for ch, st in env.channels.items():
                if st.status != "discovered":
                    continue
                first = None; deg = None
                for ob in st.observations:
                    if ob.get("result") == "direction" and ob.get("svd_deg") is not None:
                        first = np.asarray(ob["position"], dtype=float)
                        deg = float(ob["svd_deg"]); break
                if first is None or deg is None:
                    continue
                P = st.poly if st.poly is not None and len(st.poly) > 0 else initial_track_polygon(first, deg)
                tracks[int(ch)] = {"first": first.copy(), "deg": deg, "P": P,
                                   "nobs": max(1, sum(1 for ob in st.observations if ob.get("result") == "direction"))}

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


def _circle_rect_min_dist(center, half, radius=1800.0):
    """圆心到轴对齐矩形的最近距离；用于保留与目标圆相交的闭单元。"""
    c = np.asarray(center, dtype=float)
    dx = max(abs(c[0]) - half, 0.0)
    dy = max(abs(c[1]) - half, 0.0)
    return math.hypot(dx, dy)


def _q3_absence_grid(spacing: float = 200.0):
    """返回与目标圆相交的闭正方形单元中心和半边长。

    旧实现只保留中心位于圆内的格子，会在圆边界留下未覆盖细片。
    新实现用圆到矩形最近距离 <= 1800 保留所有相交单元；证书检查时
    使用单元四个角点，而不是中心加外接圆半径。
    """
    global _Q3_CELLS, _Q3_CELL_R
    if _Q3_CELLS is None:
        cells = []
        limit = 1800.0
        half = spacing / 2.0
        k = int(limit / spacing) + 2
        for i in range(-k, k + 1):
            x = i * spacing
            for j in range(-k, k + 1):
                y = j * spacing
                if _circle_rect_min_dist((x, y), half, limit) <= limit + 1e-9:
                    cells.append((x, y))
        _Q3_CELLS = np.asarray(cells, dtype=float)
        _Q3_CELL_R = spacing * math.sqrt(2.0) / 2.0
    return _Q3_CELLS, _Q3_CELL_R


def _q3_proven_absent(neg_points, cells, cell_radius=None, margin: float = 1e-6) -> bool:
    """Q3 全向源不存在证书。

    对一个闭单元，只要存在一个 no_signal 测点 p 满足 p 到单元四个角点
    距离都 <= 1000 - margin，则单元内任意全向源都应被 p 接收，与实测
    矛盾，因此整个单元可排除。所有保留单元都被排除后，频道才标记 absent。
    """
    pts = np.asarray(neg_points, dtype=float)
    if len(pts) == 0 or cells is None or len(cells) == 0:
        return False
    # cell_radius 是旧接口传入的外接圆半径；正方形半边长 = R/sqrt(2)
    if cell_radius is not None and float(cell_radius) > 0:
        half = float(cell_radius) / math.sqrt(2.0)
    else:
        half = 100.0
    safe_r = 1000.0 - margin
    corners = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    for c in np.asarray(cells, dtype=float):
        cc = c[None, :] + corners
        # 距离的 max 是凸函数，矩形上最大值在角点取到
        d = np.linalg.norm(pts[:, None, :] - cc[None, :, :], axis=2)
        if not np.any(np.max(d, axis=1) <= safe_r):
            return False
    return True

def _run_full(policy: G25OPolicy, env: RadioEnv) -> dict:
    """完整滚动 G25O-R：动态选点 + 机会测向 + 滚动清除 + Q3动态无源证书。

    所有启发式只影响顺序和是否提前试清，不会绕过S3/S25覆盖证明、
    双侧区间定位和最终光学保底。
    """
    mode = env.mode
    points = policy.coverage_set(mode)
    if hasattr(env, "coverage_points"):
        env.coverage_points = points
        env.n_coverage = len(points)
    unvisited = set(range(len(points)))
    tracks: Dict[int, dict] = {}
    neg_points: Dict[int, list] = {ch: [] for ch in CHANNELS}
    proven_absent: set = set()
    if mode == 3:
        cells, cell_radius = _q3_absence_grid()
        q4_absent = None
    else:
        if policy.enable_q4_cert:
            cells, half = grid_cells_intersecting_disk(200.0, 1800.0)
            cell_radius = half
            q4_absent = {ch: np.zeros(len(cells), dtype=bool) for ch in CHANNELS}
        else:
            cells, cell_radius, q4_absent = None, None, None

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
        if cells is None:
            return
        if mode == 3:
            for ch in CHANNELS:
                if env.channels[ch].status != "unknown" or ch in proven_absent:
                    continue
                if _q3_proven_absent(neg_points[ch], cells, cell_radius):
                    env.channels[ch].status = "absent"
                    proven_absent.add(ch)
                    policy.dynamic_absent_channels += 1
            return
        # Q4：位置-方向局部凸包证书；只有全部单元被排除才标记频道不存在
        if q4_absent is None:
            return
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if len(neg_points[ch]) < 3:
                continue
            new = q4_absent_cells(neg_points[ch], cells, cell_radius) & (~q4_absent[ch])
            if new.any():
                q4_absent[ch] |= new
                policy.dynamic_absent_cells += int(new.sum())
            if bool(q4_absent[ch].all()):
                env.channels[ch].status = "absent"
                proven_absent.add(ch)
                policy.dynamic_absent_channels += 1

    def scan_point(idx: int) -> None:
        p = points[idx]
        cell_idx = cell_index_of_point(p, cells, cell_radius) if (mode == 4 and cells is not None) else -1
        chs = []
        for ch in CHANNELS:
            if env.channels[ch].status != "unknown" or ch in proven_absent:
                continue
            if idx in env.channels[ch].scan_points:
                continue
            if mode == 4 and q4_absent is not None and cell_idx >= 0 and q4_absent[ch][cell_idx]:
                continue
            chs.append(ch)
        chs.sort(key=lambda c: (c != env.current_channel, c))
        for ch in chs:
            if env.done:
                return
            measure_and_observe(ch, p, coverage_idx=idx, is_refine=False)
        update_absence()

    def geom_angle_bonus(idx: int) -> float:
        """在候选源估计位置处计算两条观测方向的交叉正弦。

        旧实现计算的是候选测点 p 处的夹角，会把共线观测奖励成 180 度。
        新实现使用 |cross(s1-g, p-g)|/(|s1-g||p-g|)，共线时为 0。
        """
        p = points[idx]
        bonus = 0.0
        for ch, st in tracks.items():
            c, r = _poly_center_radius(st["P"])
            if not np.isfinite(r):
                continue
            g = np.asarray(c, dtype=float)
            a = np.asarray(st["first"], dtype=float)
            sine = bearing_cross_sine(a, g, p)
            if sine <= 0.0 and float(np.linalg.norm(a - g)) > 1e-6 and float(np.linalg.norm(p - g)) > 1e-6:
                continue
            ang = math.asin(min(1.0, max(0.0, float(sine))))
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
            sine = bearing_cross_sine(a, c, p)
            d = float(np.linalg.norm(p - c))
            thr = 0.15 if mode == 3 else 0.12
            if d <= 1200.0 and sine > thr:
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
        "dynamic_absent_channels": int(policy.dynamic_absent_channels),
        "dynamic_absent_cells": int(policy.dynamic_absent_cells),
    }