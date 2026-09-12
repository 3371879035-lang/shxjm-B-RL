"""本地训练/评估环境。

它不是官方模拟器的替代品，只用于在没有官方接口时训练和验证调度算法。
所有随机源、误差场和反馈都必须与题面硬规则一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, pi, radians, sin, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .coverage import s3_points, s4_points
from .geometry import (DOMAIN_RADIUS, MAX_RECEIVE_RADIUS, circle_outer_halfplanes,
                       circle_outer_polygon, conservative_clear_certificate, domain_polygon,
                       intersect_halfplanes, minimum_enclosing_circle, perpendicular,
                       polygon_centroid, wedge_halfplanes)
from .resolver import optical_fallback_points, reliable_clear
from .protocol import OFFICIAL_BEARING_ENVELOPE_DEG, quantize_bearing_deg

CHANNELS = list(range(1, 21))
MAX_SOURCES = 16
MIN_SOURCES = 10
NEAR_DIST = 5.0
CLEAR_DIST = 20.0
CLEAR_MARGIN = 19.5


@dataclass
class Source:
    channel: int
    position: np.ndarray
    radius: float
    kind: str = "omni"          # omni | directional
    direction_deg: float = 0.0  # 定向源发射方向；全向源忽略


@dataclass
class ChannelState:
    channel: int
    status: str = "unknown"     # unknown | discovered | cleared | absent
    observations: List[dict] = field(default_factory=list)
    scan_points: set = field(default_factory=set)
    poly: np.ndarray = field(default_factory=lambda: domain_polygon())
    has_direction: bool = False
    has_near: bool = False
    near_pos: Optional[np.ndarray] = None
    local_extra_count: int = 0
    fallback_started: bool = False
    clear_fail_count: int = 0
    type_hint: str = "unknown"  # unknown | omni | directional，仅辅助，不作为硬证据
    no_signal_after_dir: int = 0
    last_result: str = "none"
    _poly_radius: float = 0.0
    _poly_center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def _refresh_cache(self) -> None:
        if self.poly is None or len(self.poly) == 0:
            self._poly_radius = 0.0
            self._poly_center = np.zeros(2, dtype=float)
            return
        c = minimum_enclosing_circle(self.poly)
        self._poly_radius = float(c.radius)
        self._poly_center = np.asarray(c.center, dtype=float)

    def update_poly(self, poly: np.ndarray) -> None:
        if poly is not None and len(poly) > 0:
            self.poly = poly
            self._refresh_cache()
        else:
            # 若代理保守近似导致空集，保留原集合；保证层不能随意删除真实位置。
            pass

    @property
    def poly_radius(self) -> float:
        return float(self._poly_radius)

    @property
    def poly_center(self) -> np.ndarray:
        return self._poly_center


def normalize_deg(a: float) -> float:
    return float(a % 360.0)


class SpatialErrorField:
    """每局每频道固定的空间误差场，值域[-1,1]度。

    同一地点重复测量返回相同误差，不会因重复测量而平均掉误差。
    """
    def __init__(self, rng: np.random.Generator):
        # 每频道随机相位；训练集可随机化，评估集固定种子复现。
        self.phases = rng.uniform(0, 2 * pi, size=(21, 4))
        self.amps = rng.uniform(0.25, 0.85, size=(21, 4))

    def value(self, channel: int, x: float, y: float) -> float:
        p = self.phases[channel]
        a = self.amps[channel]
        v = (a[0] * sin(0.00090 * x + 0.00110 * y + p[0]) +
             a[1] * sin(0.00035 * x - 0.00130 * y + p[1]) +
             a[2] * sin(-0.00150 * x + 0.00070 * y + p[2]) +
             a[3] * sin(0.00065 * x + 0.00045 * y + p[3]))
        # 归一化到[-1,1]；理论上四分量和<=3.4，乘0.29并截断。
        v = v * 0.29
        return float(np.clip(v, -1.0, 1.0))


def random_sources(mode: int, n_sources: int, rng: np.random.Generator,
                   edge_bias: float = 0.15, cluster: bool = False,
                   directional_fraction: float = 0.45) -> List[Source]:
    """生成合法源集合。训练分布是我们设定的，不冒充官方分布。"""
    n_sources = int(np.clip(n_sources, MIN_SOURCES, MAX_SOURCES))
    chosen = rng.choice(CHANNELS, size=n_sources, replace=False).tolist()
    sources: List[Source] = []
    # 可选一个聚集中心
    if cluster:
        center = rng.normal(0, 450, size=2)
        center = center / max(np.linalg.norm(center), 1e-9) * min(np.linalg.norm(center), 1200)
    for ch in chosen:
        if cluster and rng.random() < 0.65:
            pos = center + rng.normal(0, 240, size=2)
            if np.linalg.norm(pos) > DOMAIN_RADIUS:
                pos = pos / np.linalg.norm(pos) * (DOMAIN_RADIUS * rng.uniform(0.85, 1.0))
        else:
            r = DOMAIN_RADIUS * (rng.random() ** 0.5)
            if rng.random() < edge_bias:
                r = DOMAIN_RADIUS * rng.uniform(0.90, 1.0)
            th = rng.uniform(0, 2 * pi)
            pos = np.array([r * cos(th), r * sin(th)])
        radius = float(rng.uniform(1000.0, 1500.0))
        if rng.random() < 0.25:
            radius = float(rng.uniform(1000.0, 1080.0))
        kind = "omni"
        direction = 0.0
        if mode == 4 and rng.random() < directional_fraction:
            kind = "directional"
            # 一部分朝外，制造背向失联困难
            if rng.random() < 0.5 and np.linalg.norm(pos) > 1e-6:
                direction = normalize_deg(degrees(atan2(pos[1], pos[0])) + rng.normal(0, 25))
            else:
                direction = rng.uniform(0, 360)
        sources.append(Source(ch, pos, radius, kind, direction))
    return sources


class RadioEnv:
    """遵守题面动作计时与反馈规则的本地二维环境。"""
    def __init__(self, mode: int = 3, n_sources: int = 12, seed: Optional[int] = None,
                 step_limit: int = 20000, virtual_limit: float = 360000.0,
                 edge_bias: float = 0.15, cluster: bool = False,
                 bearing_decimals: Optional[int] = None):
        assert mode in (3, 4)
        self.mode = mode
        self.requested_n = n_sources
        self.base_seed = seed
        self.step_limit = step_limit
        self.virtual_limit = virtual_limit
        self.edge_bias = edge_bias
        self.cluster = cluster
        self.bearing_decimals = bearing_decimals
        self.coverage_points = s3_points() if mode == 3 else s4_points()
        self.n_coverage = len(self.coverage_points)
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, n_sources: Optional[int] = None,
              sources: Optional[List[Source]] = None) -> None:
        if seed is not None:
            self.base_seed = seed
        self.rng = np.random.default_rng(self.base_seed)
        self.error_field = SpatialErrorField(self.rng)
        if sources is None:
            n = self.requested_n if n_sources is None else n_sources
            self.sources = random_sources(self.mode, n, self.rng, edge_bias=self.edge_bias,
                                          cluster=self.cluster)
        else:
            self.sources = [Source(s.channel, np.asarray(s.position, dtype=float), s.radius,
                                   s.kind, s.direction_deg) for s in sources]
        self.source_by_channel = {s.channel: s for s in self.sources}
        self.pos = np.zeros(2, dtype=float)
        self.current_channel = 1
        self.virtual_time = 0.0
        self.step_count = 0
        self.done = False
        self.success = False
        self.channels: Dict[int, ChannelState] = {ch: ChannelState(ch) for ch in CHANNELS}
        for st in self.channels.values():
            st._refresh_cache()
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_fail = 0
        self.n_switch = 0
        self.move_distance = 0.0
        self.scan_actions = 0
        self.refine_actions = 0
        self.fallback_actions = 0
        # 当前解算出的动作映射，由 MacroEnv 使用；环境本身不依赖它。
        self.last_action_info = {}

    # ---------------- 反馈计算 ----------------
    def _visible(self, src: Source, p: np.ndarray) -> bool:
        d = float(np.linalg.norm(p - src.position))
        if d > src.radius + 1e-9:
            return False
        if src.kind == "directional":
            n = np.array([cos(radians(src.direction_deg)), sin(radians(src.direction_deg))])
            if float(np.dot(n, p - src.position)) < -1e-9:
                return False
        return True

    def _bearing(self, p: np.ndarray, target: np.ndarray) -> float:
        return normalize_deg(degrees(atan2(target[1] - p[1], target[0] - p[0])))

    def measure(self, position: Sequence[float], channel: int,
                coverage_idx: Optional[int] = None,
                is_refine: bool = False) -> dict:
        p = np.asarray(position, dtype=float)
        if p.shape != (2,):
            raise ValueError("position must be 2D")
        if coverage_idx is not None:
            idx = int(coverage_idx)
            if idx < 0 or idx >= int(self.n_coverage):
                raise ValueError("coverage_idx outside the configured coverage set")
            expected = np.asarray(self.coverage_points[idx], dtype=float)
            if float(np.linalg.norm(p - expected)) > 1e-6:
                raise ValueError("coverage_idx does not match the measured position")
        src = self.source_by_channel.get(int(channel))
        result = "no_signal"
        svd = None
        if src is not None and self.channels[src.channel].status != "cleared":
            if self._visible(src, p):
                d = float(np.linalg.norm(p - src.position))
                if d <= NEAR_DIST + 1e-9:
                    result = "near"
                else:
                    result = "direction"
                    err = self.error_field.value(src.channel, float(p[0]), float(p[1]))
                    svd = normalize_deg(self._bearing(p, src.position) + err)
                    if self.bearing_decimals is not None:
                        svd = quantize_bearing_deg(svd, self.bearing_decimals)
        move = float(np.linalg.norm(p - self.pos))
        switch = 1.0 if int(channel) != self.current_channel else 0.0
        dt = move / 5.0 + switch + 5.0
        # 更新时钟与位置
        self.move_distance += move
        self.pos = p.copy()
        if int(channel) != self.current_channel:
            self.n_switch += 1
        self.current_channel = int(channel)
        self.virtual_time += dt
        self.step_count += 1
        self.n_measure += 1
        if is_refine:
            self.refine_actions += 1
        # 更新频道账本
        st = self.channels[int(channel)]
        obs = {"position": p.copy(), "result": result}
        if svd is not None:
            obs["svd_deg"] = float(svd)
        st.observations.append(obs)
        if coverage_idx is not None:
            st.scan_points.add(int(coverage_idx))
        if result == "direction":
            st.status = "discovered" if st.status == "unknown" else st.status
            st.has_direction = True
            envelope = OFFICIAL_BEARING_ENVELOPE_DEG if self.bearing_decimals is not None else 1.0
            newpoly = intersect_halfplanes(wedge_halfplanes(p, float(svd), eps_deg=envelope), initial=st.poly, add_domain=False)
            # 加上检测时有效接收半径上界的外切近似（真实源必在半径<=1500内）
            newpoly = intersect_halfplanes(circle_outer_halfplanes(p, MAX_RECEIVE_RADIUS, n=64),
                                           initial=newpoly, add_domain=False)
            st.update_poly(newpoly)
            if st.type_hint == "unknown":
                st.type_hint = "omni"
        elif result == "near":
            st.status = "discovered" if st.status == "unknown" else st.status
            st.has_near = True
            st.near_pos = p.copy()
            near_poly = circle_outer_polygon(p, NEAR_DIST, n=32)
            st.update_poly(near_poly)
            if st.type_hint == "unknown":
                st.type_hint = "omni"
        else:
            # no_signal
            if st.status == "discovered" and st.has_direction:
                st.no_signal_after_dir += 1
                st.type_hint = "directional"
            if st.status == "unknown":
                if len(st.scan_points) >= self.n_coverage:
                    st.status = "absent"
            elif st.status == "discovered":
                # 第4问 no_signal 可能来自背向，不能删除观测点附近圆盘。
                pass
        st.last_result = result
        if is_refine and (st.has_direction or st.has_near):
            st.local_extra_count = min(6, st.local_extra_count + 1)
        if self._completion_certificate():
            self.done = True
            self.success = True
        if self.virtual_time >= self.virtual_limit or self.step_count >= self.step_limit:
            self.done = True
        return {
            "accepted": True,
            "measure_result": result,
            "svd_deg": svd,
            "virtual_time_s": self.virtual_time,
            "delta_time_s": dt,
            "position": p.copy(),
            "channel": int(channel),
        }

    def clear(self, position: Sequence[float], channel: int) -> dict:
        p = np.asarray(position, dtype=float)
        src = self.source_by_channel.get(int(channel))
        success = False
        if src is not None and self.channels[src.channel].status != "cleared":
            if float(np.linalg.norm(p - src.position)) <= CLEAR_DIST + 1e-9:
                success = True
        move = float(np.linalg.norm(p - self.pos))
        dt = move / 5.0 + (5.0 if success else 3.0)
        self.move_distance += move
        self.pos = p.copy()
        self.virtual_time += dt
        self.step_count += 1
        self.n_clear += 1
        if success:
            self.channels[int(channel)].status = "cleared"
        else:
            self.channels[int(channel)].clear_fail_count += 1
            self.n_clear_fail += 1
        if self._completion_certificate():
            self.done = True
            self.success = True
        if self.virtual_time >= self.virtual_limit or self.step_count >= self.step_limit:
            self.done = True
        return {
            "accepted": True,
            "clear_result": "success" if success else "no_target_in_range",
            "virtual_time_s": self.virtual_time,
            "delta_time_s": dt,
            "position": p.copy(),
            "channel": int(channel),
        }

    # ---------------- 完成证书 ----------------
    def cleared_count(self) -> int:
        return sum(1 for st in self.channels.values() if st.status == "cleared")

    def absent_count(self) -> int:
        return sum(1 for st in self.channels.values() if st.status == "absent")

    def discovered_count(self) -> int:
        return sum(1 for st in self.channels.values() if st.status == "discovered")

    def unresolved_count(self) -> int:
        return sum(1 for st in self.channels.values() if st.status in ("unknown", "discovered"))

    def completion_certificate(self) -> bool:
        if self.cleared_count() >= MAX_SOURCES:
            return True
        return all(st.status in ("cleared", "absent") for st in self.channels.values())

    def _completion_certificate(self) -> bool:
        return self.completion_certificate()

    # ---------------- 观测/状态 ----------------
    def channel_features(self) -> np.ndarray:
        feats = []
        ncov = max(self.n_coverage, 1)
        for ch in CHANNELS:
            st = self.channels[ch]
            status = st.status
            # 4维 one-hot
            feats.extend([1.0 if status == "unknown" else 0.0,
                          1.0 if status == "discovered" else 0.0,
                          1.0 if status == "cleared" else 0.0,
                          1.0 if status == "absent" else 0.0])
            feats.append(1.0 if st.has_direction else 0.0)
            feats.append(1.0 if st.has_near else 0.0)
            feats.append(min(len(st.observations), 20) / 20.0)
            feats.append(min(st.local_extra_count, 6) / 6.0)
            feats.append(1.0 if st.fallback_started else 0.0)
            # 保守集合的半径和中心；未知/absent 时给0
            rad = st.poly_radius / DOMAIN_RADIUS if st.poly is not None and len(st.poly) else 0.0
            c = st.poly_center if st.poly is not None and len(st.poly) else np.zeros(2)
            feats.extend([rad, c[0] / DOMAIN_RADIUS, c[1] / DOMAIN_RADIUS])
            feats.append(len(st.scan_points) / ncov)
            feats.append(st.clear_fail_count / 5.0)
            feats.append(0.0 if st.type_hint == "unknown" else (0.5 if st.type_hint == "omni" else 1.0))
            feats.append(min(st.no_signal_after_dir, 6) / 6.0)
            feats.append(1.0 if st.last_result == "no_signal" else 0.0)
            feats.append(1.0 if st.last_result == "direction" else 0.0)
        return np.asarray(feats, dtype=np.float32)

    def global_features(self) -> np.ndarray:
        uv = self.virtual_time / 3600.0
        return np.asarray([
            self.pos[0] / DOMAIN_RADIUS,
            self.pos[1] / DOMAIN_RADIUS,
            self.current_channel / 20.0,
            uv,
            1.0 if self.mode == 4 else 0.0,
            self.cleared_count() / 16.0,
            self.absent_count() / 20.0,
            self.discovered_count() / 16.0,
            self.unresolved_count() / 20.0,
            self.n_coverage / 31.0,
            min(self.n_measure, 500) / 500.0,
            min(self.n_clear, 500) / 500.0,
            min(self.n_clear_fail, 200) / 200.0,
            min(self.move_distance, 50000) / 50000.0,
        ], dtype=np.float32)

    def state_vector(self) -> np.ndarray:
        return np.concatenate([self.global_features(), self.channel_features()]).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return int(len(self.state_vector()))


# --------------- 动作定义 ---------------
@dataclass
class MacroAction:
    kind: str                  # SCAN | REFINE | RESOLVE | EXIT
    point: Optional[np.ndarray] = None
    channel: Optional[int] = None
    scan_idx: Optional[int] = None
    fallback: bool = False


class MacroEnv:
    """96 槽位固定动作空间的宏动作包装器。"""
    N_SCAN = 31
    N_REFINE = 48
    N_RESOLVE = 16
    N_EXIT = 1
    N_ACTIONS = N_SCAN + N_REFINE + N_RESOLVE + N_EXIT  # 96

    def __init__(self, mode: int = 3, n_sources: int = 12, seed: Optional[int] = None,
                 step_limit: int = 20000, virtual_limit: float = 360000.0,
                 edge_bias: float = 0.15, cluster: bool = False):
        self.env = RadioEnv(mode=mode, n_sources=n_sources, seed=seed, step_limit=step_limit,
                            virtual_limit=virtual_limit, edge_bias=edge_bias, cluster=cluster)
        self.mode = mode
        self._actions: Dict[int, MacroAction] = {}
        self._last_state = None
        self._last_phi = 0.0
        self.early_fallback = False

    @property
    def state_dim(self) -> int:
        return int(self.env.state_dim)

    def reset(self, seed: Optional[int] = None, n_sources: Optional[int] = None,
              sources: Optional[List[Source]] = None) -> np.ndarray:
        self.env.reset(seed=seed, n_sources=n_sources, sources=sources)
        self._actions = {}
        self._last_phi = self._potential()
        self._build_actions()
        return self.state()

    def state(self) -> np.ndarray:
        return self.env.state_vector()

    def action_mask(self) -> np.ndarray:
        self._build_actions()
        mask = np.zeros(self.N_ACTIONS, dtype=bool)
        for k in self._actions:
            mask[k] = True
        return mask

    def action_mapping(self) -> Dict[int, MacroAction]:
        self._build_actions()
        return dict(self._actions)

    def _potential(self) -> float:
        # 仅由可观测账本构造的势函数；终止时为0。
        unknown = sum(1 for st in self.env.channels.values() if st.status == "unknown")
        discovered = sum(1 for st in self.env.channels.values() if st.status == "discovered")
        rad_sum = sum(min(st.poly_radius, DOMAIN_RADIUS) / DOMAIN_RADIUS
                      for st in self.env.channels.values() if st.status == "discovered")
        return -(5.0 * unknown + 20.0 * discovered + 5.0 * rad_sum)

    def _build_actions(self) -> None:
        acts: Dict[int, MacroAction] = {}
        if self.env.done:
            return
        # SCAN：覆盖点仍有未知频道未扫描时有效
        for idx in range(min(self.N_SCAN, self.env.n_coverage)):
            p = self.env.coverage_points[idx]
            need = any(st.status == "unknown" and idx not in st.scan_points
                       for st in self.env.channels.values())
            if need:
                acts[idx] = MacroAction("SCAN", point=p.copy(), scan_idx=idx)
        # 已发现未清除频道，按频道号排序分配 16 个 REFINE/RESOLVE 槽
        discovered = sorted([st.channel for st in self.env.channels.values()
                             if st.status == "discovered"])
        for j, ch in enumerate(discovered[:self.N_RESOLVE]):
            st = self.env.channels[ch]
            base_r = self.N_SCAN + j * 3
            # REFINE候选：局部测量额度耗尽后只能进入可靠清除或光学保底
            if st.local_extra_count < 6 and not st.fallback_started:
                from .resolver import refine_candidates
                cands = refine_candidates(st.observations, st.poly, self.env.pos, max_candidates=3)
                for t, p in enumerate(cands):
                    acts[base_r + t] = MacroAction("REFINE", point=p.copy(), channel=ch)
            # RESOLVE：满足 reliable clear，或已达局部尝试上限，或已有near
            ok_clear = st.poly_radius <= CLEAR_MARGIN
            center = st.poly_center
            rad = st.poly_radius
            fallback = False
            if ok_clear or st.has_near:
                target = st.near_pos.copy() if st.has_near and st.near_pos is not None else center
                acts[self.N_SCAN + self.N_REFINE + j] = MacroAction("RESOLVE", point=target, channel=ch)
            elif st.has_direction and st.observations:
                # 允许网络在首次失联后选择光学保底；保底始终可完成该源。
                fallback = True
                if st.observations:
                    first = None
                    for ob in st.observations:
                        if ob.get("result") == "direction":
                            first = ob
                            break
                    if first is None:
                        first = st.observations[0]
                    pts = optical_fallback_points(first["position"], float(first.get("svd_deg", 0.0)))
                    acts[self.N_SCAN + self.N_REFINE + j] = MacroAction(
                        "RESOLVE", point=pts[0], channel=ch, fallback=True)
        # EXIT
        if self.env.completion_certificate():
            acts[self.N_SCAN + self.N_REFINE + self.N_RESOLVE] = MacroAction("EXIT")
        self._actions = acts

    def step(self, slot: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._build_actions()
        if int(slot) not in self._actions:
            # 非法槽位：不执行，给出小惩罚并保持可继续；训练时应通过mask避免。
            return self.state(), -1.0, False, {"invalid_action": True}
        act = self._actions[int(slot)]
        old_phi = self._potential()
        old_time = self.env.virtual_time
        info = {"action": act}
        if act.kind == "SCAN":
            idx = int(act.scan_idx)
            p = act.point
            # 按当前频道优先、其余频道号的顺序扫描该点仍需发现的未知频道
            chs = [ch for ch in CHANNELS if self.env.channels[ch].status == "unknown"
                   and idx not in self.env.channels[ch].scan_points]
            chs.sort(key=lambda c: (c != self.env.current_channel, c))
            self.env.scan_actions += 1
            for ch in chs:
                # 每个未知频道都要测；若中途发现则停止该点的该频道义务（已发现转入局部定位）
                self.env.measure(p, ch, coverage_idx=idx)
                if self.env.done:
                    break
            info["scan_idx"] = idx
        elif act.kind == "REFINE":
            self.env.measure(act.point, int(act.channel), is_refine=True)
            info["channel"] = act.channel
        elif act.kind == "RESOLVE":
            ch = int(act.channel)
            st = self.env.channels[ch]
            if act.fallback:
                st.fallback_started = True
                self.env.fallback_actions += 1
                first = None
                for ob in st.observations:
                    if ob.get("result") == "direction":
                        first = ob
                        break
                if first is None:
                    first = st.observations[0]
                pts = optical_fallback_points(first["position"], float(first.get("svd_deg", 0.0)))
                # 完整光学保底宏动作：依次清除直到成功或点集耗尽。
                for p in pts:
                    if self.env.channels[ch].status == "cleared":
                        break
                    self.env.clear(p, ch)
                    if self.env.done:
                        break
                info["fallback_points"] = len(pts)
            else:
                self.env.clear(act.point, ch)
            info["channel"] = ch
        elif act.kind == "EXIT":
            if self.env.completion_certificate():
                self.env.done = True
                self.env.success = True
            info["exit"] = True
        delta_t = float(self.env.virtual_time - old_time)
        # 奖励：实际虚拟时间 + 势函数差；正常终止 phi=0。
        if self.env.done and self.env.success:
            new_phi = 0.0
        else:
            new_phi = self._potential()
        reward = -delta_t / 100.0 + (new_phi - old_phi)
        if self.env.done and not self.env.success:
            reward -= 100.0  # 未完成终端惩罚，远大于正常时间尺度
        self._last_phi = new_phi
        info.update({
            "delta_time": delta_t,
            "virtual_time": self.env.virtual_time,
            "cleared": self.env.cleared_count(),
            "absent": self.env.absent_count(),
            "discovered": self.env.discovered_count(),
        })
        return self.state(), float(reward), bool(self.env.done), info

    def completion_certificate(self) -> bool:
        return self.env.completion_certificate()

    def heuristic_action(self) -> int:
        """A0 风格的确定性保底动作，用于基线/行为克隆和异常接管。"""
        self._build_actions()
        if not self._actions:
            return self.N_SCAN + self.N_REFINE + self.N_RESOLVE
        # 1) 可清除的优先清除
        for slot, a in self._actions.items():
            if a.kind == "RESOLVE" and not a.fallback:
                return slot
        # 1b) 可选：一旦某已知源出现失联，直接进入光学保底（用于对照实验）
        if self.early_fallback:
            for slot, a in self._actions.items():
                if a.kind == "RESOLVE" and a.fallback and a.channel is not None:
                    st = self.env.channels[int(a.channel)]
                    if st.no_signal_after_dir > 0:
                        return slot
        # 2) 有未知频道：最近覆盖点（open path nearest-neighbor）
        scan_slots = [s for s, a in self._actions.items() if a.kind == "SCAN"]
        if scan_slots:
            def d(s):
                p = self._actions[s].point
                return float(np.linalg.norm(np.asarray(p) - self.env.pos))
            return min(scan_slots, key=d)
        # 3) 需局部细化：选最近候选
        ref_slots = [s for s, a in self._actions.items() if a.kind == "REFINE"]
        if ref_slots:
            def d2(s):
                p = self._actions[s].point
                return float(np.linalg.norm(np.asarray(p) - self.env.pos))
            return min(ref_slots, key=d2)
        # 4) 光学保底
        for slot, a in self._actions.items():
            if a.kind == "RESOLVE":
                return slot
        # 5) EXIT
        for slot, a in self._actions.items():
            if a.kind == "EXIT":
                return slot
        return next(iter(self._actions))
