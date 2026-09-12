"""官方模拟器 HTTP+JSON 适配器与远程宏动作控制器。

与 local_env 共享几何、覆盖点、动作槽位和带动作屏蔽的 PPO 推理，
但在线状态只来自官方反馈，不使用真实源位置、半径或方向。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .coverage import s3_points, s4_points
from .geometry import (DOMAIN_RADIUS, MAX_RECEIVE_RADIUS, circle_outer_halfplanes,
                       circle_outer_polygon, domain_polygon, intersect_halfplanes)
from .local_env import (CHANNELS, CLEAR_MARGIN, MAX_SOURCES, ChannelState, MacroAction,
                        MacroEnv, NEAR_DIST)
from .resolver import reliable_clear
from .protocol import (ActionIOError, DeadlineExceeded,
                       OFFICIAL_BEARING_ENVELOPE_DEG)


class OfficialClient:
    """串行、幂等的官方接口客户端。"""
    def __init__(self, base_url: str, robot_id: str, timeout: float = 8.0,
                 log_path: Optional[str] = None, max_network_retries: int = 5):
        self.base_url = base_url.rstrip("/")
        self.robot_id = str(robot_id)
        self.timeout = float(timeout)
        self.max_network_retries = int(max_network_retries)
        self.log_path = log_path
        self.counter = 0
        self.last_virtual_time = 0.0
        self.remaining_real_duration_s: Optional[float] = None
        self.action_deadline_monotonic: Optional[float] = None
        self.requests: List[dict] = []
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def _new_request_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter}-{int(time.time()*1000)}"

    def _log(self, entry: dict) -> None:
        self.requests.append(entry)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def post(self, path: str, payload: dict) -> dict:
        """发送并重试完全相同的请求；网络重试复用原 request_id。"""
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_err = None
        for attempt in range(self.max_network_retries):
            if (path not in {"/enter", "/exit"}
                    and self.action_deadline_monotonic is not None
                    and time.monotonic() >= self.action_deadline_monotonic):
                self._log({"t": time.time(), "path": path, "payload": payload,
                           "attempt": attempt + 1, "error_type": "DeadlineExceeded",
                           "error": "action deadline reached"})
                raise DeadlineExceeded("official action deadline reached; reserve time for exit")
            request_timeout = self.timeout
            if path not in {"/enter", "/exit"} and self.action_deadline_monotonic is not None:
                request_timeout = min(request_timeout, max(
                    0.05, self.action_deadline_monotonic - time.monotonic()))
            req = Request(self.base_url + path, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=request_timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    try:
                        out = json.loads(raw)
                    except (json.JSONDecodeError, TypeError) as exc:
                        self._log({"t": time.time(), "path": path, "payload": payload,
                                   "attempt": attempt + 1, "error_type": type(exc).__name__,
                                   "error": "invalid JSON response", "raw": raw[:1000]})
                        raise ActionIOError("official response is not valid JSON") from exc
                    self._log({"t": time.time(), "path": path, "payload": payload, "response": out})
                    if out.get("accepted") is True and "virtual_time_s" in out:
                        try:
                            vt = float(out["virtual_time_s"])
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise ActionIOError("official response has invalid virtual_time_s") from exc
                        if not np.isfinite(vt) or vt + 1e-9 < self.last_virtual_time:
                            raise ActionIOError("official virtual_time_s must be finite and monotonic")
                        self.last_virtual_time = vt
                    return out
            except HTTPError as e:
                # HTTP 业务错误体可能仍是 JSON；读取后返回，避免错误重试改变状态。
                try:
                    raw = e.read().decode("utf-8")
                    out = json.loads(raw)
                except Exception:
                    out = {"accepted": False, "http_error": int(e.code), "raw": str(e)}
                self._log({"t": time.time(), "path": path, "payload": payload,
                           "http_error": int(e.code), "response": out})
                return out
            except (URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e
                self._log({"t": time.time(), "path": path, "payload": payload,
                           "attempt": attempt + 1, "error_type": type(e).__name__,
                           "error": str(e)})
                if attempt + 1 < self.max_network_retries:
                    delay = 0.2 * (attempt + 1)
                    if path not in {"/enter", "/exit"} and self.action_deadline_monotonic is not None:
                        delay = min(delay, max(0.0, self.action_deadline_monotonic - time.monotonic()))
                    if delay > 0.0:
                        time.sleep(delay)
                continue
        raise ActionIOError(f"network failed after retries: {last_err}")

    def base(self, prefix: str) -> dict:
        return {"arena_id": "default", "robot_id": self.robot_id,
                "request_id": self._new_request_id(prefix)}

    def enter(self) -> dict:
        out = self.post("/enter", self.base("enter"))
        if out.get("accepted") is True:
            if "remaining_real_duration_s" not in out:
                raise ActionIOError("official /enter response missing remaining_real_duration_s")
            try:
                remaining = float(out["remaining_real_duration_s"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ActionIOError("official /enter response has invalid remaining duration") from exc
            if not np.isfinite(remaining) or remaining <= 0.0:
                raise ActionIOError("official remaining duration must be finite and positive")
            self.remaining_real_duration_s = remaining
            self.last_virtual_time = 0.0
        return out

    def set_action_deadline(self, reserve_exit_s: float = 30.0) -> None:
        if self.remaining_real_duration_s is None:
            raise ActionIOError("cannot set deadline before accepted /enter")
        usable = float(self.remaining_real_duration_s) - float(reserve_exit_s)
        if usable <= 0.0:
            raise DeadlineExceeded("official remaining duration is below the exit reserve")
        self.action_deadline_monotonic = time.monotonic() + usable

    def measure(self, position: Sequence[float], channel: int) -> dict:
        p = self.base("measure")
        p["position"] = {"x": float(position[0]), "y": float(position[1])}
        p["channel"] = int(channel)
        return self.post("/measure", p)

    def clear(self, position: Sequence[float], channel: int) -> dict:
        p = self.base("clear")
        p["position"] = {"x": float(position[0]), "y": float(position[1])}
        p["channel"] = int(channel)
        return self.post("/clear", p)

    def exit(self) -> dict:
        return self.post("/exit", self.base("exit"))


class RemoteBelief:
    """仅由官方反馈维护的频道账本和保守可行域。"""
    def __init__(self, client: OfficialClient, mode: int = 3):
        self.client = client
        self.mode = int(mode)
        assert self.mode in (3, 4)
        self.coverage_points = s3_points() if self.mode == 3 else s4_points()
        self.n_coverage = len(self.coverage_points)
        self.pos = np.zeros(2, dtype=float)
        self.current_channel = 1
        self.virtual_time = 0.0
        self.channels: Dict[int, ChannelState] = {ch: ChannelState(ch) for ch in CHANNELS}
        for st in self.channels.values():
            st._refresh_cache()
        self.done = False
        self.success = False
        self.n_measure = 0
        self.n_clear = 0
        self.n_clear_fail = 0
        self.n_switch = 0
        self.move_distance = 0.0
        self.scan_actions = 0
        self.refine_actions = 0
        self.fallback_actions = 0
        self.invalid_responses = 0

    def _dt(self, out: dict) -> float:
        if "virtual_time_s" not in out:
            raise ActionIOError("official response missing virtual_time_s")
        try:
            vt = float(out["virtual_time_s"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ActionIOError("official response has invalid virtual_time_s") from exc
        if not np.isfinite(vt) or vt + 1e-9 < self.virtual_time:
            raise ActionIOError("official virtual_time_s must be finite and monotonic")
        dt = vt - self.virtual_time
        self.virtual_time = vt
        return dt

    def _validate_coverage(self, position: np.ndarray, coverage_idx: Optional[int]) -> None:
        if coverage_idx is None:
            return
        idx = int(coverage_idx)
        if idx < 0 or idx >= int(self.n_coverage):
            raise ActionIOError("coverage_idx outside the configured coverage set")
        expected = np.asarray(self.coverage_points[idx], dtype=float)
        if float(np.linalg.norm(position - expected)) > 1e-6:
            raise ActionIOError("coverage_idx does not match the measured position")

    def _move_to(self, pos: Sequence[float]) -> float:
        p = np.asarray(pos, dtype=float)
        d = float(np.linalg.norm(p - self.pos))
        self.move_distance += d
        self.pos = p.copy()
        return d

    def _update_after_direction(self, st: ChannelState, pos: np.ndarray, svd: float) -> None:
        st.status = "discovered" if st.status == "unknown" else st.status
        st.has_direction = True
        newpoly = intersect_halfplanes(
            __import__("brl.geometry", fromlist=["wedge_halfplanes"]).wedge_halfplanes(
                pos, svd, eps_deg=OFFICIAL_BEARING_ENVELOPE_DEG),
            initial=st.poly, add_domain=False)
        newpoly = intersect_halfplanes(
            circle_outer_halfplanes(pos, MAX_RECEIVE_RADIUS, n=64),
            initial=newpoly, add_domain=False)
        st.update_poly(newpoly)

    def measure(self, position: Sequence[float], channel: int,
                coverage_idx: Optional[int] = None, is_refine: bool = False) -> dict:
        p = np.asarray(position, dtype=float)
        self._validate_coverage(p, coverage_idx)
        try:
            out = self.client.measure(p, int(channel))
        except ActionIOError:
            raise
        except Exception as exc:
            raise ActionIOError("official /measure transport failed") from exc
        if out.get("accepted") is not True:
            self.invalid_responses += 1
            raise ActionIOError(f"/measure not accepted: {out}")
        result = str(out.get("measure_result", ""))
        if result not in ("direction", "near", "no_signal"):
            self.invalid_responses += 1
            raise ActionIOError(f"/measure invalid result: {out}")
        if result == "direction":
            try:
                value = float(out["svd_deg"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self.invalid_responses += 1
                raise ActionIOError(f"/measure missing direction: {out}") from exc
            if not np.isfinite(value) or not (0.0 <= value < 360.0):
                self.invalid_responses += 1
                raise ActionIOError(f"/measure non-finite direction: {out}")
        # Validate the clock before mutating position, channel, observations or
        # coverage evidence.
        dt = self._dt(out)
        # 成功 measure 后当前位置与测向机频道更新
        self._move_to(p)
        if int(channel) != self.current_channel:
            self.n_switch += 1
        self.current_channel = int(channel)
        self.n_measure += 1
        if is_refine:
            self.refine_actions += 1
        st = self.channels[int(channel)]
        result = str(out["measure_result"])
        obs = {"position": p.copy(), "result": result}
        svd = out.get("svd_deg")
        if svd is not None:
            obs["svd_deg"] = float(svd)
        st.observations.append(obs)
        if coverage_idx is not None:
            st.scan_points.add(int(coverage_idx))
        if result == "direction" and svd is not None:
            self._update_after_direction(st, p, float(svd))
            if st.type_hint == "unknown":
                st.type_hint = "omni"
        elif result == "near":
            st.status = "discovered" if st.status == "unknown" else st.status
            st.has_near = True
            st.near_pos = p.copy()
            st.update_poly(circle_outer_polygon(p, NEAR_DIST, n=32))
            if st.type_hint == "unknown":
                st.type_hint = "omni"
        else:
            if st.status == "unknown":
                if len(st.scan_points) >= self.n_coverage:
                    st.status = "absent"
            elif st.status == "discovered" and st.has_direction:
                st.no_signal_after_dir += 1
                st.type_hint = "directional"
        st.last_result = result
        if is_refine and (st.has_direction or st.has_near):
            st.local_extra_count = min(6, st.local_extra_count + 1)
        if self.completion_certificate():
            self.done = True
            self.success = True
        return {"accepted": True, "measure_result": result, "svd_deg": svd,
                "virtual_time_s": self.virtual_time, "delta_time_s": dt}

    def clear(self, position: Sequence[float], channel: int) -> dict:
        p = np.asarray(position, dtype=float)
        st = self.channels[int(channel)]
        try:
            out = self.client.clear(p, int(channel))
        except ActionIOError:
            raise
        except Exception as exc:
            raise ActionIOError("official /clear transport failed") from exc
        if out.get("accepted") is not True:
            self.invalid_responses += 1
            raise ActionIOError(f"/clear not accepted: {out}")
        if out.get("clear_result") not in ("success", "no_target_in_range"):
            self.invalid_responses += 1
            raise ActionIOError(f"/clear invalid result: {out}")
        dt = self._dt(out)
        self._move_to(p)
        self.n_clear += 1
        if out.get("clear_result") == "success":
            st.status = "cleared"
        else:
            st.clear_fail_count += 1
            self.n_clear_fail += 1
        if self.completion_certificate():
            self.done = True
            self.success = True
        return {"accepted": True, "clear_result": out.get("clear_result"),
                "virtual_time_s": self.virtual_time, "delta_time_s": dt}

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

    # ---- 状态向量：与 local_env.RadioEnv 保持一致 ----
    def channel_features(self) -> np.ndarray:
        feats = []
        ncov = max(self.n_coverage, 1)
        for ch in CHANNELS:
            st = self.channels[ch]
            status = st.status
            feats.extend([1.0 if status == "unknown" else 0.0,
                          1.0 if status == "discovered" else 0.0,
                          1.0 if status == "cleared" else 0.0,
                          1.0 if status == "absent" else 0.0])
            feats.append(1.0 if st.has_direction else 0.0)
            feats.append(1.0 if st.has_near else 0.0)
            feats.append(min(len(st.observations), 20) / 20.0)
            feats.append(min(st.local_extra_count, 6) / 6.0)
            feats.append(1.0 if st.fallback_started else 0.0)
            feats.append(st.poly_radius / DOMAIN_RADIUS)
            c = st.poly_center
            feats.extend([c[0] / DOMAIN_RADIUS, c[1] / DOMAIN_RADIUS])
            feats.append(len(st.scan_points) / ncov)
            feats.append(st.clear_fail_count / 5.0)
            feats.append(0.0 if st.type_hint == "unknown" else (0.5 if st.type_hint == "omni" else 1.0))
            feats.append(min(st.no_signal_after_dir, 6) / 6.0)
            feats.append(1.0 if st.last_result == "no_signal" else 0.0)
            feats.append(1.0 if st.last_result == "direction" else 0.0)
        return np.asarray(feats, dtype=np.float32)

    def global_features(self) -> np.ndarray:
        return np.asarray([
            self.pos[0] / DOMAIN_RADIUS,
            self.pos[1] / DOMAIN_RADIUS,
            self.current_channel / 20.0,
            self.virtual_time / 3600.0,
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


class RemoteMacroEnv(MacroEnv):
    """把 MacroEnv 的 96 槽位动作语义映射到官方 HTTP 接口。"""
    def __init__(self, client: OfficialClient, mode: int = 3):
        self.client = client
        self.mode = int(mode)
        self.env = RemoteBelief(client, mode)
        self._actions: Dict[int, MacroAction] = {}
        self._last_phi = 0.0
        self.early_fallback = False
        self.exited = False

    def reset(self) -> np.ndarray:
        enter = self.client.enter()
        if enter.get("accepted") is not True:
            raise RuntimeError(f"/enter not accepted: {enter}")
        self.env = RemoteBelief(self.client, self.mode)
        self.env.done = False
        self.env.success = False
        self.exited = False
        self._actions = {}
        self._last_phi = self._potential()
        self._build_actions()
        return self.state()

    @property
    def state_dim(self) -> int:
        return int(self.env.state_dim)

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

    def step(self, slot: int) -> Tuple[np.ndarray, float, bool, dict]:
        self._build_actions()
        if int(slot) not in self._actions:
            return self.state(), -1.0, bool(self.env.done), {"invalid_action": True, "slot": int(slot)}
        act = self._actions[int(slot)]
        old_phi = self._potential()
        old_v = self.env.virtual_time
        info = {"action": act}
        if act.kind == "SCAN":
            idx = int(act.scan_idx)
            p = act.point
            chs = [ch for ch in CHANNELS if self.env.channels[ch].status == "unknown"
                   and idx not in self.env.channels[ch].scan_points]
            chs.sort(key=lambda c: (c != self.env.current_channel, c))
            self.env.scan_actions += 1
            for ch in chs:
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
                pts = __import__("brl.resolver", fromlist=["optical_fallback_points"]).optical_fallback_points(
                    first["position"], float(first.get("svd_deg", 0.0)))
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
            self.client.exit()
            self.exited = True
            self.env.done = True
            self.env.success = True
            info["exit"] = True
        dt = float(self.env.virtual_time - old_v)
        new_phi = 0.0 if (self.env.done and self.env.success) else self._potential()
        reward = -dt / 100.0 + (new_phi - old_phi)
        if self.env.done and not self.env.success:
            reward -= 100.0
        self._last_phi = new_phi
        info.update({"delta_time": dt, "virtual_time": self.env.virtual_time,
                     "cleared": self.env.cleared_count(),
                     "absent": self.env.absent_count(),
                     "discovered": self.env.discovered_count()})
        return self.state(), float(reward), bool(self.env.done), info

    def completion_certificate(self) -> bool:
        return self.env.completion_certificate()
