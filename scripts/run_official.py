"""官方模拟器运行入口（问题3/4）。使用方案B：PPO调度 + 几何/覆盖保底。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def default_robot_id() -> str:
    cfg = ROOT / "config" / "team.json"
    if cfg.exists():
        try:
            return str(json.loads(cfg.read_text(encoding="utf-8")).get("robot_id", ""))
        except Exception:
            return ""
    return ""

import numpy as np
import torch

from brl.policy import MaskedActorCritic
from brl.remote import OfficialClient, RemoteMacroEnv


def load_policy(path: str, state_dim: int, hidden: int = 128, device: str = "cpu"):
    policy = MaskedActorCritic(state_dim, n_actions=96, hidden=hidden)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True, choices=[3, 4])
    ap.add_argument("--robot-id", type=str, default=default_robot_id())
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:2026")
    ap.add_argument("--model", type=str, default="", help="PPO checkpoint; 留空则用A0保底策略")
    ap.add_argument("--strategy", type=str, default="hybrid", choices=["ppo", "hybrid", "heuristic"])
    ap.add_argument("--log", type=str, default="")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--wait-interface-s", type=float, default=300.0)
    args = ap.parse_args()

    log_path = args.log or str(ROOT / "results" / f"official_mode{args.mode}_{int(time.time())}.jsonl")
    client = OfficialClient(args.base_url, args.robot_id, log_path=log_path)

    # 等待模拟器接口开放；连接失败时重试 /enter（模拟器未开放时通常连接被拒绝）。
    env = RemoteMacroEnv(client, mode=args.mode)
    wait_start = time.time()
    while True:
        try:
            state = env.reset()
            t0 = time.time()
            break
        except Exception as e:
            if time.time() - wait_start > args.wait_interface_s:
                raise
            print(f"[wait] interface not ready: {e}")
            time.sleep(1.0)

    policy = None
    if args.model:
        policy = load_policy(args.model, env.state_dim, hidden=args.hidden)

    print(f"[enter] remaining_real_duration_s={client.remaining_real_duration_s}")
    decision_path = str(Path(log_path).with_suffix(".decisions.jsonl"))
    steps = 0
    while not env.env.done:
        # 现实时间保护：如果剩余时间已经很低，使用启发式保底动作完成剩余义务。
        if client.remaining_real_duration_s is not None:
            remain = client.remaining_real_duration_s - (time.time() - t0)
            # 该估计不含新请求的网络与模拟器内部时间，仅作保守提醒。
            if remain < 3.0:
                print("[warn] real time nearly exhausted")
                break
        mask = env.action_mask()
        prior = env.heuristic_action()
        policy_action = None
        if policy is not None and args.strategy in ("ppo", "hybrid"):
            s_t = torch.as_tensor(env.state(), dtype=torch.float32)
            m_t = torch.as_tensor(mask, dtype=torch.bool)
            a, logp, value = policy.act(s_t, m_t, deterministic=True,
                                        prior_action=torch.as_tensor(prior, dtype=torch.long))
            policy_action = int(a)
            if args.strategy == "hybrid":
                # 简单安全门：若策略选的不是先验且概率优势不明显，则保守使用A0。
                if a != prior:
                    logits, _ = policy(torch.as_tensor(env.state(), dtype=torch.float32).unsqueeze(0),
                                       m_t.unsqueeze(0),
                                       torch.as_tensor(prior, dtype=torch.long).unsqueeze(0))
                    p = torch.softmax(logits, dim=-1)[0]
                    if float(p[a]) < 2.0 * float(p[prior]):
                        a = prior
        else:
            a = prior
        chosen = int(a)
        with open(decision_path, "a", encoding="utf-8") as df:
            df.write(json.dumps({
                "step": steps + 1,
                "virtual_time_s": float(env.env.virtual_time),
                "cleared": int(env.env.cleared_count()),
                "absent": int(env.env.absent_count()),
                "discovered": int(env.env.discovered_count()),
                "unresolved": int(env.env.unresolved_count()),
                "prior_slot": int(prior),
                "policy_slot": policy_action,
                "chosen_slot": chosen,
                "chosen_equals_prior": bool(chosen == int(prior)),
                "mask_legal": bool(mask[chosen]) if chosen < len(mask) else False,
                "strategy": args.strategy,
            }, ensure_ascii=False) + "\n")
        state, reward, done, info = env.step(chosen)
        steps += 1
        if steps % 20 == 0:
            act = info.get("action")
            print(f"[step {steps}] V={env.env.virtual_time:.1f} cleared={env.env.cleared_count()} "
                  f"absent={env.env.absent_count()} discovered={env.env.discovered_count()} "
                  f"action={getattr(act, 'kind', None)}")

    if env.env.done and env.env.success:
        # 最后一发 clear 可能直接触发完成证书，此时循环在 EXIT 宏动作之前结束。
        # 因此这里必须补一次 /exit，确保主动结束并生成行为日志。
        if not getattr(env, "exited", False):
            try:
                exit_resp = client.exit()
                env.exited = True
                print(f"[exit] {exit_resp}")
            except Exception as e:
                print(f"[exit warning] {e}")
        print(f"[done] virtual_time={env.env.virtual_time:.3f}s cleared={env.env.cleared_count()} steps={steps}")
    else:
        print(f"[incomplete] virtual_time={env.env.virtual_time:.3f}s cleared={env.env.cleared_count()} steps={steps}")
    # 正常完成时 /exit 已由 EXIT 宏动作发出；其余情况不要再调用 /exit。
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
