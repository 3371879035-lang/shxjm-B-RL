"""连续多局官方演练 runner：每局等待接口、自动 /enter、执行 PPO/混合策略、自动 /exit。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from brl.policy import MaskedActorCritic
from brl.remote import OfficialClient, RemoteMacroEnv


def default_robot_id() -> str:
    cfg = ROOT / "config" / "team.json"
    if cfg.exists():
        try:
            return str(json.loads(cfg.read_text(encoding="utf-8")).get("robot_id", ""))
        except Exception:
            return ""
    return ""


def load_policy(path: str, state_dim: int, hidden: int = 128):
    p = MaskedActorCritic(state_dim, n_actions=96, hidden=hidden)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    p.load_state_dict(ck["state_dict"])
    p.eval()
    return p


def choose_action(env: RemoteMacroEnv, policy, strategy: str):
    mask = env.action_mask()
    prior = env.heuristic_action()
    if policy is None or strategy == "heuristic":
        return prior, prior, None
    state = env.state()
    s_t = torch.as_tensor(state, dtype=torch.float32)
    m_t = torch.as_tensor(mask, dtype=torch.bool)
    a, _, _ = policy.act(s_t, m_t, deterministic=True,
                         prior_action=torch.as_tensor(prior, dtype=torch.long))
    policy_action = int(a)
    chosen = int(a)
    if strategy == "hybrid" and chosen != prior:
        logits, _ = policy(s_t.unsqueeze(0), m_t.unsqueeze(0),
                           torch.as_tensor(prior, dtype=torch.long).unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)[0]
        if float(probs[chosen]) < 2.0 * float(probs[prior]):
            chosen = prior
    return int(chosen), int(prior), policy_action


def run_one(index: int, mode: int, model_path: str, strategy: str, base_url: str,
            robot_id: str, out_dir: Path, wait_interface_s: float,
            max_macro_steps: int = 1000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    req_log = out_dir / f"run_{index:02d}.requests.jsonl"
    dec_log = out_dir / f"run_{index:02d}.decisions.jsonl"
    dec_log.unlink(missing_ok=True)
    client = OfficialClient(base_url, robot_id, log_path=str(req_log))
    env = RemoteMacroEnv(client, mode=mode)

    # 等待接口开放；每局结束后模拟器会关闭接口，下一局需要用户重新启动演练。
    t0 = time.time()
    while True:
        try:
            env.reset()
            break
        except Exception as e:
            if time.time() - t0 > wait_interface_s:
                return {"index": index, "mode": mode, "entered": False,
                        "error": f"interface wait timeout: {e}"}
            time.sleep(1.0)

    real_start = time.time()
    policy = load_policy(model_path, env.state_dim) if model_path else None
    steps = 0
    deviations = 0
    gate_fallbacks = 0
    while not env.env.done and steps < max_macro_steps:
        chosen, prior, policy_action = choose_action(env, policy, strategy)
        if policy_action is not None and policy_action != prior:
            deviations += 1
        if chosen != policy_action and policy_action is not None:
            gate_fallbacks += 1
        with open(dec_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": steps + 1,
                "virtual_time_s": float(env.env.virtual_time),
                "cleared": int(env.env.cleared_count()),
                "absent": int(env.env.absent_count()),
                "discovered": int(env.env.discovered_count()),
                "prior_slot": int(prior),
                "policy_slot": int(policy_action) if policy_action is not None else None,
                "chosen_slot": int(chosen),
                "policy_equals_prior": bool(policy_action == prior) if policy_action is not None else None,
                "gate_fallback": bool(chosen != policy_action) if policy_action is not None else False,
            }, ensure_ascii=False) + "\n")
        env.step(chosen)
        steps += 1

    if env.env.success and not getattr(env, "exited", False):
        try:
            client.exit()
            env.exited = True
        except Exception:
            pass
    real_duration = time.time() - real_start
    result = {
        "index": index, "mode": mode, "entered": True,
        "success": bool(env.env.success),
        "cleared": int(env.env.cleared_count()),
        "absent": int(env.env.absent_count()),
        "virtual_time_s": float(env.env.virtual_time),
        "avg_clear_time_s": float(env.env.virtual_time / max(env.env.cleared_count(), 1)),
        "measure_calls": int(env.env.n_measure),
        "clear_calls": int(env.env.n_clear),
        "clear_failures": int(env.env.n_clear_fail),
        "switches": int(env.env.n_switch),
        "fallback_actions": int(env.env.fallback_actions),
        "macro_steps": int(steps),
        "real_duration_s_local": float(real_duration),
        "policy_deviations_from_prior": int(deviations),
        "hybrid_gate_fallbacks": int(gate_fallbacks),
        "request_log": str(req_log),
    }
    (out_dir / f"run_{index:02d}.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True, choices=[3, 4])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--strategy", type=str, default="ppo", choices=["ppo", "hybrid", "heuristic"])
    ap.add_argument("--base-url", type=str, default="http://127.0.0.1:2026")
    ap.add_argument("--robot-id", type=str, default=default_robot_id())
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--wait-interface-s", type=float, default=1200.0)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "results" / f"batch_mode{args.mode}_{stamp}")
    results = []
    print(f"[batch] mode={args.mode} runs={args.runs} strategy={args.strategy} out={out_dir}", flush=True)
    for i in range(1, args.runs + 1):
        print(f"[batch] waiting for run {i}/{args.runs} interface...", flush=True)
        res = run_one(i, args.mode, args.model, args.strategy, args.base_url,
                      args.robot_id, out_dir, args.wait_interface_s)
        results.append(res)
        if res.get("entered"):
            print(f"[batch] run {i} done: success={res['success']} cleared={res['cleared']} "
                  f"V={res['virtual_time_s']:.1f}s steps={res['macro_steps']} "
                  f"deviations={res['policy_deviations_from_prior']}", flush=True)
        else:
            print(f"[batch] run {i} not entered: {res.get('error')}", flush=True)
    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch] all done, summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
