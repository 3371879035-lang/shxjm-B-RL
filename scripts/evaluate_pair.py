"""本地配对评估：启发式 A0 与 PPO 方案B。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from brl.evaluate import evaluate_policy, paired_test, save_results, summarize
from brl.policy import MaskedActorCritic


def load_policy(path: str, state_dim: int, hidden: int = 128, device: str = "cpu"):
    p = MaskedActorCritic(state_dim, n_actions=96, hidden=hidden)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    p.load_state_dict(ckpt["state_dict"])
    p.eval()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True, choices=[3, 4])
    ap.add_argument("--n-sources", type=int, default=12)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--seeds", type=str, default="5000-5100")
    ap.add_argument("--eval-both", action="store_true")
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    baseline = evaluate_policy(None, args.mode, args.n_sources, seeds, heuristic=True)
    base_sum = summarize(baseline)
    print("[A0]", json.dumps(base_sum, ensure_ascii=False))
    base_path = args.save + "_baseline.jsonl" if args.save else f"results/eval_baseline_mode{args.mode}.jsonl"
    save_results(baseline, base_path)
    if args.model:
        state_dim = len(__import__("brl.local_env", fromlist=["MacroEnv"]).MacroEnv(
            mode=args.mode, n_sources=args.n_sources, seed=0).state())
        policy = load_policy(args.model, state_dim)
        ppo = evaluate_policy(policy, args.mode, args.n_sources, seeds)
        ppo_sum = summarize(ppo)
        print("[PPO]", json.dumps(ppo_sum, ensure_ascii=False))
        print("[paired]", json.dumps(paired_test(baseline, ppo), ensure_ascii=False))
        ppo_path = args.save + "_ppo.jsonl" if args.save else f"results/eval_ppo_mode{args.mode}.jsonl"
        save_results(ppo, ppo_path)


if __name__ == "__main__":
    main()
