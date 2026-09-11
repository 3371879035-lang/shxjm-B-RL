"""训练方案B的带动作屏蔽 PPO 调度器。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 允许从项目根目录运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from brl.evaluate import evaluate_policy, paired_test, save_results, summarize
from brl.local_env import MacroEnv
from brl.ppo import PPOConfig, PPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=3, choices=[3, 4])
    ap.add_argument("--n-sources", type=int, default=12, help="训练环境的目标源数；训练中会随机化10-16")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--bc-episodes", type=int, default=30)
    ap.add_argument("--bc-epochs", type=int, default=3)
    ap.add_argument("--bc-coef", type=float, default=0.05)
    ap.add_argument("--prior-logit", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--eval-seeds", type=str, default="3000-3020")
    args = ap.parse_args()

    if args.eval_seeds:
        a, b = args.eval_seeds.split("-")
        eval_seeds = list(range(int(a), int(b) + 1))
    else:
        eval_seeds = list(range(3000, 3020))

    env = MacroEnv(mode=args.mode, n_sources=args.n_sources, seed=args.seed)
    cfg = PPOConfig(rollout_steps=args.rollout, total_steps=args.steps,
                    lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                    seed=args.seed, hidden=args.hidden, bc_coef=args.bc_coef)
    trainer = PPOTrainer(env, cfg)
    trainer.policy.prior_logit.data.fill_(args.prior_logit)

    out_dir = ROOT / "results" / f"mode{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bc_episodes > 0:
        print(f"[BC] episodes={args.bc_episodes} epochs={args.bc_epochs}")
        trainer.behavior_clone(n_episodes=args.bc_episodes, epochs=args.bc_epochs)

    baseline = evaluate_policy(None, args.mode, args.n_sources, eval_seeds, heuristic=True)
    initial = evaluate_policy(trainer.policy, args.mode, args.n_sources, eval_seeds)
    print("[eval before]", json.dumps(summarize(initial), ensure_ascii=False))
    print("[eval baseline]", json.dumps(summarize(baseline), ensure_ascii=False))

    logs = trainer.train(args.steps)
    final = evaluate_policy(trainer.policy, args.mode, args.n_sources, eval_seeds)
    print("[eval after]", json.dumps(summarize(final), ensure_ascii=False))
    print("[paired after]", json.dumps(paired_test(baseline, final), ensure_ascii=False))

    save_path = Path(args.save) if args.save else (out_dir / f"ppo_mode{args.mode}_seed{args.seed}.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_path))
    with open(out_dir / f"train_log_mode{args.mode}_seed{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "history": logs,
                   "baseline_summary": summarize(baseline),
                   "final_summary": summarize(final),
                   "paired": paired_test(baseline, final)},
                  f, ensure_ascii=False, indent=2)
    save_results(final, out_dir / f"eval_final_mode{args.mode}_seed{args.seed}.jsonl")
    print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
