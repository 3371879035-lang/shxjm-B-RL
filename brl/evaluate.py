"""本地评估与配对实验工具。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .local_env import MacroEnv, Source, random_sources
from .policy import MaskedActorCritic


@dataclass
class EpisodeResult:
    mode: int
    seed: int
    n_sources: int
    policy: str
    success: bool
    cleared: int
    absent: int
    virtual_time: float
    avg_clear_time: float
    wall_time: float
    move_distance: float
    measurements: int
    clear_ops: int
    clear_fail: int
    switches: int
    fallback_actions: int
    macro_steps: int
    invalid_actions: int = 0


def run_episode(env: MacroEnv, policy: Optional[MaskedActorCritic] = None,
                heuristic: bool = False, deterministic: bool = True,
                device: str = "cpu", max_macro_steps: int = 1000) -> EpisodeResult:
    state = env.reset()
    wall0 = time.time()
    steps = 0
    invalid = 0
    while not env.env.done and steps < max_macro_steps:
        mask = env.action_mask()
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break
        if heuristic or policy is None:
            a = env.heuristic_action()
        else:
            s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            m_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            prior = env.heuristic_action()
            a, _, _ = policy.act(s_t, m_t, deterministic=deterministic,
                                 prior_action=torch.as_tensor(prior, device=device))
            if not mask[a]:
                a = int(legal[0])
                invalid += 1
        state, reward, done, info = env.step(int(a))
        steps += 1
    wall = time.time() - wall0
    cleared = env.env.cleared_count()
    vt = float(env.env.virtual_time)
    return EpisodeResult(
        mode=env.mode,
        seed=int(env.env.base_seed if env.env.base_seed is not None else -1),
        n_sources=len(env.env.sources),
        policy="heuristic" if (heuristic or policy is None) else "ppo",
        success=bool(env.env.success),
        cleared=int(cleared),
        absent=int(env.env.absent_count()),
        virtual_time=vt,
        avg_clear_time=(vt / cleared) if cleared > 0 else float("nan"),
        wall_time=wall,
        move_distance=float(env.env.move_distance),
        measurements=int(env.env.n_measure),
        clear_ops=int(env.env.n_clear),
        clear_fail=int(env.env.n_clear_fail),
        switches=int(env.env.n_switch),
        fallback_actions=int(env.env.fallback_actions),
        macro_steps=steps,
        invalid_actions=invalid,
    )


def evaluate_policy(policy: Optional[MaskedActorCritic], mode: int, n_sources: int,
                    seeds: Sequence[int], heuristic: bool = False,
                    device: str = "cpu", cluster: bool = False) -> List[EpisodeResult]:
    results = []
    for seed in seeds:
        env = MacroEnv(mode=mode, n_sources=n_sources, seed=int(seed), cluster=cluster)
        res = run_episode(env, policy=policy, heuristic=heuristic, device=device)
        results.append(res)
    return results


def summarize(results: Sequence[EpisodeResult]) -> dict:
    if not results:
        return {}
    success = np.array([r.success for r in results], dtype=float)
    cleared = np.array([r.cleared for r in results], dtype=float)
    total = np.array([r.n_sources for r in results], dtype=float)
    vt = np.array([r.virtual_time for r in results], dtype=float)
    avg = np.array([r.avg_clear_time for r in results], dtype=float)
    wall = np.array([r.wall_time for r in results], dtype=float)
    return {
        "episodes": len(results),
        "success_rate": float(success.mean()),
        "mean_cleared_ratio": float((cleared / np.maximum(total, 1)).mean()),
        "mean_virtual_time": float(vt.mean()),
        "median_virtual_time": float(np.median(vt)),
        "mean_avg_clear_time_success_only": float(np.nanmean(avg)),
        "mean_wall_time": float(wall.mean()),
        "mean_measurements": float(np.mean([r.measurements for r in results])),
        "mean_clear_ops": float(np.mean([r.clear_ops for r in results])),
        "mean_clear_fail": float(np.mean([r.clear_fail for r in results])),
        "mean_move_distance": float(np.mean([r.move_distance for r in results])),
        "mean_switches": float(np.mean([r.switches for r in results])),
        "fallback_rate": float(np.mean([1.0 if r.fallback_actions > 0 else 0.0 for r in results])),
    }


def save_results(results: Sequence[EpisodeResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def paired_test(results_a: Sequence[EpisodeResult], results_b: Sequence[EpisodeResult]) -> dict:
    """同场景配对比较：B 相对 A 的虚拟时间差（B-A，负值表示B更快）。"""
    n = min(len(results_a), len(results_b))
    a = np.array([results_a[i].virtual_time for i in range(n)], dtype=float)
    b = np.array([results_b[i].virtual_time for i in range(n)], dtype=float)
    diff = b - a
    return {
        "n_pairs": n,
        "mean_time_a": float(a.mean()),
        "mean_time_b": float(b.mean()),
        "mean_diff_b_minus_a": float(diff.mean()),
        "median_diff_b_minus_a": float(np.median(diff)),
        "stderr_diff": float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "wins_b": int(np.sum(diff < 0)),
        "ties": int(np.sum(np.abs(diff) < 1e-9)),
        "wins_a": int(np.sum(diff > 0)),
    }
