"""用于固定 96 槽位动作空间的 PPO 训练器。"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .local_env import MacroEnv
from .policy import MaskedActorCritic


@dataclass
class PPOConfig:
    rollout_steps: int = 4096
    total_steps: int = 50000
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    epochs: int = 5
    batch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    bc_coef: float = 0.05
    max_grad_norm: float = 0.5
    seed: int = 0
    hidden: int = 128
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logps: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.next_states: List[np.ndarray] = []
        self.priors: List[int] = []

    def add(self, s, m, a, logp, v, r, done, ns, prior):
        self.states.append(s)
        self.masks.append(m)
        self.actions.append(int(a))
        self.logps.append(float(logp))
        self.values.append(float(v))
        self.rewards.append(float(r))
        self.dones.append(bool(done))
        self.next_states.append(ns)
        self.priors.append(int(prior))

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__init__()


class PPOTrainer:
    def __init__(self, env: MacroEnv, config: Optional[PPOConfig] = None):
        self.env = env
        self.cfg = config or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        self.state_dim = env.state_dim
        self.policy = MaskedActorCritic(self.state_dim, env.N_ACTIONS, hidden=self.cfg.hidden).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.train_rng = np.random.default_rng(int(self.cfg.seed) + 12345)
        self.episode_count = 0
        self._train_state = None
        self._ep_start_v = 0.0
        self.global_step = 0
        self.update_count = 0
        self.history: List[dict] = []

    def _to_tensor(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _reset_train_env(self) -> np.ndarray:
        seed = int(self.train_rng.integers(0, 2**31 - 1))
        # 训练中随机化源数与场景，避免记住单个案例。
        n = int(self.train_rng.integers(10, 17)) if self.env.mode == 3 else int(self.train_rng.integers(10, 17))
        return self.env.reset(seed=seed, n_sources=n)

    def collect_rollout(self) -> Dict[str, float]:
        self.policy.eval()
        if self._train_state is None:
            self._train_state = self._reset_train_env()
            self._ep_start_v = self.env.env.virtual_time
        state = self._train_state
        episode_returns: List[float] = []
        episode_virtual: List[float] = []
        ep_reward = 0.0
        ep_start_v = self._ep_start_v
        episodes_done = 0
        while len(self.buffer) < self.cfg.rollout_steps:
            mask = self.env.action_mask()
            # 应该至少有一个合法动作；否则说明环境设计问题
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                raise RuntimeError("no legal action")
            s_t = self._to_tensor(state)
            m_t = self._to_tensor(mask, dtype=torch.bool)
            prior = self.env.heuristic_action()
            a, logp, v = self.policy.act(s_t, m_t, deterministic=False,
                                         prior_action=torch.as_tensor(prior, device=self.device))
            next_state, reward, done, info = self.env.step(a)
            next_mask = self.env.action_mask() if not done else np.zeros(self.env.N_ACTIONS, dtype=bool)
            self.buffer.add(state, mask, a, logp, v, reward, done, next_state, prior)
            state = next_state
            ep_reward += reward
            self.global_step += 1
            if done:
                episode_returns.append(ep_reward)
                episode_virtual.append(float(self.env.env.virtual_time - ep_start_v))
                ep_reward = 0.0
                episodes_done += 1
                self.episode_count += 1
                state = self._reset_train_env()
                self._ep_start_v = self.env.env.virtual_time
                ep_start_v = self._ep_start_v
        self.policy.train()
        self._train_state = state
        if len(self.buffer) == 0:
            return {}
        # GAE
        states = self._to_tensor(np.asarray(self.buffer.states))
        masks = self._to_tensor(np.asarray(self.buffer.masks), dtype=torch.bool)
        actions = self._to_tensor(np.asarray(self.buffer.actions), dtype=torch.long)
        priors = self._to_tensor(np.asarray(self.buffer.priors), dtype=torch.long)
        old_logps = self._to_tensor(np.asarray(self.buffer.logps))
        values = self._to_tensor(np.asarray(self.buffer.values))
        rewards = self._to_tensor(np.asarray(self.buffer.rewards))
        dones = self._to_tensor(np.asarray(self.buffer.dones), dtype=torch.float32)
        next_states = self._to_tensor(np.asarray(self.buffer.next_states))
        with torch.no_grad():
            next_masks_np = np.asarray(self.buffer.masks)  # 不使用next_mask列，用当前近似
            _, next_values = self.policy(next_states, None)  # 值函数不使用动作先验
            next_values = next_values
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_v = next_values[t]
                    nonterminal = 1.0 - dones[t]
                else:
                    next_v = values[t + 1]
                    nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * nonterminal * next_v - values[t]
                last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        # PPO update
        n = len(self.buffer)
        inds = np.arange(n)
        clip_fracs = []
        bc_losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        for epoch in range(self.cfg.epochs):
            np.random.shuffle(inds)
            for start in range(0, n, self.cfg.batch_size):
                idx = inds[start:start + self.cfg.batch_size]
                mb_states = states[idx]
                mb_masks = masks[idx]
                mb_actions = actions[idx]
                mb_old_logps = old_logps[idx]
                mb_adv = advantages[idx]
                mb_returns = returns[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                logp, entropy, value = self.policy.evaluate(mb_states, mb_masks, mb_actions, priors[idx])
                # 与 A0 保底动作的行为克隆正则，防止 PPO 在稀疏奖励下退化。
                logits_bc, _ = self.policy(mb_states, mb_masks, priors[idx])
                bc_loss = nn.functional.cross_entropy(logits_bc, priors[idx])
                ratio = torch.exp(logp - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, mb_returns)
                entropy_loss = -entropy.mean()
                loss = (policy_loss + self.cfg.value_coef * value_loss
                        + self.cfg.entropy_coef * entropy_loss
                        + self.cfg.bc_coef * bc_loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                clip_fracs.append(float(((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                bc_losses.append(float(bc_loss.item()))
                entropies.append(float(entropy.mean().item()))
        stats = {
            "global_step": self.global_step,
            "episodes_done": episodes_done,
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "mean_episode_virtual_time": float(np.mean(episode_virtual)) if episode_virtual else 0.0,
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "bc_loss": float(bc_losses[-1]) if bc_losses else 0.0,
            "buffer_size": n,
        }
        self.buffer.clear()
        self.update_count += 1
        self.history.append(stats)
        return stats

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.policy.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.env.N_ACTIONS,
            "hidden": self.cfg.hidden,
            "config": asdict(self.cfg),
            "global_step": self.global_step,
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.history = list(ckpt.get("history", []))

    def behavior_clone(self, n_episodes: int = 20, epochs: int = 3) -> None:
        """用 A0 启发式动作做行为克隆，给 PPO 一个可完成任务的初始策略。"""
        self.policy.train()
        states = []
        masks = []
        actions = []
        for ep in range(n_episodes):
            self._reset_train_env()
            while not self.env.env.done:
                s = self.env.state()
                m = self.env.action_mask()
                a = self.env.heuristic_action()
                if not m[a]:
                    break
                states.append(s)
                masks.append(m)
                actions.append(a)
                self.env.step(a)
        if not states:
            return
        X = self._to_tensor(np.asarray(states))
        M = self._to_tensor(np.asarray(masks), dtype=torch.bool)
        Y = self._to_tensor(np.asarray(actions), dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, M, Y)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)
        ce = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for xb, mb, yb in dl:
                logits, _ = self.policy(xb, mb)
                loss = ce(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

    def train(self, total_steps: Optional[int] = None) -> List[dict]:
        target = total_steps or self.cfg.total_steps
        logs = []
        t0 = time.time()
        while self.global_step < target:
            stats = self.collect_rollout()
            if stats:
                logs.append(stats)
                print(f"[PPO] step={stats['global_step']} ep={stats['episodes_done']} "
                      f"ret={stats['mean_episode_return']:.2f} "
                      f"V={stats['mean_episode_virtual_time']:.1f} "
                      f"pi_loss={stats['policy_loss']:.4f} v_loss={stats['value_loss']:.2f} "
                      f"H={stats['entropy']:.3f} wall={time.time()-t0:.1f}s")
        return logs
