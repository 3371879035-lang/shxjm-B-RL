"""带动作屏蔽的 actor-critic 策略网络。"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedActorCritic(nn.Module):
    """固定 96 槽位动作空间。状态由确定性账本压缩得到。

    采用 128 维隐层 MLP；账本本身保存精确历史，因此这里不做 GRU 也能工作。
    如需论文中的 GRU 版本，可另设 recurrent=True 但正式推理仍需满足现实时间预算。
    """
    def __init__(self, state_dim: int, n_actions: int = 96, hidden: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
        self.prior_logit = nn.Parameter(torch.tensor(2.0))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                prior_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        logits = self.policy(h)
        if prior_action is not None:
            # A0 启发式动作作为策略先验/候选池中的保底动作；PPO 学习相对先验的修正。
            prior_action = prior_action.long().view(-1)
            prior_bias = self.prior_logit.detach() * F.one_hot(prior_action, self.n_actions).to(logits.dtype)
            if logits.dim() == 1:
                prior_bias = prior_bias.view(-1)
            logits = logits + prior_bias
        if action_mask is not None:
            # mask=True 表示合法
            mask = action_mask.bool()
            neg = torch.finfo(logits.dtype).min / 4.0
            logits = logits.masked_fill(~mask, neg)
        value = self.value(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state: torch.Tensor, action_mask: torch.Tensor,
            deterministic: bool = False,
            prior_action: Optional[torch.Tensor] = None) -> Tuple[int, float, float]:
        was_1d = state.dim() == 1
        if was_1d:
            state = state.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        if prior_action is not None:
            if prior_action.dim() == 0:
                prior_action = prior_action.unsqueeze(0)
        logits, value = self.forward(state, action_mask, prior_action)
        if deterministic:
            a = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            a = torch.multinomial(probs, num_samples=1).squeeze(-1)
        logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(-1)).squeeze(-1)
        if was_1d:
            return int(a.item()), float(logp.item()), float(value.item())
        return int(a.item()), float(logp.item()), float(value.item())

    def evaluate(self, states: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor,
                 prior_actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(states, masks, prior_actions)
        logp = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return logp, entropy, value
