"""Policy networks — MLP [512, 256, 128], ELU (see ARCHITECTURE.md §5)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        self.actor_backbone = nn.Sequential(*layers)
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        critic_layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            critic_layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        critic_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.actor_backbone(obs)
        mean = self.action_mean(features)
        return mean, self.action_log_std.expand_as(mean)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


class ActorExportWrapper(nn.Module):
    """ONNX export: actor mean only (Sentis deployment per ARCHITECTURE.md §6)."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.actor_backbone = ac.actor_backbone
        self.action_mean = ac.action_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_mean(self.actor_backbone(obs))


def build_actor_critic(obs_dim: int, action_dim: int) -> ActorCritic:
    return ActorCritic(obs_dim, action_dim)
