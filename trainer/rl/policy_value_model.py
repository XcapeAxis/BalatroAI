from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trainer.models.encoder import BalatroEncoder, BalatroEncoderConfig
from trainer.models.rl_policy import RLPolicy
from trainer.models.rl_value import RLValue


class PolicyValueModel(nn.Module):
    """Shared encoder with policy/value heads for PPO-lite."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.shared_encoder = BalatroEncoder(
            BalatroEncoderConfig(
                input_dim=self.obs_dim,
                latent_dim=max(8, int(latent_dim)),
                hidden_dim=max(16, int(hidden_dim)),
                dropout=float(dropout),
            )
        )
        self.policy_head = RLPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            encoder=self.shared_encoder,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.value_head = RLValue(
            obs_dim=self.obs_dim,
            encoder=self.shared_encoder,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def policy_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy_head(obs)

    def state_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_logits(obs), self.state_value(obs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "state_dict": self.state_dict(),
        }

