from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trainer.models.encoder import BalatroEncoder, BalatroEncoderConfig


class RLPolicy(nn.Module):
    """Policy head over a shared encoder for discrete RL actions."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        encoder: nn.Module | None = None,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if encoder is None:
            encoder = BalatroEncoder(
                BalatroEncoderConfig(
                    input_dim=self.obs_dim,
                    latent_dim=max(8, int(latent_dim)),
                    hidden_dim=max(16, int(hidden_dim)),
                    dropout=float(dropout),
                )
            )
        self.encoder = encoder
        self.latent_dim = int(self._infer_latent_dim(self.obs_dim))
        self.action_head = nn.Sequential(
            nn.Linear(self.latent_dim, max(16, self.latent_dim)),
            nn.ReLU(),
            nn.Linear(max(16, self.latent_dim), self.action_dim),
        )

    def _infer_latent_dim(self, obs_dim: int) -> int:
        with torch.no_grad():
            probe = torch.zeros((1, int(obs_dim)), dtype=torch.float32)
            encoded = self.encoder(probe)
            if encoded.ndim != 2:
                raise ValueError(f"encoder output must be [B, D], got shape={tuple(encoded.shape)}")
            return int(encoded.shape[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(obs)
        return self.action_head(latent)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim, "action_dim": self.action_dim}, path)

    def load_encoder_state_dict(self, state_dict: dict[str, Any], strict: bool = False) -> Any:
        return self.encoder.load_state_dict(state_dict, strict=strict)

