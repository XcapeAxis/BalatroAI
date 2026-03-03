from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trainer.models.encoder import BalatroEncoder, BalatroEncoderConfig


@dataclass(frozen=True)
class StateEncoderConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1


class StateEncoder(nn.Module):
    """State encoder wrapper that reuses the shared Balatro dense backbone."""

    def __init__(self, config: StateEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = BalatroEncoder(
            BalatroEncoderConfig(
                input_dim=int(config.input_dim),
                latent_dim=int(config.latent_dim),
                hidden_dim=int(config.hidden_dim),
                dropout=float(config.dropout),
            )
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)


class SSLProjectionHead(nn.Module):
    def __init__(self, *, input_dim: int, projection_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(32, int(input_dim))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(projection_dim)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


__all__ = ["SSLProjectionHead", "StateEncoder", "StateEncoderConfig"]
