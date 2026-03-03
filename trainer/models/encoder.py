from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BalatroEncoderConfig:
    input_dim: int
    latent_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1


class BalatroEncoder(nn.Module):
    """Shared lightweight encoder for dense feature vectors used by P36 tasks."""

    def __init__(self, config: BalatroEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(int(config.input_dim), int(config.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.latent_dim)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

