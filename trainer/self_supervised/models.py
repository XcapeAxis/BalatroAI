from __future__ import annotations

import torch
import torch.nn as nn


class P33SelfSupMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


__all__ = ["P33SelfSupMLP"]

