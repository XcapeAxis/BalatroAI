from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SelfSupModelConfig:
    phase_vocab_size: int = 32
    action_vocab_size: int = 64
    hand_type_vocab_size: int = 16
    numeric_dim: int = 6
    embed_dim: int = 24
    hidden_dim: int = 128
    dropout: float = 0.1


class SelfSupEncoder(nn.Module):
    """Lightweight state-action encoder for trajectory self-supervision."""

    def __init__(
        self,
        *,
        phase_vocab_size: int,
        action_vocab_size: int,
        numeric_dim: int,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.phase_emb = nn.Embedding(max(2, int(phase_vocab_size)), int(embed_dim))
        self.action_emb = nn.Embedding(max(2, int(action_vocab_size)), int(embed_dim))
        self.numeric_proj = nn.Sequential(
            nn.Linear(int(numeric_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(hidden_dim) + (2 * int(embed_dim)), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )

    def forward(
        self,
        *,
        phase_ids: torch.Tensor,
        action_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        phase_h = self.phase_emb(phase_ids)
        action_h = self.action_emb(action_ids)
        num_h = self.numeric_proj(numeric_features)
        fused = torch.cat([num_h, phase_h, action_h], dim=-1)
        return self.fuse(fused)


class ScoreDeltaHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim // 2), 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class HandTypeHead(nn.Module):
    def __init__(self, hidden_dim: int, hand_type_vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim // 2), int(hand_type_vocab_size)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class SelfSupMultiHeadModel(nn.Module):
    def __init__(self, config: SelfSupModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SelfSupEncoder(
            phase_vocab_size=config.phase_vocab_size,
            action_vocab_size=config.action_vocab_size,
            numeric_dim=config.numeric_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.score_delta_head = ScoreDeltaHead(config.hidden_dim)
        self.hand_type_head = HandTypeHead(config.hidden_dim, config.hand_type_vocab_size)

    def forward(
        self,
        *,
        phase_ids: torch.Tensor,
        action_ids: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encoder(
            phase_ids=phase_ids,
            action_ids=action_ids,
            numeric_features=numeric_features,
        )
        return {
            "latent": latent,
            "score_delta": self.score_delta_head(latent),
            "hand_type_logits": self.hand_type_head(latent),
        }
