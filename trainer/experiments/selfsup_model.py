from __future__ import annotations

import torch
import torch.nn as nn


class ReplayEncoder(nn.Module):
    """Lightweight shared encoder for replay-step features."""

    def __init__(
        self,
        *,
        action_vocab_size: int,
        numeric_dim: int,
        action_embed_dim: int = 16,
        hidden_dim: int = 96,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(int(max(2, action_vocab_size)), int(action_embed_dim))
        self.numeric_proj = nn.Linear(int(numeric_dim), int(hidden_dim))
        self.backbone = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) + int(action_embed_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, action_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = self.action_embed(action_ids)
        numeric_hidden = self.numeric_proj(numeric)
        fused = torch.cat([embedded, numeric_hidden], dim=-1)
        return self.backbone(fused)


class ReplaySelfSupModel(nn.Module):
    """P36 replay self-supervised model with pluggable heads."""

    def __init__(
        self,
        *,
        action_vocab_size: int,
        numeric_dim: int,
        action_embed_dim: int = 16,
        hidden_dim: int = 96,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ReplayEncoder(
            action_vocab_size=action_vocab_size,
            numeric_dim=numeric_dim,
            action_embed_dim=action_embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.mask_head = nn.Linear(int(latent_dim), int(action_vocab_size))
        self.next_delta_head = nn.Linear(int(latent_dim), 1)

    def encode(self, action_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return self.encoder(action_ids, numeric)

    def predict_mask_logits(self, action_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        latent = self.encode(action_ids, numeric)
        return self.mask_head(latent)

    def predict_next_delta(self, action_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        latent = self.encode(action_ids, numeric)
        return self.next_delta_head(latent).squeeze(-1)


__all__ = ["ReplayEncoder", "ReplaySelfSupModel"]
