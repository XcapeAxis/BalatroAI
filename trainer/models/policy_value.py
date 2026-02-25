"""Policy-Value model for hand+shop decisions."""

from __future__ import annotations

import torch
import torch.nn as nn

from trainer.features_shop import SHOP_CONTEXT_DIM


class PolicyValueModel(nn.Module):
    def __init__(self, max_actions: int, max_shop_actions: int):
        super().__init__()
        self.rank_emb = nn.Embedding(16, 16)
        self.suit_emb = nn.Embedding(8, 8)

        self.card_proj = nn.Sequential(
            nn.Linear(16 + 8 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.ctx_proj = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.hand_policy_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, max_actions),
        )
        self.hand_value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.shop_proj = nn.Sequential(
            nn.Linear(SHOP_CONTEXT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.shop_policy_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, max_shop_actions),
        )
        self.shop_value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _encode_hand(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rank = batch["rank"]
        suit = batch["suit"]
        chip = batch["chip"].unsqueeze(-1)
        enh = batch["enh"].unsqueeze(-1)
        edt = batch["edt"].unsqueeze(-1)
        seal = batch["seal"].unsqueeze(-1)
        pad = batch["pad"]

        r = self.rank_emb(rank)
        s = self.suit_emb(suit)
        card_x = torch.cat([r, s, chip, enh, edt, seal], dim=-1)
        card_h = self.card_proj(card_x)
        pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum

        ctx_h = self.ctx_proj(batch["context"])
        return torch.cat([pooled, ctx_h], dim=-1)

    def forward_hand(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode_hand(batch)
        return self.hand_policy_head(h), self.hand_value_head(h).squeeze(-1)

    def forward_shop(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shop_proj(batch["shop_context"])
        return self.shop_policy_head(h), self.shop_value_head(h).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_hand(batch)
