from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from trainer import action_space

try:
    import torch.nn as _torch_nn_base
except Exception:
    _torch_nn_base = None


MAX_ACTIONS = action_space.max_actions()


@dataclass
class MVPHandPolicyConfig:
    max_actions: int = MAX_ACTIONS
    rank_dim: int = 24
    suit_dim: int = 10
    numeric_dim: int = 24
    card_hidden: int = 128
    context_hidden: int = 96
    hidden_dim: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MVPHandPolicyConfig":
        raw = dict(payload or {})
        return cls(
            max_actions=int(raw.get("max_actions") or MAX_ACTIONS),
            rank_dim=int(raw.get("rank_dim") or 24),
            suit_dim=int(raw.get("suit_dim") or 10),
            numeric_dim=int(raw.get("numeric_dim") or 24),
            card_hidden=int(raw.get("card_hidden") or raw.get("card_width") or 128),
            context_hidden=int(raw.get("context_hidden") or raw.get("context_width") or 96),
            hidden_dim=int(raw.get("hidden_dim") or raw.get("hidden_width") or 256),
            dropout=float(raw.get("dropout") or 0.1),
        )


class MVPHandPolicy(_torch_nn_base.Module if _torch_nn_base is not None else object):
    """MVP-S2 用的轻量手牌阶段策略网络。"""

    def __init__(self, nn, config: MVPHandPolicyConfig | None = None):
        super().__init__()
        cfg = config or MVPHandPolicyConfig()
        self.config = cfg
        self.rank_emb = nn.Embedding(16, cfg.rank_dim)
        self.suit_emb = nn.Embedding(8, cfg.suit_dim)
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, cfg.numeric_dim),
            nn.LayerNorm(cfg.numeric_dim),
            nn.GELU(),
        )
        card_in = cfg.rank_dim + cfg.suit_dim + cfg.numeric_dim
        self.card_proj = nn.Sequential(
            nn.Linear(card_in, cfg.card_hidden),
            nn.LayerNorm(cfg.card_hidden),
            nn.GELU(),
        )
        self.card_gate = nn.Linear(cfg.card_hidden, 1)
        self.ctx_proj = nn.Sequential(
            nn.Linear(12, cfg.context_hidden),
            nn.LayerNorm(cfg.context_hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.card_hidden + cfg.context_hidden, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.max_actions),
        )

    def forward(self, batch):
        torch = __import__("torch")
        rank = batch["rank"]
        suit = batch["suit"]
        pad = batch["pad"]
        context = batch["context"]
        numeric = torch.stack(
            [
                batch["chip"],
                batch["enh"],
                batch["edt"],
                batch["seal"],
            ],
            dim=-1,
        )
        rank_h = self.rank_emb(rank)
        suit_h = self.suit_emb(suit)
        numeric_h = self.numeric_proj(numeric)
        card_x = torch.cat([rank_h, suit_h, numeric_h], dim=-1)
        card_h = self.card_proj(card_x)
        gate_logits = self.card_gate(card_h).squeeze(-1)
        gate_logits = torch.where(pad > 0, gate_logits, torch.full_like(gate_logits, -1e9))
        pooled = (card_h * torch.softmax(gate_logits, dim=1).unsqueeze(-1)).sum(dim=1)
        context = torch.sign(context) * torch.log1p(torch.abs(context))
        context_h = self.ctx_proj(context)
        fused = torch.cat([pooled, context_h], dim=-1)
        return self.head(fused)
