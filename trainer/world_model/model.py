from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trainer.models.ssl_state_encoder import StateEncoder, StateEncoderConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.model") from exc
    return torch, nn, F


@dataclass(frozen=True)
class WorldModelConfig:
    input_dim: int = 48
    latent_dim: int = 32
    hidden_dim: int = 96
    action_embed_dim: int = 32
    action_vocab_size: int = 4096
    resource_dim: int = 5
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "WorldModelConfig":
        payload = raw if isinstance(raw, dict) else {}
        return cls(
            input_dim=max(8, int(payload.get("input_dim") or 48)),
            latent_dim=max(8, int(payload.get("latent_dim") or 32)),
            hidden_dim=max(16, int(payload.get("hidden_dim") or 96)),
            action_embed_dim=max(8, int(payload.get("action_embed_dim") or 32)),
            action_vocab_size=max(32, int(payload.get("action_vocab_size") or 4096)),
            resource_dim=max(1, int(payload.get("resource_dim") or 5)),
            dropout=float(payload.get("dropout") or 0.1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LatentWorldModel:
    def __new__(cls, config: WorldModelConfig):
        torch, nn, F = _require_torch()

        class _Impl(nn.Module):
            def __init__(self, model_config: WorldModelConfig) -> None:
                super().__init__()
                self.config = model_config
                self.encoder = StateEncoder(
                    StateEncoderConfig(
                        input_dim=int(model_config.input_dim),
                        latent_dim=int(model_config.latent_dim),
                        hidden_dim=int(model_config.hidden_dim),
                        dropout=float(model_config.dropout),
                    )
                )
                self.action_embedding = nn.Embedding(
                    int(model_config.action_vocab_size),
                    int(model_config.action_embed_dim),
                )
                head_input = int(model_config.latent_dim) + int(model_config.action_embed_dim)
                hidden = max(32, int(model_config.hidden_dim))
                self.transition = nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.ReLU(),
                    nn.Dropout(float(model_config.dropout)),
                    nn.Linear(hidden, int(model_config.latent_dim)),
                )
                self.reward_head = nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                self.score_head = nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )
                self.resource_head = nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, int(model_config.resource_dim)),
                )
                self.uncertainty_head = nn.Sequential(
                    nn.Linear(head_input, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def encode(self, obs_t):
                return self.encoder(obs_t)

            def joint(self, z_t, action_ids):
                action_ids = action_ids.long().clamp(min=0, max=int(self.config.action_vocab_size) - 1)
                action_emb = self.action_embedding(action_ids)
                return torch.cat([z_t, action_emb], dim=-1)

            def forward(self, obs_t, action_ids, obs_t1=None):
                z_t = self.encode(obs_t)
                joint = self.joint(z_t, action_ids)
                z_next_pred = self.transition(joint)
                reward_pred = self.reward_head(joint).squeeze(-1)
                score_pred = self.score_head(joint).squeeze(-1)
                resource_pred = self.resource_head(joint)
                uncertainty_pred = F.softplus(self.uncertainty_head(joint).squeeze(-1))
                z_next_target = self.encode(obs_t1) if obs_t1 is not None else None
                next_state_proxy = z_next_pred.mean(dim=-1)
                return {
                    "z_t": z_t,
                    "z_next_pred": z_next_pred,
                    "z_next_target": z_next_target,
                    "reward_pred": reward_pred,
                    "score_pred": score_pred,
                    "resource_pred": resource_pred,
                    "uncertainty_pred": uncertainty_pred,
                    "next_state_proxy": next_state_proxy,
                }

        return _Impl(config)


def build_world_model(config: WorldModelConfig):
    return LatentWorldModel(config)


def save_world_model_checkpoint(
    *,
    path: str | Path,
    model: Any,
    optimizer: Any | None,
    config: WorldModelConfig,
    extra: dict[str, Any] | None = None,
) -> str:
    torch, _, _ = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "p45_world_model_checkpoint_v1",
        "model_config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (optimizer.state_dict() if optimizer is not None else {}),
        "extra": dict(extra or {}),
    }
    torch.save(payload, str(target))
    return str(target.resolve())


def load_world_model_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch, _, _ = _require_torch()
    checkpoint_path = Path(path)
    payload = torch.load(str(checkpoint_path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"world model checkpoint must be mapping: {checkpoint_path}")
    return payload


def load_world_model_from_checkpoint(path: str | Path, *, device: Any = "cpu") -> tuple[Any, WorldModelConfig, dict[str, Any]]:
    payload = load_world_model_checkpoint(path)
    model_config = WorldModelConfig.from_mapping(payload.get("model_config") if isinstance(payload.get("model_config"), dict) else {})
    model = build_world_model(model_config)
    model.load_state_dict(payload.get("model_state_dict") if isinstance(payload.get("model_state_dict"), dict) else {}, strict=False)
    model.to(device)
    model.eval()
    return model, model_config, payload
