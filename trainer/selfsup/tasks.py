from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainer.models.encoder import BalatroEncoder, BalatroEncoderConfig


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for P36 self-supervised tasks") from exc
    return torch, nn


def load_dataset_rows(path: Path, *, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                rows.append(item)
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
    return rows


def _row_state_vector(row: dict[str, Any]) -> list[float]:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    aux = row.get("aux") if isinstance(row.get("aux"), dict) else {}
    base = state.get("vector") if isinstance(state.get("vector"), list) else []
    vec = [float(x) for x in base]
    vec.extend(
        [
            float(aux.get("score_delta_t") or 0.0) / 250.0,
            float(aux.get("reward_t") or 0.0) / 250.0,
        ]
    )
    return vec


def _row_next_state_vector(row: dict[str, Any], default_dim: int) -> list[float]:
    future = row.get("future") if isinstance(row.get("future"), dict) else {}
    base = future.get("next_state_vector") if isinstance(future.get("next_state_vector"), list) else []
    vec = [float(x) for x in base]
    if len(vec) < int(default_dim):
        vec.extend([0.0 for _ in range(int(default_dim) - len(vec))])
    elif len(vec) > int(default_dim):
        vec = vec[: int(default_dim)]
    return vec


@dataclass(frozen=True)
class FutureValueBatch:
    features: list[list[float]]
    targets: list[float]


@dataclass(frozen=True)
class ActionTypeBatch:
    features_t: list[list[float]]
    features_tp1: list[list[float]]
    labels: list[int]
    label_vocab: dict[str, int]


def build_future_value_batch(rows: list[dict[str, Any]]) -> FutureValueBatch:
    features: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        future = row.get("future") if isinstance(row.get("future"), dict) else {}
        features.append(_row_state_vector(row))
        targets.append(float(future.get("delta_chips_k") or 0.0))
    return FutureValueBatch(features=features, targets=targets)


def build_action_type_batch(rows: list[dict[str, Any]]) -> ActionTypeBatch:
    features_t: list[list[float]] = []
    features_tp1: list[list[float]] = []
    label_tokens: list[str] = []
    inferred_dim = len(_row_state_vector(rows[0])) if rows else 0
    for row in rows:
        future = row.get("future") if isinstance(row.get("future"), dict) else {}
        token = str(future.get("next_action_type") or "UNKNOWN").upper()
        features_t.append(_row_state_vector(row))
        features_tp1.append(_row_next_state_vector(row, inferred_dim if inferred_dim > 0 else 1))
        label_tokens.append(token)
    vocab = {token: idx for idx, token in enumerate(sorted(set(label_tokens)))} if label_tokens else {"UNKNOWN": 0}
    labels = [int(vocab.get(token, 0)) for token in label_tokens]
    return ActionTypeBatch(
        features_t=features_t,
        features_tp1=features_tp1,
        labels=labels,
        label_vocab=vocab,
    )


class SelfSupFutureValueTask:
    def __init__(self, *, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        encoder_cfg = BalatroEncoderConfig(
            input_dim=int(input_dim),
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.model = nn.ModuleDict(
            {
                "encoder": BalatroEncoder(encoder_cfg),
                "head": nn.Linear(int(latent_dim), 1),
            }
        )

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def forward(self, x):
        latent = self.model["encoder"](x)
        pred = self.model["head"](latent).squeeze(-1)
        return pred

    def loss(self, pred, target):
        return self.torch.nn.functional.mse_loss(pred, target)


class SelfSupActionTypeTask:
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        torch, nn = _require_torch()
        self.torch = torch
        self.nn = nn
        encoder_cfg = BalatroEncoderConfig(
            input_dim=int(input_dim),
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.model = nn.ModuleDict(
            {
                "encoder": BalatroEncoder(encoder_cfg),
                "classifier": nn.Sequential(
                    nn.Linear(int(latent_dim) * 2, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(num_classes)),
                ),
            }
        )

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def forward(self, x_t, x_tp1):
        latent_t = self.model["encoder"](x_t)
        latent_tp1 = self.model["encoder"](x_tp1)
        logits = self.model["classifier"](self.torch.cat([latent_t, latent_tp1], dim=-1))
        return logits

    def loss(self, logits, labels):
        return self.torch.nn.functional.cross_entropy(logits, labels)


__all__ = [
    "ActionTypeBatch",
    "FutureValueBatch",
    "SelfSupActionTypeTask",
    "SelfSupFutureValueTask",
    "build_action_type_batch",
    "build_future_value_batch",
    "load_dataset_rows",
]
