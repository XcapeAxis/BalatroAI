from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.hybrid.router_schema import supported_controller_ids


CHECKPOINT_SCHEMA = "p52_learned_router_checkpoint_v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for learned router model") from exc
    return torch, nn


class LearnedRouterModelBase:
    pass


def build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | tuple[int, ...] | None = None,
    dropout: float = 0.10,
) -> Any:
    torch, nn = _require_torch()
    dims = [int(input_dim), *[max(4, int(dim)) for dim in (hidden_dims or [128, 64])], int(output_dim)]
    layers: list[Any] = []
    for index in range(len(dims) - 1):
        in_dim = dims[index]
        out_dim = dims[index + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
    return nn.Sequential(*layers)


def apply_controller_mask(logits: Any, available_mask: Any) -> Any:
    torch, _nn = _require_torch()
    mask = available_mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
    invalid_rows = mask.sum(dim=-1) <= 0
    if bool(torch.any(invalid_rows)):
        masked_logits[invalid_rows] = logits[invalid_rows]
    return masked_logits


def softmax_with_mask(logits: Any, available_mask: Any, *, temperature: float = 1.0) -> Any:
    torch, _nn = _require_torch()
    safe_temperature = max(1e-6, float(temperature))
    masked_logits = apply_controller_mask(logits / safe_temperature, available_mask)
    return torch.softmax(masked_logits, dim=-1)


def predict_router_distribution(
    *,
    model: Any,
    feature_vector: list[float],
    available_mask: list[float],
    device: Any,
    temperature: float = 1.0,
) -> dict[str, Any]:
    torch, _nn = _require_torch()
    model.eval()
    with torch.no_grad():
        features = torch.tensor([feature_vector], dtype=torch.float32, device=device)
        mask = torch.tensor([available_mask], dtype=torch.float32, device=device)
        logits = model(features)
        probs = softmax_with_mask(logits, mask, temperature=temperature)[0].detach().cpu().tolist()
        logits_row = apply_controller_mask(logits, mask)[0].detach().cpu().tolist()
    controller_ids = supported_controller_ids()
    distribution = {controller_id: float(prob) for controller_id, prob in zip(controller_ids, probs)}
    best_controller = max(controller_ids, key=lambda controller_id: distribution.get(controller_id, 0.0))
    return {
        "controller_ids": controller_ids,
        "probabilities": distribution,
        "logits": {controller_id: float(value) for controller_id, value in zip(controller_ids, logits_row)},
        "selected_controller": best_controller,
        "confidence": float(distribution.get(best_controller, 0.0)),
    }


def save_router_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    torch, _nn = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def load_router_checkpoint(path: str | Path, *, map_location: str | None = None) -> dict[str, Any]:
    torch, _nn = _require_torch()
    payload = torch.load(Path(path), map_location=map_location or "cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid learned router checkpoint payload: {path}")
    if str(payload.get("schema") or "") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected learned router checkpoint schema: {payload.get('schema')}")
    return payload


def build_model_from_checkpoint(path: str | Path, *, map_location: str | None = None) -> tuple[Any, dict[str, Any]]:
    payload = load_router_checkpoint(path, map_location=map_location)
    model_config = payload.get("model_config") if isinstance(payload.get("model_config"), dict) else {}
    input_dim = int(model_config.get("input_dim") or 0)
    output_dim = int(model_config.get("output_dim") or len(supported_controller_ids()))
    hidden_dims = model_config.get("hidden_dims") if isinstance(model_config.get("hidden_dims"), list) else [128, 64]
    dropout = float(model_config.get("dropout") or 0.10)
    model = build_mlp(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout)
    state_dict = payload.get("state_dict") if isinstance(payload.get("state_dict"), dict) else {}
    model.load_state_dict(state_dict)
    return model, payload


def build_checkpoint_payload(
    *,
    state_dict: dict[str, Any],
    model_config: dict[str, Any],
    feature_encoder: dict[str, Any],
    controller_ids: list[str] | None = None,
    training_summary: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "generated_at": _now_iso(),
        "state_dict": state_dict,
        "model_config": dict(model_config or {}),
        "feature_encoder": dict(feature_encoder or {}),
        "controller_ids": list(controller_ids or supported_controller_ids()),
        "training_summary": dict(training_summary or {}),
        "calibration": dict(calibration or {}),
    }
