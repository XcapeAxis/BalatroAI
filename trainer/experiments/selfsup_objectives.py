from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_replay_steps(path: str | Path, *, max_samples: int = 0, valid_only: bool = True) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().lstrip("\ufeff")
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            if valid_only and not bool(row.get("valid_for_training", False)):
                continue
            steps.append(row)
            if max_samples > 0 and len(steps) >= int(max_samples):
                break
    return steps


def build_action_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    tokens = sorted({str(r.get("action_type") or "UNKNOWN").upper() for r in rows})
    if "[MASK]" not in tokens:
        tokens = ["[MASK]"] + tokens
    if "UNKNOWN" not in tokens:
        tokens = ["UNKNOWN"] + tokens
    vocab: dict[str, int] = {}
    for idx, tok in enumerate(tokens):
        vocab[tok] = idx
    return vocab


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _numeric_features(row: dict[str, Any]) -> list[float]:
    resources = row.get("resources_delta") if isinstance(row.get("resources_delta"), dict) else {}
    state_hashes = row.get("state_hashes") if isinstance(row.get("state_hashes"), dict) else {}
    valid = 1.0 if bool(row.get("valid_for_training", False)) else 0.0
    return [
        _safe_float(row.get("score_delta"), 0.0) / 300.0,
        _safe_float(row.get("reward"), 0.0) / 300.0,
        _safe_float(resources.get("chips_delta"), 0.0) / 300.0,
        _safe_float(resources.get("mult_delta"), 0.0) / 20.0,
        _safe_float(resources.get("money_delta"), 0.0) / 20.0,
        _safe_float(resources.get("hands_left_delta"), 0.0) / 10.0,
        _safe_float(resources.get("discards_left_delta"), 0.0) / 10.0,
        min(1.0, len(state_hashes) / 24.0),
        valid,
    ]


def build_training_rows(
    rows: list[dict[str, Any]],
    *,
    action_vocab: dict[str, int],
    mask_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    mask_id = int(action_vocab.get("[MASK]", 0))
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        action_token = str(row.get("action_type") or "UNKNOWN").upper()
        action_id = int(action_vocab.get(action_token, action_vocab.get("UNKNOWN", 0)))
        masked = rng.random() < float(mask_ratio)
        input_action_id = mask_id if masked else action_id

        next_delta_target = 0.0
        if idx + 1 < len(rows):
            next_delta_target = _safe_float(rows[idx + 1].get("score_delta"), 0.0)
        out.append(
            {
                "input_action_id": input_action_id,
                "target_action_id": action_id,
                "mask_target": 1.0 if masked else 0.0,
                "numeric": _numeric_features(row),
                "next_delta_target": float(next_delta_target),
            }
        )
    return out


def compute_mask_loss(model, batch, torch_mod):
    logits = model.predict_mask_logits(batch["input_action_ids"], batch["numeric"])
    losses = torch_mod.nn.functional.cross_entropy(logits, batch["target_action_ids"], reduction="none")
    weights = batch["mask_target"]
    denom = torch_mod.clamp(weights.sum(), min=1.0)
    weighted = (losses * weights).sum() / denom
    return weighted, logits


def compute_next_delta_loss(model, batch, torch_mod):
    preds = model.predict_next_delta(batch["input_action_ids"], batch["numeric"])
    return torch_mod.nn.functional.mse_loss(preds, batch["next_delta_target"]), preds


def compute_objective_loss(
    *,
    objective_type: str,
    model,
    batch,
    torch_mod,
    mask_weight: float = 1.0,
    next_delta_weight: float = 1.0,
):
    objective = str(objective_type or "hybrid").strip().lower()

    mask_loss = torch_mod.tensor(0.0, device=batch["numeric"].device)
    next_loss = torch_mod.tensor(0.0, device=batch["numeric"].device)
    logits = None
    preds = None

    if objective in {"mask", "hybrid"}:
        mask_loss, logits = compute_mask_loss(model, batch, torch_mod)
    if objective in {"next_delta", "hybrid"}:
        next_loss, preds = compute_next_delta_loss(model, batch, torch_mod)

    if objective == "mask":
        total = mask_loss
    elif objective == "next_delta":
        total = next_loss
    else:
        total = mask_loss * float(mask_weight) + next_loss * float(next_delta_weight)

    return {
        "total": total,
        "mask_loss": mask_loss,
        "next_delta_loss": next_loss,
        "logits": logits,
        "preds": preds,
    }


__all__ = [
    "build_action_vocab",
    "build_training_rows",
    "compute_mask_loss",
    "compute_next_delta_loss",
    "compute_objective_loss",
    "load_replay_steps",
]
