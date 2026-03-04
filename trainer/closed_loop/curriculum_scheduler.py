from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_weights(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, val in raw.items():
        token = str(key).strip()
        if not token:
            continue
        out[token] = max(0.0, _safe_float(val, 0.0))
    return out


def _normalize_slice_weights(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for slice_key, weights in raw.items():
        key = str(slice_key).strip()
        if not key:
            continue
        out[key] = _normalize_weights(weights)
    return out


def _default_curriculum(*, default_epochs: int) -> dict[str, Any]:
    return {
        "enabled": False,
        "phases": [
            {
                "name": "single_stage",
                "phase_index": 1,
                "epochs": max(1, int(default_epochs)),
                "target_rows": 0,
                "source_weights": {},
                "slice_weights": {},
            }
        ],
    }


def build_curriculum_plan(
    *,
    curriculum_cfg: dict[str, Any] | None,
    default_epochs: int,
    quick: bool,
    seeds: list[str] | None = None,
) -> dict[str, Any]:
    cfg = curriculum_cfg if isinstance(curriculum_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    phases_raw = cfg.get("phases") if isinstance(cfg.get("phases"), list) else []
    if not phases_raw:
        base = _default_curriculum(default_epochs=default_epochs)
        base["seeds"] = [str(s) for s in (seeds or [])]
        return base

    phases: list[dict[str, Any]] = []
    for idx, phase in enumerate(phases_raw, start=1):
        if not isinstance(phase, dict):
            continue
        phase_epochs = _safe_int(phase.get("epochs"), max(1, default_epochs))
        phase_target_rows = _safe_int(phase.get("target_rows"), 0)
        if quick:
            phase_target_rows = min(max(32, phase_target_rows or 160), 240)
        phases.append(
            {
                "name": str(phase.get("name") or f"phase_{idx}"),
                "phase_index": idx,
                "epochs": max(1, phase_epochs),
                "target_rows": max(0, phase_target_rows),
                "source_weights": _normalize_weights(phase.get("source_weights")),
                "slice_weights": _normalize_slice_weights(phase.get("slice_weights")),
                "notes": str(phase.get("notes") or ""),
            }
        )
    if not phases:
        base = _default_curriculum(default_epochs=default_epochs)
        base["seeds"] = [str(s) for s in (seeds or [])]
        return base
    return {
        "enabled": bool(enabled),
        "phase_count": len(phases),
        "phases": phases,
        "seeds": [str(s) for s in (seeds or [])],
    }


def _entry_slice_token(entry: dict[str, Any], slice_key: str) -> str:
    if slice_key in entry:
        return str(entry.get(slice_key))
    labels = entry.get("slice_labels") if isinstance(entry.get("slice_labels"), dict) else {}
    return str(labels.get(slice_key, "unknown"))


def compute_entry_phase_weight(entry: dict[str, Any], phase: dict[str, Any]) -> float:
    source_weights = phase.get("source_weights") if isinstance(phase.get("source_weights"), dict) else {}
    source_type = str(entry.get("source_type") or "unknown")
    source_weight = _safe_float(source_weights.get(source_type), 1.0) if source_weights else 1.0
    score = max(0.0, source_weight)

    slice_weights = phase.get("slice_weights") if isinstance(phase.get("slice_weights"), dict) else {}
    for slice_key, weights in slice_weights.items():
        if not isinstance(weights, dict):
            continue
        token = _entry_slice_token(entry, str(slice_key))
        if token in weights:
            score *= max(0.0, _safe_float(weights.get(token), 1.0))
        elif "default" in weights:
            score *= max(0.0, _safe_float(weights.get("default"), 1.0))
    return max(0.0, score)


def build_phase_allocations(
    *,
    entries: list[dict[str, Any]],
    phase: dict[str, Any],
    default_target_rows: int,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    target_rows = _safe_int(phase.get("target_rows"), 0) or max(1, int(default_target_rows))
    weighted: list[tuple[dict[str, Any], float, int]] = []
    denom = 0.0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        weight = compute_entry_phase_weight(entry, phase)
        if weight <= 0.0:
            continue
        sample_count = max(1, _safe_int(entry.get("sample_count"), 1))
        weighted_score = float(weight) * float(sample_count)
        denom += weighted_score
        weighted.append((entry, weighted_score, sample_count))
    if not weighted or denom <= 0.0:
        return []

    planned: list[dict[str, Any]] = []
    remaining = int(target_rows)
    for idx, (entry, weighted_score, sample_count) in enumerate(weighted):
        if idx == len(weighted) - 1:
            take = min(sample_count, remaining)
        else:
            take = int(round((weighted_score / denom) * float(target_rows)))
            take = max(0, min(sample_count, take))
            take = min(take, remaining)
        if take <= 0 and remaining > 0:
            take = min(1, sample_count, remaining)
        if take <= 0:
            continue
        planned.append(
            {
                "sample_id": str(entry.get("sample_id") or ""),
                "path": str(entry.get("path") or ""),
                "source_type": str(entry.get("source_type") or ""),
                "slice_labels": entry.get("slice_labels") if isinstance(entry.get("slice_labels"), dict) else {},
                "take_rows": int(take),
                "entry_weighted_score": float(weighted_score),
                "sample_count": int(sample_count),
            }
        )
        remaining -= int(take)
        if remaining <= 0:
            break

    if remaining > 0 and planned:
        for row in planned:
            room = int(row.get("sample_count") or 0) - int(row.get("take_rows") or 0)
            if room <= 0:
                continue
            bump = min(room, remaining)
            row["take_rows"] = int(row.get("take_rows") or 0) + int(bump)
            remaining -= int(bump)
            if remaining <= 0:
                break
    return planned

