from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import infer_source_run_id, read_json, read_jsonl, to_abs_path
from trainer.world_model.schema import (
    WorldModelSample,
    action_token_from_parts,
    fit_vector,
    hash_signal,
    make_sample_id,
    normalize_action_bucket,
    normalize_phase,
    phase_one_hot,
    resource_delta_vector,
    safe_float,
    safe_int,
    stable_hash_int,
)


SUPPORTED_SOURCE_TYPES = {
    "rl_rollout",
    "rollout",
    "selfsup_dataset",
    "p36_dataset",
    "trace_jsonl",
    "trace",
    "replay_manifest",
    "replay_mix_manifest",
}


def _extract_state_hashes(row: dict[str, Any]) -> dict[str, str]:
    if isinstance(row.get("state_hashes"), dict):
        return {
            str(key): str(value)
            for key, value in row.get("state_hashes", {}).items()
            if str(value).strip()
        }
    out: dict[str, str] = {}
    for key, value in row.items():
        if str(key).startswith("state_hash_") and str(value).strip():
            out[str(key)] = str(value)
    return out


def _extract_episode_id(row: dict[str, Any], *, fallback_path: Path) -> str:
    direct = str(row.get("episode_id") or "").strip()
    if direct:
        return direct
    seed = str(row.get("seed") or "").strip()
    run_id = str(row.get("run_id") or infer_source_run_id(fallback_path) or "").strip()
    if seed or run_id:
        return f"{fallback_path.stem}|{seed or 'noseed'}|{run_id or 'norun'}"
    return fallback_path.stem


def _extract_action_info(row: dict[str, Any], *, phase: str) -> tuple[str, dict[str, Any], int]:
    numeric_action = -1
    action_type = str(row.get("action_type") or "").strip().upper()
    action_payload = row.get("action_payload") if isinstance(row.get("action_payload"), dict) else {}

    if isinstance(row.get("action"), int):
        numeric_action = int(row.get("action"))
    elif isinstance(row.get("action"), dict):
        action_obj = dict(row.get("action") or {})
        action_type = str(action_obj.get("action_type") or action_type or "OTHER").strip().upper()
        action_payload = action_obj
    if numeric_action < 0 and row.get("action_id") is not None:
        numeric_action = safe_int(row.get("action_id"), -1)
    if not action_type and numeric_action >= 0:
        action_type = f"ACTION_{numeric_action}"
    if not action_type:
        action_type = "OTHER"
    token = action_token_from_parts(
        phase=phase,
        action_type=action_type,
        action_payload=action_payload,
        numeric_action=(numeric_action if numeric_action >= 0 else None),
    )
    return token, action_payload, numeric_action


def _scaled(value: Any, scale: float, *, lo: float = -4.0, hi: float = 4.0) -> float:
    if scale <= 0.0:
        return 0.0
    raw = safe_float(value, 0.0) / float(scale)
    return max(lo, min(hi, raw))


def _generic_feature_vector(row: dict[str, Any], *, feature_dim: int, source_type: str) -> list[float]:
    if isinstance(row.get("obs_vector"), list):
        return fit_vector(list(row.get("obs_vector") or []), feature_dim)
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    if isinstance(state.get("vector"), list):
        return fit_vector(list(state.get("vector") or []), feature_dim)

    phase = normalize_phase(row.get("phase") or row.get("state") or state.get("phase"))
    hashes = _extract_state_hashes(row)
    hash_values = [hash_signal(value) for _key, value in sorted(hashes.items())]
    while len(hash_values) < 16:
        hash_values.append(0.0)
    hash_values = hash_values[:16]
    resources = row.get("resources_delta") if isinstance(row.get("resources_delta"), dict) else {}

    features = [
        (sum(hash_values) / max(1, len([v for v in hash_values if v > 0.0])) if any(v > 0.0 for v in hash_values) else 0.0),
        max(hash_values or [0.0]),
        min([v for v in hash_values if v > 0.0] or [0.0]),
        float(len(hashes)) / 16.0,
        _scaled(row.get("reward"), 250.0),
        _scaled(row.get("score_delta"), 500.0),
        _scaled(resources.get("chips_delta"), 5000.0),
        _scaled(resources.get("money_delta"), 100.0),
        _scaled(resources.get("mult_delta"), 20.0),
        _scaled(resources.get("hands_left_delta"), 5.0),
        _scaled(resources.get("discards_left_delta"), 5.0),
        1.0 if bool(row.get("valid_for_training", True)) else 0.0,
        1.0 if bool(row.get("done") or ((row.get("meta") or {}).get("done") if isinstance(row.get("meta"), dict) else False)) else 0.0,
        _scaled(row.get("step_id"), 120.0),
        _scaled(len(_extract_action_info(row, phase=phase)[1]), 8.0),
        hash_signal(row.get("seed")),
        hash_signal(source_type),
        _scaled(row.get("ante"), 8.0),
        _scaled(row.get("round_num"), 20.0),
        _scaled(row.get("action_mask_legal_count"), 600.0),
        safe_float(row.get("action_mask_density"), 0.0),
    ]
    features.extend(phase_one_hot(phase))
    features.extend(hash_values)
    return fit_vector(features, feature_dim)


def _derive_slice_labels(row: dict[str, Any], *, phase: str, action_token: str) -> dict[str, Any]:
    ante = safe_int(row.get("ante"), 0)
    round_num = safe_int(row.get("round_num"), 0)
    action_bucket = normalize_action_bucket(
        str(row.get("action_type") or ((row.get("action") or {}).get("action_type") if isinstance(row.get("action"), dict) else "") or action_token)
    )

    if ante <= 1 and round_num <= 2:
        stage = "early"
    elif ante <= 3 and round_num <= 8:
        stage = "mid"
    else:
        stage = "late"

    mask_density = safe_float(row.get("action_mask_density"), 0.0)
    if mask_density > 0.0 and mask_density < 0.15:
        pressure = "high"
    elif mask_density > 0.0 and mask_density < 0.35:
        pressure = "medium"
    else:
        pressure = "low"

    return {
        "slice_stage": stage,
        "slice_resource_pressure": pressure,
        "slice_action_type": action_bucket.lower(),
        "slice_position_sensitive": bool(action_bucket == "POSITION"),
        "slice_stateful_joker_present": "unknown",
    }


def _assign_split(episode_id: str, *, train_ratio: float) -> str:
    score = stable_hash_int(f"split|{episode_id}", 10000) / 10000.0
    return "train" if score < max(0.0, min(1.0, float(train_ratio))) else "val"


def _sample_from_transition(
    *,
    source_id: str,
    source_type: str,
    source_category: str,
    source_path: Path,
    episode_id: str,
    step_id: int,
    seed: str,
    phase: str,
    action_token: str,
    action_numeric: int,
    obs_t: list[float],
    obs_t1: list[float],
    reward_t: float,
    score_delta_t: float,
    resource_delta_t: list[float],
    done_t: bool,
    valid_for_training: bool,
    split: str,
    action_vocab_size: int,
    metadata: dict[str, Any] | None = None,
    slice_labels: dict[str, Any] | None = None,
) -> WorldModelSample:
    sample_id = make_sample_id(
        [
            source_type,
            source_path,
            episode_id,
            step_id,
            seed,
            action_token,
        ]
    )
    return WorldModelSample(
        sample_id=sample_id,
        source_id=source_id,
        source_type=source_type,
        source_path=str(source_path),
        source_run_id=infer_source_run_id(source_path),
        seed=str(seed or ""),
        episode_id=episode_id,
        step_id=int(step_id),
        split=split,
        valid_for_training=bool(valid_for_training),
        phase_t=phase,
        action_token=action_token,
        action_id=stable_hash_int(action_token, max(1, int(action_vocab_size))),
        action_numeric=int(action_numeric),
        obs_t=list(obs_t),
        obs_t1=list(obs_t1),
        reward_t=float(reward_t),
        score_delta_t=float(score_delta_t),
        resource_delta_t=list(resource_delta_t),
        done_t=bool(done_t),
        source_category=source_category,
        slice_labels=dict(slice_labels or {}),
        metadata=dict(metadata or {}),
    )


def _build_rl_rollout_samples(
    *,
    source_id: str,
    source_type: str,
    path: Path,
    feature_dim: int,
    action_vocab_size: int,
    train_ratio: float,
    max_samples: int,
) -> tuple[list[WorldModelSample], list[str]]:
    rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    warnings: list[str] = []
    out: list[WorldModelSample] = []
    for idx, row in enumerate(rows):
        episode_id = _extract_episode_id(row, fallback_path=path)
        next_row = rows[idx + 1] if idx + 1 < len(rows) else None
        same_episode = isinstance(next_row, dict) and _extract_episode_id(next_row, fallback_path=path) == episode_id
        phase = normalize_phase(row.get("phase"))
        action_token, _payload, action_numeric = _extract_action_info(row, phase=phase)
        obs_t = fit_vector(list(row.get("obs_vector") or []), feature_dim)
        obs_t1 = fit_vector(list((next_row or {}).get("obs_vector") or row.get("obs_vector") or []), feature_dim) if same_episode else list(obs_t)
        reward_t = safe_float(row.get("reward"), 0.0)
        score_delta_t = safe_float(row.get("score_delta"), reward_t)
        done_t = bool(row.get("terminated") or row.get("truncated"))
        split = _assign_split(episode_id, train_ratio=train_ratio)
        slices = _derive_slice_labels(row, phase=phase, action_token=action_token)
        out.append(
            _sample_from_transition(
                source_id=source_id,
                source_type=source_type,
                source_category="sim_rollout",
                source_path=path,
                episode_id=episode_id,
                step_id=safe_int(row.get("step_id"), idx),
                seed=str(row.get("seed") or ""),
                phase=phase,
                action_token=action_token,
                action_numeric=action_numeric,
                obs_t=obs_t,
                obs_t1=obs_t1,
                reward_t=reward_t,
                score_delta_t=score_delta_t,
                resource_delta_t=[float(score_delta_t), 0.0, 0.0, 0.0, 0.0],
                done_t=done_t,
                valid_for_training=not bool(row.get("invalid_action", False)),
                split=split,
                action_vocab_size=action_vocab_size,
                metadata={
                    "schema": str(row.get("schema") or ""),
                    "legal_action_count": safe_int(row.get("action_mask_legal_count"), 0),
                    "action_mask_density": safe_float(row.get("action_mask_density"), 0.0),
                },
                slice_labels=slices,
            )
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    if not out:
        warnings.append(f"no rollout transitions found: {path}")
    return out, warnings


def _build_selfsup_dataset_samples(
    *,
    source_id: str,
    source_type: str,
    path: Path,
    feature_dim: int,
    action_vocab_size: int,
    train_ratio: float,
    max_samples: int,
) -> tuple[list[WorldModelSample], list[str]]:
    rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    warnings: list[str] = []
    out: list[WorldModelSample] = []
    for idx, row in enumerate(rows):
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        aux = row.get("aux") if isinstance(row.get("aux"), dict) else {}
        future = row.get("future") if isinstance(row.get("future"), dict) else {}
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        phase = normalize_phase(state.get("phase"))
        action_type = str(state.get("action_type") or "OTHER").strip().upper()
        action_token = action_token_from_parts(phase=phase, action_type=action_type, action_payload={})
        episode_id = str(meta.get("trajectory_id") or meta.get("run_id") or f"{path.stem}|{idx}")
        split = _assign_split(episode_id, train_ratio=train_ratio)
        slices = _derive_slice_labels(
            {
                "phase": phase,
                "action_type": action_type,
                "score_delta": aux.get("score_delta_t"),
                "reward": aux.get("reward_t"),
            },
            phase=phase,
            action_token=action_token,
        )
        out.append(
            _sample_from_transition(
                source_id=source_id,
                source_type=source_type,
                source_category="reconstructed_trace",
                source_path=path,
                episode_id=episode_id,
                step_id=safe_int(meta.get("step_id"), safe_int(meta.get("step_idx"), idx)),
                seed=str(aux.get("seed") or ""),
                phase=phase,
                action_token=action_token,
                action_numeric=-1,
                obs_t=fit_vector(list(state.get("vector") or []), feature_dim),
                obs_t1=fit_vector(list(future.get("next_state_vector") or []), feature_dim),
                reward_t=safe_float(aux.get("reward_t"), 0.0),
                score_delta_t=safe_float(aux.get("score_delta_t"), 0.0),
                resource_delta_t=[safe_float(aux.get("score_delta_t"), 0.0), 0.0, 0.0, 0.0, 0.0],
                done_t=bool(future.get("terminal_within_k")),
                valid_for_training=True,
                split=split,
                action_vocab_size=action_vocab_size,
                metadata={
                    "lookahead_k": safe_int(meta.get("lookahead_k"), 0),
                    "source": str(aux.get("source") or ""),
                    "future_delta_chips_k": safe_float(future.get("delta_chips_k"), 0.0),
                },
                slice_labels=slices,
            )
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    if not out:
        warnings.append(f"no selfsup transitions found: {path}")
    return out, warnings


def _build_trace_jsonl_samples(
    *,
    source_id: str,
    source_type: str,
    path: Path,
    feature_dim: int,
    action_vocab_size: int,
    train_ratio: float,
    max_samples: int,
) -> tuple[list[WorldModelSample], list[str]]:
    rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    warnings: list[str] = []
    out: list[WorldModelSample] = []
    for idx, row in enumerate(rows):
        episode_id = _extract_episode_id(row, fallback_path=path)
        next_row = rows[idx + 1] if idx + 1 < len(rows) else None
        same_episode = isinstance(next_row, dict) and _extract_episode_id(next_row, fallback_path=path) == episode_id
        phase = normalize_phase(row.get("phase") or row.get("state"))
        action_token, _payload, action_numeric = _extract_action_info(row, phase=phase)
        reward_t = safe_float(row.get("reward"), 0.0)
        score_delta_t = safe_float(row.get("score_delta"), reward_t)
        valid_for_training = bool(row.get("valid_for_training", True))
        done_t = bool(row.get("done") or ((row.get("meta") or {}).get("done") if isinstance(row.get("meta"), dict) else False))
        split = _assign_split(episode_id, train_ratio=train_ratio)
        obs_t = _generic_feature_vector(row, feature_dim=feature_dim, source_type=source_type)
        obs_t1 = (
            _generic_feature_vector(next_row, feature_dim=feature_dim, source_type=source_type)
            if same_episode and isinstance(next_row, dict)
            else list(obs_t)
        )
        slices = _derive_slice_labels(row, phase=phase, action_token=action_token)
        out.append(
            _sample_from_transition(
                source_id=source_id,
                source_type=source_type,
                source_category="reconstructed_trace",
                source_path=path,
                episode_id=episode_id,
                step_id=safe_int(row.get("step_id"), idx),
                seed=str(row.get("seed") or ""),
                phase=phase,
                action_token=action_token,
                action_numeric=action_numeric,
                obs_t=obs_t,
                obs_t1=obs_t1,
                reward_t=reward_t,
                score_delta_t=score_delta_t,
                resource_delta_t=resource_delta_vector(row.get("resources_delta") if isinstance(row.get("resources_delta"), dict) else None),
                done_t=done_t,
                valid_for_training=valid_for_training,
                split=split,
                action_vocab_size=action_vocab_size,
                metadata={
                    "schema": str(row.get("schema") or ""),
                    "invalid_reason": str(row.get("invalid_reason") or ""),
                    "trace_path": str(((row.get("meta") or {}).get("trace_path") if isinstance(row.get("meta"), dict) else "") or ""),
                },
                slice_labels=slices,
            )
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    if not out:
        warnings.append(f"no trace transitions found: {path}")
    return out, warnings


def _build_replay_manifest_samples(
    *,
    source_id: str,
    path: Path,
    feature_dim: int,
    action_vocab_size: int,
    train_ratio: float,
    max_samples: int,
) -> tuple[list[WorldModelSample], list[str]]:
    payload = read_json(path)
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return [], [f"replay manifest unreadable: {path}"]
    entries = payload.get("selected_entries") if isinstance(payload.get("selected_entries"), list) else []
    out: list[WorldModelSample] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_path = Path(str(entry.get("path") or ""))
        if not entry_path.is_absolute():
            entry_path = (path.parent.parent.parent.parent / entry_path).resolve()
        if not entry_path.exists():
            warnings.append(f"manifest entry missing: {entry_path}")
            continue
        low_name = entry_path.name.lower()
        format_hint = str(entry.get("format_hint") or "").strip().lower()
        per_source_limit = 0
        if max_samples > 0:
            per_source_limit = max(1, max_samples - len(out))
            if per_source_limit <= 0:
                break
        if "rollout" in low_name:
            samples, child_warnings = _build_rl_rollout_samples(
                source_id=str(entry.get("source_id") or source_id),
                source_type=str(entry.get("source_type") or "rl_rollout"),
                path=entry_path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=per_source_limit,
            )
        elif low_name == "dataset.jsonl":
            samples, child_warnings = _build_selfsup_dataset_samples(
                source_id=str(entry.get("source_id") or source_id),
                source_type=str(entry.get("source_type") or "selfsup_dataset"),
                path=entry_path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=per_source_limit,
            )
        elif format_hint in {"replay_steps_v1", "jsonl_unknown", "p13_session_trace"} or "trace" in low_name or "replay_steps" in low_name:
            samples, child_warnings = _build_trace_jsonl_samples(
                source_id=str(entry.get("source_id") or source_id),
                source_type=str(entry.get("source_type") or "trace_jsonl"),
                path=entry_path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=per_source_limit,
            )
        else:
            warnings.append(f"unsupported replay manifest entry skipped: {entry_path.name} format_hint={format_hint}")
            continue
        out.extend(samples)
        warnings.extend(child_warnings)
        if max_samples > 0 and len(out) >= max_samples:
            out = out[:max_samples]
            break
    if not out and not warnings:
        warnings.append(f"no usable samples resolved from replay manifest: {path}")
    return out, warnings


def _normalized_paths(repo_root: Path, source_cfg: dict[str, Any]) -> list[Path]:
    raw_paths = source_cfg.get("paths") if isinstance(source_cfg.get("paths"), list) else []
    if not raw_paths and str(source_cfg.get("path") or "").strip():
        raw_paths = [str(source_cfg.get("path") or "")]
    out: list[Path] = []
    for item in raw_paths:
        token = str(item).strip()
        if not token:
            continue
        out.append(to_abs_path(repo_root, token))
    return out


def _gather_matches(root: Path, patterns: list[str]) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    if not root.exists():
        return []
    out: list[Path] = []
    for pattern in patterns:
        for hit in root.glob(pattern):
            if hit.is_file():
                out.append(hit.resolve())
    return sorted(set(out), key=lambda p: str(p))


def resolve_source_files(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    quick: bool,
) -> tuple[list[Path], list[str]]:
    source_type = str(source_cfg.get("type") or "").strip().lower()
    warnings: list[str] = []
    if source_type not in SUPPORTED_SOURCE_TYPES:
        return [], [f"unsupported source type: {source_type}"]

    roots = _normalized_paths(repo_root, source_cfg)
    patterns = source_cfg.get("patterns") if isinstance(source_cfg.get("patterns"), list) else []
    if source_type in {"rl_rollout", "rollout"}:
        if not roots:
            roots = [
                (repo_root / "docs/artifacts/p42/rollouts").resolve(),
                (repo_root / "docs/artifacts/p44/rollouts").resolve(),
            ]
        if not patterns:
            patterns = ["**/rollout_steps.jsonl", "**/rollout_buffer.jsonl"]
    elif source_type in {"selfsup_dataset", "p36_dataset"}:
        if not roots:
            roots = [(repo_root / "docs/artifacts/p36/selfsup_datasets").resolve()]
        if not patterns:
            patterns = ["**/dataset.jsonl"]
    elif source_type in {"trace_jsonl", "trace"}:
        if not roots:
            roots = [
                (repo_root / "docs/artifacts/p13").resolve(),
                (repo_root / "docs/artifacts/p36/replay").resolve(),
                (repo_root / "docs/artifacts/p32").resolve(),
            ]
        if not patterns:
            patterns = ["**/oracle_trace*.jsonl", "**/replay_steps.jsonl", "**/session_*.jsonl"]
    elif source_type in {"replay_manifest", "replay_mix_manifest"}:
        if not roots:
            roots = [
                (repo_root / "docs/artifacts/p40/replay_mixer").resolve(),
                (repo_root / "docs/artifacts/p41/replay_mixer").resolve(),
            ]
        if not patterns:
            patterns = ["**/replay_mix_manifest.json"]

    files: list[Path] = []
    for root in roots:
        files.extend(_gather_matches(root, [str(pattern) for pattern in patterns]))
    files = sorted(set(files), key=lambda p: str(p), reverse=True)
    max_files = safe_int(source_cfg.get("max_files"), 0)
    if quick and max_files <= 0:
        max_files = 2
    if max_files > 0:
        files = files[:max_files]
    if not files:
        warnings.append(f"no files resolved for source type={source_type}")
    return files, warnings


def build_samples_from_source_config(
    *,
    repo_root: Path,
    source_cfg: dict[str, Any],
    feature_dim: int,
    action_vocab_size: int,
    train_ratio: float,
    quick: bool,
) -> tuple[list[WorldModelSample], dict[str, Any], list[str]]:
    source_id = str(source_cfg.get("id") or source_cfg.get("type") or "source")
    source_type = str(source_cfg.get("type") or "").strip().lower()
    files, warnings = resolve_source_files(repo_root=repo_root, source_cfg=source_cfg, quick=quick)
    max_samples = safe_int(source_cfg.get("max_samples"), 0)
    if quick and max_samples <= 0:
        max_samples = 256

    samples: list[WorldModelSample] = []
    per_file_counts: list[dict[str, Any]] = []
    for path in files:
        remaining = max_samples - len(samples) if max_samples > 0 else 0
        if max_samples > 0 and remaining <= 0:
            break
        if source_type in {"rl_rollout", "rollout"}:
            built, child_warnings = _build_rl_rollout_samples(
                source_id=source_id,
                source_type="rl_rollout",
                path=path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=(remaining if remaining > 0 else 0),
            )
        elif source_type in {"selfsup_dataset", "p36_dataset"}:
            built, child_warnings = _build_selfsup_dataset_samples(
                source_id=source_id,
                source_type="selfsup_dataset",
                path=path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=(remaining if remaining > 0 else 0),
            )
        elif source_type in {"trace_jsonl", "trace"}:
            built, child_warnings = _build_trace_jsonl_samples(
                source_id=source_id,
                source_type="trace_jsonl",
                path=path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=(remaining if remaining > 0 else 0),
            )
        else:
            built, child_warnings = _build_replay_manifest_samples(
                source_id=source_id,
                path=path,
                feature_dim=feature_dim,
                action_vocab_size=action_vocab_size,
                train_ratio=train_ratio,
                max_samples=(remaining if remaining > 0 else 0),
            )
        samples.extend(built)
        warnings.extend(child_warnings)
        per_file_counts.append({"path": str(path), "sample_count": len(built)})
        if max_samples > 0 and len(samples) >= max_samples:
            samples = samples[:max_samples]
            break

    split_counter: Counter[str] = Counter(sample.split for sample in samples)
    seed_counter: Counter[str] = Counter(sample.seed for sample in samples if sample.seed)
    stats = {
        "source_id": source_id,
        "source_type": source_type,
        "file_count": len(files),
        "sample_count": len(samples),
        "train_samples": int(split_counter.get("train", 0)),
        "val_samples": int(split_counter.get("val", 0)),
        "seed_count": len(seed_counter),
        "seeds": sorted(seed_counter.keys()),
        "files": per_file_counts,
        "warnings_count": len(warnings),
    }
    return samples, stats, warnings


def summarize_dataset_samples(samples: list[WorldModelSample]) -> dict[str, Any]:
    split_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    source_category_counter: Counter[str] = Counter()
    seed_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    valid_true = 0
    valid_false = 0
    reward_values: list[float] = []
    score_values: list[float] = []
    episode_counter: Counter[str] = Counter()
    slice_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for sample in samples:
        split_counter[sample.split] += 1
        source_counter[sample.source_type] += 1
        source_category_counter[sample.source_category] += 1
        if sample.seed:
            seed_counter[sample.seed] += 1
        action_counter[sample.action_token] += 1
        reward_values.append(float(sample.reward_t))
        score_values.append(float(sample.score_delta_t))
        episode_counter[sample.episode_id] += 1
        if sample.valid_for_training:
            valid_true += 1
        else:
            valid_false += 1
        for key, value in (sample.slice_labels or {}).items():
            slice_counts[str(key)][str(value)] += 1

    def _dist(counter: Counter[str]) -> list[dict[str, Any]]:
        total = max(1, sum(counter.values()))
        return [
            {"label": label, "count": int(count), "ratio": float(count) / float(total)}
            for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    return {
        "sample_count": len(samples),
        "split_distribution": _dist(split_counter),
        "source_distribution": _dist(source_counter),
        "source_category_distribution": _dist(source_category_counter),
        "seed_count": len(seed_counter),
        "seeds": sorted(seed_counter.keys()),
        "episode_count": len(episode_counter),
        "invalid_ratio": (float(valid_false) / max(1, len(samples))),
        "valid_samples": int(valid_true),
        "invalid_samples": int(valid_false),
        "reward_mean": (sum(reward_values) / max(1, len(reward_values))),
        "reward_min": min(reward_values) if reward_values else 0.0,
        "reward_max": max(reward_values) if reward_values else 0.0,
        "score_delta_mean": (sum(score_values) / max(1, len(score_values))),
        "score_delta_min": min(score_values) if score_values else 0.0,
        "score_delta_max": max(score_values) if score_values else 0.0,
        "top_actions": _dist(Counter(dict(action_counter.most_common(12)))),
        "slice_distribution": {
            key: _dist(counter)
            for key, counter in sorted(slice_counts.items())
        },
    }
