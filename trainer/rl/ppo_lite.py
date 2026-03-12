from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer.closed_loop.replay_manifest import build_seeds_payload
from trainer.closed_loop.failure_buckets import KNOWN_FAILURE_BUCKETS, scarce_failure_buckets
from trainer.monitoring.progress_schema import append_progress_event as append_unified_progress_event
from trainer.monitoring.progress_schema import build_progress_event, get_gpu_mem_mb
from trainer.registry.checkpoint_registry import register_checkpoint
from trainer.rl.checkpointing import save_torch_checkpoint, write_manifest
from trainer.rl.curriculum_rl import load_curriculum_scheduler
from trainer.rl.diagnostics import run_diagnostics
from trainer.rl.distributed_rollout import run_distributed_rollout
from trainer.rl.eval_multi_seed import run_multi_seed_evaluation
from trainer.rl.ppo_config import PPOConfig
from trainer.runtime.runtime_profile import load_runtime_profile


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            try:
                payload = json.loads(text)
            except Exception:
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
                else:
                    raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.distributions import Categorical
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.rl.ppo_lite") from exc
    return torch, F, Categorical


def _is_oom_error(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _component_name_for_schema(schema: str) -> str:
    token = str(schema or "").strip().lower()
    if token.startswith("p42_"):
        return "p42_rl"
    return "p44_rl"


def _resolved_runtime_devices(runtime_profile: dict[str, Any]) -> tuple[str, str]:
    resolved_profile = runtime_profile.get("resolved_profile") if isinstance(runtime_profile.get("resolved_profile"), dict) else {}
    resolved = runtime_profile.get("resolved") if isinstance(runtime_profile.get("resolved"), dict) else {}
    if not resolved and isinstance(resolved_profile.get("resolved"), dict):
        resolved = dict(resolved_profile.get("resolved") or {})
    learner_device = str(resolved.get("learner_device") or resolved.get("device") or "cpu")
    rollout_device = str(resolved.get("rollout_device") or "cpu")
    return learner_device, rollout_device


def _cpu_model_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.snapshot()
    try:
        torch, _, _ = _require_torch()
    except Exception:
        return snapshot
    state_dict = snapshot.get("state_dict")
    if isinstance(state_dict, dict):
        snapshot["state_dict"] = {
            key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in state_dict.items()
        }
    return snapshot


def _compute_gae(
    *,
    rewards: list[float],
    dones: list[bool],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    n = len(rewards)
    if n <= 0:
        return [], []
    adv = [0.0] * n
    last_adv = 0.0
    next_value = 0.0
    for idx in range(n - 1, -1, -1):
        not_done = 0.0 if bool(dones[idx]) else 1.0
        delta = float(rewards[idx]) + float(gamma) * next_value * not_done - float(values[idx])
        last_adv = delta + float(gamma) * float(gae_lambda) * not_done * last_adv
        adv[idx] = float(last_adv)
        next_value = float(values[idx])
    returns = [float(a + v) for a, v in zip(adv, values)]
    return adv, returns


def _build_masks_tensor(
    *,
    torch_mod,
    legal_action_ids: list[list[int]],
    action_dim: int,
    device,
):
    masks = torch_mod.zeros((len(legal_action_ids), int(action_dim)), dtype=torch_mod.float32, device=device)
    for row_idx, ids in enumerate(legal_action_ids):
        if not ids:
            masks[row_idx, 0] = 1.0
            continue
        for action_id in ids:
            if 0 <= int(action_id) < int(action_dim):
                masks[row_idx, int(action_id)] = 1.0
    return masks


def _masked_logits(logits, masks, torch_mod):
    neg = torch_mod.full_like(logits, -1e9)
    return torch_mod.where(masks > 0.0, logits, neg)


def _resolve_env_dimensions(cfg: PPOConfig) -> tuple[int, int]:
    from trainer.rl.env_adapter import RLEnvAdapter

    adapter = RLEnvAdapter(
        backend=str(cfg.env.backend),
        seed=str(cfg.seeds[0] if cfg.seeds else "AAAAAAA"),
        timeout_sec=float(cfg.env.timeout_sec),
        max_steps_per_episode=max(1, int(cfg.env.max_steps_per_episode)),
        max_auto_steps=max(1, int(cfg.env.max_auto_steps)),
        max_ante=max(0, int(cfg.env.max_ante)),
        auto_advance=bool(cfg.env.auto_advance),
        reward_config=cfg.env.reward,
    )
    try:
        return int(adapter.obs_dim), int(adapter.action_dim)
    finally:
        adapter.close()


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return str(result.stdout or "").strip() if int(result.returncode) == 0 else ""


def _read_jsonl_rows(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            rows.append(payload)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _read_json_payload(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalized_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _counter_ratios(counter_like: dict[str, Any]) -> dict[str, float]:
    counts = {str(key): _safe_int(value, 0) for key, value in dict(counter_like).items() if str(key).strip()}
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: round(float(value) / float(total), 6) for key, value in sorted(counts.items())}


def _normalized_weight_mapping(raw: dict[str, Any] | None) -> dict[str, float]:
    return {
        str(key).strip(): max(0.0, _safe_float(value, 0.0))
        for key, value in dict(raw or {}).items()
        if str(key).strip()
    }


def _tag_weight(tags: list[str], weights: dict[str, float]) -> tuple[float, str]:
    if not tags:
        return 1.0, ""
    normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
    if not normalized_tags:
        return 1.0, ""
    if not weights:
        return 1.0, normalized_tags[0]
    best_tag = ""
    best_weight = 1.0
    for token in normalized_tags:
        weight = max(0.0, _safe_float(weights.get(token), 0.0))
        if weight > best_weight:
            best_weight = weight
            best_tag = token
    if not best_tag:
        best_tag = normalized_tags[0]
    return best_weight, best_tag


def _resolve_hard_case_plan(*, cfg: PPOConfig, repo_root: Path) -> dict[str, Any]:
    hard_cfg = cfg.hard_case_sampling
    if not bool(hard_cfg.enabled):
        return {
            "enabled": False,
            "status": "disabled",
            "reason": "hard_case_sampling_disabled",
            "failure_pack_manifest": "",
            "selected_failure_count": 0,
            "selected_failure_seeds": [],
            "failure_seed_counts": {},
            "failure_type_counts": {},
            "seed_replay_factor": int(hard_cfg.seed_replay_factor),
            "include_base_seed": bool(hard_cfg.include_base_seed),
            "preserve_seed_identity": bool(hard_cfg.preserve_seed_identity),
        }

    manifest_token = str(hard_cfg.failure_pack_manifest or "").strip()
    manifest_path = _resolve_path(repo_root, manifest_token) if manifest_token else None
    if manifest_path is None or not manifest_path.exists():
        return {
            "enabled": True,
            "status": "stub",
            "reason": "failure_pack_manifest_missing",
            "failure_pack_manifest": str(manifest_path) if manifest_path else "",
            "selected_failure_count": 0,
            "selected_failure_seeds": [],
            "failure_seed_counts": {},
            "failure_type_counts": {},
            "seed_replay_factor": int(hard_cfg.seed_replay_factor),
            "include_base_seed": bool(hard_cfg.include_base_seed),
            "preserve_seed_identity": bool(hard_cfg.preserve_seed_identity),
        }

    payload = _read_yaml_or_json(manifest_path)
    raw_failures = payload.get("failures") if isinstance(payload.get("failures"), list) else []
    allowed_types = {str(token).strip() for token in hard_cfg.prioritize_failure_types if str(token).strip()}
    allowed_buckets = {str(token).strip() for token in hard_cfg.bucket_allowlist if str(token).strip()}
    allowed_slices = {str(token).strip() for token in hard_cfg.slice_allowlist if str(token).strip()}
    allowed_risk_tags = {str(token).strip() for token in hard_cfg.risk_tag_allowlist if str(token).strip()}
    bucket_sampling_weights = {
        str(bucket).strip(): max(0.0, _safe_float(weight, 0.0))
        for bucket, weight in (hard_cfg.bucket_sampling_weights or {}).items()
        if str(bucket).strip()
    }
    slice_sampling_weights = _normalized_weight_mapping(hard_cfg.slice_sampling_weights or {})
    risk_tag_sampling_weights = _normalized_weight_mapping(hard_cfg.risk_tag_sampling_weights or {})
    source_type_sampling_weights = {
        str(source_type).strip(): max(0.0, _safe_float(weight, 0.0))
        for source_type, weight in (hard_cfg.source_type_sampling_weights or {}).items()
        if str(source_type).strip()
    }
    bucket_quota_caps = {
        str(bucket).strip(): max(0, _safe_int(cap, 0))
        for bucket, cap in (hard_cfg.bucket_quota_caps or {}).items()
        if str(bucket).strip()
    }
    bucket_minimum_counts = {
        str(bucket).strip(): max(0, _safe_int(count, 0))
        for bucket, count in (hard_cfg.bucket_minimum_counts or {}).items()
        if str(bucket).strip()
    }
    slice_quota_caps = {
        str(tag).strip(): max(0, _safe_int(cap, 0))
        for tag, cap in (hard_cfg.slice_quota_caps or {}).items()
        if str(tag).strip()
    }
    slice_minimum_counts = {
        str(tag).strip(): max(0, _safe_int(count, 0))
        for tag, count in (hard_cfg.slice_minimum_counts or {}).items()
        if str(tag).strip()
    }
    risk_tag_quota_caps = {
        str(tag).strip(): max(0, _safe_int(cap, 0))
        for tag, cap in (hard_cfg.risk_tag_quota_caps or {}).items()
        if str(tag).strip()
    }
    source_type_quota_caps = {
        str(source_type).strip(): max(0, _safe_int(cap, 0))
        for source_type, cap in (hard_cfg.source_type_quota_caps or {}).items()
        if str(source_type).strip()
    }
    source_type_minimum_counts = {
        str(source_type).strip(): max(0, _safe_int(count, 0))
        for source_type, count in (hard_cfg.source_type_minimum_counts or {}).items()
        if str(source_type).strip()
    }
    source_variant_sampling_weights = {
        str(source_variant).strip(): max(0.0, _safe_float(weight, 0.0))
        for source_variant, weight in (hard_cfg.source_variant_sampling_weights or {}).items()
        if str(source_variant).strip()
    }
    source_variant_quota_caps = {
        str(source_variant).strip(): max(0, _safe_int(cap, 0))
        for source_variant, cap in (hard_cfg.source_variant_quota_caps or {}).items()
        if str(source_variant).strip()
    }
    source_variant_minimum_counts = {
        str(source_variant).strip(): max(0, _safe_int(count, 0))
        for source_variant, count in (hard_cfg.source_variant_minimum_counts or {}).items()
        if str(source_variant).strip()
    }
    bucket_seed_caps = {
        str(bucket).strip(): max(0, _safe_int(cap, 0))
        for bucket, cap in (hard_cfg.bucket_seed_caps or {}).items()
        if str(bucket).strip()
    }
    tracked_slice_tokens = {
        str(token).strip()
        for token in (set(slice_sampling_weights) | set(slice_minimum_counts) | set(slice_quota_caps))
        if str(token).strip()
    }

    failure_seed_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    failure_bucket_counts: Counter[str] = Counter()
    slice_tag_counts: Counter[str] = Counter()
    risk_tag_counts: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    for row in raw_failures:
        if not isinstance(row, dict):
            continue
        row_seed = str(row.get("seed") or "").strip()
        if not row_seed:
            continue
        failure_types = [str(token).strip() for token in (row.get("failure_types") or []) if str(token).strip()]
        if allowed_types and failure_types and not any(token in allowed_types for token in failure_types):
            continue
        failure_bucket = str(row.get("failure_bucket") or "unknown").strip() or "unknown"
        if allowed_buckets and failure_bucket not in allowed_buckets:
            continue
        slice_tags = [str(token).strip() for token in (row.get("slice_tags") or []) if str(token).strip()]
        risk_tags = [str(token).strip() for token in (row.get("risk_tags") or []) if str(token).strip()]
        if allowed_slices and not any(token in allowed_slices for token in slice_tags):
            continue
        if allowed_risk_tags and not any(token in allowed_risk_tags for token in risk_tags):
            continue
        source_type = str(row.get("source_type") or "unknown").strip() or "unknown"
        source_variant = str(row.get("source_variant") or "").strip()
        bucket_weight = max(0.01, bucket_sampling_weights.get(failure_bucket, 1.0))
        slice_weight, primary_slice_tag = _tag_weight(slice_tags, slice_sampling_weights)
        risk_weight, primary_risk_tag = _tag_weight(risk_tags, risk_tag_sampling_weights)
        source_type_weight = max(0.01, source_type_sampling_weights.get(source_type, 1.0))
        source_variant_weight = max(0.01, source_variant_sampling_weights.get(source_variant, 1.0)) if source_variant else 1.0
        payload = {
            "seed": row_seed,
            "failure_types": failure_types,
            "failure_bucket": failure_bucket,
            "slice_tags": slice_tags,
            "tracked_slice_tags": [
                str(tag).strip()
                for tag in slice_tags
                if str(tag).strip() and str(tag).strip() in tracked_slice_tokens
            ],
            "risk_tags": risk_tags,
            "primary_slice_tag": primary_slice_tag,
            "primary_risk_tag": primary_risk_tag,
            "episode_id": str(row.get("episode_id") or ""),
            "total_score": _safe_float(row.get("total_score"), 0.0),
            "replay_weight": max(
                0.01,
                _safe_float(row.get("replay_weight"), 1.0) * max(0.0, float(hard_cfg.replay_weight_scale)),
            ),
            "bucket_sampling_weight": bucket_weight,
            "slice_sampling_weight": slice_weight,
            "risk_tag_sampling_weight": risk_weight,
            "source_type_sampling_weight": source_type_weight,
            "source_variant_sampling_weight": source_variant_weight,
            "weighted_replay_priority": max(
                0.01,
                _safe_float(row.get("replay_weight"), 1.0)
                * max(0.0, float(hard_cfg.replay_weight_scale))
                * bucket_weight
                * slice_weight
                * risk_weight
                * source_type_weight
                * source_variant_weight,
            ),
            "selection_reason": str(row.get("selection_reason") or ""),
            "source_type": source_type,
            "source_variant": source_variant,
            "source_run_id": str(row.get("source_run_id") or ""),
        }
        candidate_rows.append(payload)

    candidate_rows.sort(
        key=lambda item: (
            -_safe_float(item.get("weighted_replay_priority"), 0.0),
            -_safe_float(item.get("bucket_sampling_weight"), 0.0),
            -_safe_float(item.get("slice_sampling_weight"), 0.0),
            -_safe_float(item.get("risk_tag_sampling_weight"), 0.0),
            -_safe_float(item.get("source_type_sampling_weight"), 0.0),
            -_safe_float(item.get("source_variant_sampling_weight"), 0.0),
            -_safe_float(item.get("replay_weight"), 0.0),
            _safe_float(item.get("total_score"), 0.0),
            str(item.get("episode_id") or ""),
        )
    )
    selected_rows: list[dict[str, Any]] = []
    selected_by_type: Counter[str] = Counter()
    selected_by_seed: Counter[str] = Counter()
    selected_by_bucket: Counter[str] = Counter()
    selected_by_bucket_seed: defaultdict[str, Counter[str]] = defaultdict(Counter)
    selected_by_slice: Counter[str] = Counter()
    selected_by_risk: Counter[str] = Counter()
    selected_by_source_type: Counter[str] = Counter()
    selected_by_source_variant: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    source_variant_counts: Counter[str] = Counter()
    per_type_cap = max(0, int(hard_cfg.max_failures_per_type or 0))
    per_seed_cap = max(0, int(hard_cfg.max_failures_per_seed or 0))
    max_failure_cases = max(1, int(hard_cfg.max_failure_cases or 0))
    primary_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    seen_primary_seeds: set[str] = set()
    selected_episode_ids: set[str] = set()
    max_primary_seeds = max(0, int(hard_cfg.max_failure_seeds or 0))

    def _row_slice_tokens(row: dict[str, Any]) -> list[str]:
        raw_tokens = row.get("tracked_slice_tags") if isinstance(row.get("tracked_slice_tags"), list) else []
        tokens: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            text = str(token).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            tokens.append(text)
        if tokens:
            return tokens
        primary_token = str(row.get("primary_slice_tag") or "").strip()
        return [primary_token] if primary_token else []

    def _row_has_slice_token(row: dict[str, Any], slice_token: str) -> bool:
        token = str(slice_token).strip()
        if not token:
            return False
        return token in _row_slice_tokens(row)

    def _slice_caps_allow(row: dict[str, Any]) -> bool:
        for slice_token in _row_slice_tokens(row):
            slice_cap = max(0, slice_quota_caps.get(slice_token, 0))
            if slice_cap > 0 and selected_by_slice[slice_token] >= slice_cap:
                return False
        return True

    def _row_is_selectable(row: dict[str, Any]) -> bool:
        seed_token = str(row.get("seed") or "")
        if per_seed_cap > 0 and selected_by_seed[seed_token] >= per_seed_cap:
            return False
        bucket_token = str(row.get("failure_bucket") or "unknown")
        bucket_cap = max(0, bucket_quota_caps.get(bucket_token, 0))
        if bucket_cap > 0 and selected_by_bucket[bucket_token] >= bucket_cap:
            return False
        bucket_seed_cap = max(0, bucket_seed_caps.get(bucket_token, 0))
        if bucket_seed_cap > 0 and selected_by_bucket_seed[bucket_token][seed_token] >= bucket_seed_cap:
            return False
        if not _slice_caps_allow(row):
            return False
        primary_risk_tag = str(row.get("primary_risk_tag") or "")
        if primary_risk_tag:
            risk_cap = max(0, risk_tag_quota_caps.get(primary_risk_tag, 0))
            if risk_cap > 0 and selected_by_risk[primary_risk_tag] >= risk_cap:
                return False
        source_type_token = str(row.get("source_type") or "unknown")
        source_type_cap = max(0, source_type_quota_caps.get(source_type_token, 0))
        if source_type_cap > 0 and selected_by_source_type[source_type_token] >= source_type_cap:
            return False
        source_variant_token = str(row.get("source_variant") or "")
        source_variant_cap = max(0, source_variant_quota_caps.get(source_variant_token, 0))
        if source_variant_token and source_variant_cap > 0 and selected_by_source_variant[source_variant_token] >= source_variant_cap:
            return False
        if bool(hard_cfg.balance_across_failure_types) and per_type_cap > 0:
            failure_tokens = row.get("failure_types") if isinstance(row.get("failure_types"), list) else []
            if any(selected_by_type[str(token)] >= per_type_cap for token in failure_tokens):
                return False
        episode_id = str(row.get("episode_id") or "")
        if episode_id and episode_id in selected_episode_ids:
            return False
        return True

    def _select_row(row: dict[str, Any]) -> None:
        selected_row = dict(row)
        selected_row["selected_for_training"] = True
        selected_rows.append(selected_row)
        seed_token = str(row.get("seed") or "")
        bucket_token = str(row.get("failure_bucket") or "unknown")
        primary_risk_tag = str(row.get("primary_risk_tag") or "")
        source_type_token = str(row.get("source_type") or "unknown")
        source_variant_token = str(row.get("source_variant") or "")
        episode_id = str(row.get("episode_id") or "")
        selected_by_seed[seed_token] += 1
        selected_by_bucket[bucket_token] += 1
        selected_by_bucket_seed[bucket_token][seed_token] += 1
        for slice_token in _row_slice_tokens(row):
            selected_by_slice[slice_token] += 1
        if primary_risk_tag:
            selected_by_risk[primary_risk_tag] += 1
        selected_by_source_type[source_type_token] += 1
        if source_variant_token:
            selected_by_source_variant[source_variant_token] += 1
        failure_seed_counts[seed_token] += 1
        for token in row.get("failure_types") if isinstance(row.get("failure_types"), list) else []:
            failure_type_counts[str(token)] += 1
            selected_by_type[str(token)] += 1
        failure_bucket_counts[bucket_token] += 1
        for tag in row.get("slice_tags") if isinstance(row.get("slice_tags"), list) else []:
            slice_tag_counts[str(tag)] += 1
        for tag in row.get("risk_tags") if isinstance(row.get("risk_tags"), list) else []:
            risk_tag_counts[str(tag)] += 1
        source_type_counts[str(row.get("source_type") or "unknown")] += 1
        if source_variant_token:
            source_variant_counts[source_variant_token] += 1
        if episode_id:
            selected_episode_ids.add(episode_id)

    for row in candidate_rows:
        seed_token = str(row.get("seed") or "")
        if seed_token and seed_token not in seen_primary_seeds and (max_primary_seeds <= 0 or len(seen_primary_seeds) < max_primary_seeds):
            primary_rows.append(row)
            seen_primary_seeds.add(seed_token)
        else:
            secondary_rows.append(row)

    ordered_rows = list(primary_rows) + list(secondary_rows)

    for bucket_token, target_count in sorted(
        bucket_minimum_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        minimum_count = max(0, target_count)
        if minimum_count <= 0:
            continue
        for row in ordered_rows:
            if str(row.get("failure_bucket") or "unknown") != bucket_token:
                continue
            if not _row_is_selectable(row):
                continue
            _select_row(row)
            if len(selected_rows) >= max_failure_cases or selected_by_bucket[bucket_token] >= minimum_count:
                break
        if len(selected_rows) >= max_failure_cases:
            break

    for source_type_token, target_count in sorted(
        source_type_minimum_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        minimum_count = max(0, target_count)
        if minimum_count <= 0:
            continue
        for row in ordered_rows:
            if str(row.get("source_type") or "unknown") != source_type_token:
                continue
            if not _row_is_selectable(row):
                continue
            _select_row(row)
            if len(selected_rows) >= max_failure_cases or selected_by_source_type[source_type_token] >= minimum_count:
                break
        if len(selected_rows) >= max_failure_cases:
            break

    for source_variant_token, target_count in sorted(
        source_variant_minimum_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        minimum_count = max(0, target_count)
        if minimum_count <= 0:
            continue
        for row in ordered_rows:
            if str(row.get("source_variant") or "") != source_variant_token:
                continue
            if not _row_is_selectable(row):
                continue
            _select_row(row)
            if len(selected_rows) >= max_failure_cases or selected_by_source_variant[source_variant_token] >= minimum_count:
                break
        if len(selected_rows) >= max_failure_cases:
            break

    for slice_token, target_count in sorted(
        slice_minimum_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        minimum_count = max(0, target_count)
        if minimum_count <= 0:
            continue
        for row in ordered_rows:
            if not _row_has_slice_token(row, slice_token):
                continue
            if not _row_is_selectable(row):
                continue
            _select_row(row)
            if len(selected_rows) >= max_failure_cases or selected_by_slice[slice_token] >= minimum_count:
                break
        if len(selected_rows) >= max_failure_cases:
            break

    for row in ordered_rows:
        if not _row_is_selectable(row):
            continue
        _select_row(row)
        if len(selected_rows) >= max_failure_cases:
            break

    replay_weight_by_seed: defaultdict[str, float] = defaultdict(float)
    for row in selected_rows:
        replay_weight_by_seed[str(row.get("seed") or "")] += _safe_float(row.get("replay_weight"), 1.0)
    mean_seed_weight = statistics.mean(replay_weight_by_seed.values()) if replay_weight_by_seed else 1.0
    seed_replay_counts: dict[str, int] = {}
    base_factor = max(1, int(hard_cfg.seed_replay_factor or 1))
    max_failure_seed_count = max(0, int(hard_cfg.max_failure_seeds or 0))
    for seed, replay_weight in sorted(
        replay_weight_by_seed.items(),
        key=lambda item: (-item[1], item[0]),
    )[: max_failure_seed_count or None]:
        replay_count = max(1, int(round(base_factor * _safe_ratio(replay_weight, mean_seed_weight))))
        if per_seed_cap > 0:
            replay_count = min(replay_count, max(1, per_seed_cap * base_factor))
        seed_replay_counts[str(seed)] = replay_count

    selected_failure_seeds = [
        seed
        for seed, _count in sorted(
            seed_replay_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    status = "ok" if selected_failure_seeds else "stub"
    reason = "" if status == "ok" else "no_failure_rows_selected"
    return {
        "enabled": True,
        "status": status,
        "reason": reason,
        "failure_pack_manifest": str(manifest_path),
        "failure_pack_status": str(payload.get("status") or ""),
        "selected_failure_count": len(selected_rows),
        "selected_failure_seeds": selected_failure_seeds,
        "seed_replay_counts": seed_replay_counts,
        "failure_seed_counts": dict(sorted(failure_seed_counts.items())),
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "failure_bucket_counts": dict(sorted(failure_bucket_counts.items())),
        "bucket_sampling_weights": dict(sorted(bucket_sampling_weights.items())),
        "bucket_quota_caps": dict(sorted(bucket_quota_caps.items())),
        "bucket_minimum_counts": dict(sorted(bucket_minimum_counts.items())),
        "bucket_seed_caps": dict(sorted(bucket_seed_caps.items())),
        "slice_sampling_weights": dict(sorted(slice_sampling_weights.items())),
        "slice_minimum_counts": dict(sorted(slice_minimum_counts.items())),
        "slice_quota_caps": dict(sorted(slice_quota_caps.items())),
        "risk_tag_sampling_weights": dict(sorted(risk_tag_sampling_weights.items())),
        "risk_tag_quota_caps": dict(sorted(risk_tag_quota_caps.items())),
        "source_type_sampling_weights": dict(sorted(source_type_sampling_weights.items())),
        "source_type_minimum_counts": dict(sorted(source_type_minimum_counts.items())),
        "source_type_quota_caps": dict(sorted(source_type_quota_caps.items())),
        "source_variant_sampling_weights": dict(sorted(source_variant_sampling_weights.items())),
        "source_variant_minimum_counts": dict(sorted(source_variant_minimum_counts.items())),
        "source_variant_quota_caps": dict(sorted(source_variant_quota_caps.items())),
        "bucket_selected_counts": dict(sorted(selected_by_bucket.items())),
        "bucket_selected_ratios": _counter_ratios(selected_by_bucket),
        "scarce_failure_buckets": scarce_failure_buckets(failure_bucket_counts),
        "slice_tag_counts": dict(sorted(slice_tag_counts.items())),
        "risk_tag_counts": dict(sorted(risk_tag_counts.items())),
        "slice_selected_counts": dict(sorted(selected_by_slice.items())),
        "slice_selected_ratios": _counter_ratios(selected_by_slice),
        "risk_selected_counts": dict(sorted(selected_by_risk.items())),
        "risk_selected_ratios": _counter_ratios(selected_by_risk),
        "source_type_selected_counts": dict(sorted(selected_by_source_type.items())),
        "source_type_selected_ratios": _counter_ratios(selected_by_source_type),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "source_variant_selected_counts": dict(sorted(selected_by_source_variant.items())),
        "source_variant_selected_ratios": _counter_ratios(selected_by_source_variant),
        "source_variant_counts": dict(sorted(source_variant_counts.items())),
        "selected_failures_preview": selected_rows[:16],
        "mean_replay_weight": float(statistics.mean([_safe_float(row.get('replay_weight'), 0.0) for row in selected_rows]))
        if selected_rows
        else 0.0,
        "mean_weighted_replay_priority": float(
            statistics.mean([_safe_float(row.get("weighted_replay_priority"), 0.0) for row in selected_rows])
        )
        if selected_rows
        else 0.0,
        "failure_type_coverage": int(len(failure_type_counts)),
        "failure_bucket_coverage": int(len(failure_bucket_counts)),
        "seed_replay_factor": int(hard_cfg.seed_replay_factor),
        "include_base_seed": bool(hard_cfg.include_base_seed),
        "preserve_seed_identity": bool(hard_cfg.preserve_seed_identity),
        "balance_across_failure_types": bool(hard_cfg.balance_across_failure_types),
        "max_failures_per_type": int(hard_cfg.max_failures_per_type),
        "max_failures_per_seed": int(hard_cfg.max_failures_per_seed),
        "known_failure_buckets": list(KNOWN_FAILURE_BUCKETS),
    }


def _build_rollout_seed_schedule(
    *,
    base_seeds: list[str],
    hard_case_plan: dict[str, Any],
) -> list[str]:
    schedule: list[str] = []
    if bool(hard_case_plan.get("include_base_seed", True)):
        schedule.extend([str(seed).strip() for seed in base_seeds if str(seed).strip()])
    if str(hard_case_plan.get("status") or "") != "ok":
        return schedule or [str(seed).strip() for seed in base_seeds if str(seed).strip()]
    replay_counts = hard_case_plan.get("seed_replay_counts") if isinstance(hard_case_plan.get("seed_replay_counts"), dict) else {}
    if replay_counts:
        for seed, count in sorted(replay_counts.items(), key=lambda item: (-_safe_int(item[1], 0), str(item[0]))):
            token = str(seed).strip()
            if not token:
                continue
            schedule.extend([token] * max(1, _safe_int(count, 1)))
    else:
        replay_factor = max(1, _safe_int(hard_case_plan.get("seed_replay_factor"), 2))
        for seed in hard_case_plan.get("selected_failure_seeds") if isinstance(hard_case_plan.get("selected_failure_seeds"), list) else []:
            token = str(seed).strip()
            if not token:
                continue
            schedule.extend([token] * replay_factor)
    return schedule or [str(seed).strip() for seed in base_seeds if str(seed).strip()]


def _select_self_imitation_payload(
    *,
    steps: list[dict[str, Any]],
    worker_stats_path: Path | None,
    cfg: PPOConfig,
    stage_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    self_cfg = cfg.self_imitation
    base_step_count = len(steps)
    disabled_payload = {
        "enabled": bool(self_cfg.enabled),
        "status": "disabled" if not bool(self_cfg.enabled) else "stub",
        "selected_episode_count": 0,
        "selected_step_count": 0,
        "replay_ratio": 0.0,
        "selected_episodes": [],
        "steps": [],
    }
    if not bool(self_cfg.enabled):
        return disabled_payload
    stage_name = _normalized_token((stage_payload or {}).get("stage"))
    stage_min = _normalized_token(self_cfg.stage_min)
    if stage_min and stage_name and stage_name != stage_min:
        phase_index = _safe_int((stage_payload or {}).get("phase_index"), 0)
        if not (stage_min.isdigit() and phase_index >= _safe_int(stage_min, 0)):
            disabled_payload["status"] = "disabled"
            disabled_payload["reason"] = f"stage_min_requires:{self_cfg.stage_min}"
            disabled_payload["stage"] = str((stage_payload or {}).get("stage") or "")
            return disabled_payload
    if worker_stats_path is None or not worker_stats_path.exists():
        disabled_payload["status"] = "stub"
        disabled_payload["reason"] = "worker_stats_missing"
        return disabled_payload

    worker_stats_payload = _read_json_payload(worker_stats_path)
    worker_stats = worker_stats_payload if isinstance(worker_stats_payload, list) else []
    episodes: list[dict[str, Any]] = []
    for worker in worker_stats:
        if not isinstance(worker, dict):
            continue
        for episode in worker.get("episodes") if isinstance(worker.get("episodes"), list) else []:
            if not isinstance(episode, dict):
                continue
            score = _safe_float(episode.get("final_score"), 0.0)
            reward = _safe_float(episode.get("reward"), 0.0)
            invalid_rate = _safe_float(episode.get("invalid_action_rate"), 1.0)
            if score < float(self_cfg.min_episode_score):
                continue
            if reward < float(self_cfg.min_episode_reward):
                continue
            if invalid_rate > float(self_cfg.max_invalid_action_rate):
                continue
            metric_name = str(self_cfg.selection_metric or "final_score").strip().lower()
            metric_value = reward if metric_name == "reward" else score
            if metric_value < float(self_cfg.quality_threshold):
                continue
            episodes.append(
                {
                    "episode_id": str(episode.get("episode_id") or ""),
                    "seed": str(episode.get("seed") or ""),
                    "reward": reward,
                    "final_score": score,
                    "invalid_action_rate": invalid_rate,
                    "selection_metric": metric_name,
                    "selection_value": metric_value,
                    "slice_tags": [str(token).strip() for token in (episode.get("slice_tags") or []) if str(token).strip()],
                    "dominant_phase": str(episode.get("dominant_phase") or ""),
                    "dominant_action_type": str(episode.get("dominant_action_type") or ""),
                }
            )
    if not episodes:
        disabled_payload["status"] = "stub"
        disabled_payload["reason"] = "no_eligible_episodes"
        return disabled_payload

    filtered_episodes: list[dict[str, Any]] = []
    allowed_slices = {str(token).strip() for token in self_cfg.slice_allowlist if str(token).strip()}
    allowed_phases = {str(token).strip().lower() for token in self_cfg.phase_allowlist if str(token).strip()}
    allowed_actions = {str(token).strip().upper() for token in self_cfg.action_type_allowlist if str(token).strip()}
    for episode in episodes:
        if allowed_slices:
            episode_slice_tags = {str(token).strip() for token in (episode.get("slice_tags") or []) if str(token).strip()}
            if not episode_slice_tags.intersection(allowed_slices):
                continue
        if allowed_phases and str(episode.get("dominant_phase") or "").strip().lower() not in allowed_phases:
            continue
        if allowed_actions and str(episode.get("dominant_action_type") or "").strip().upper() not in allowed_actions:
            continue
        filtered_episodes.append(episode)
    if not filtered_episodes:
        disabled_payload["status"] = "stub"
        disabled_payload["reason"] = "no_allowlisted_episodes"
        return disabled_payload

    filtered_episodes.sort(
        key=lambda item: (
            -_safe_float(item.get("selection_value"), 0.0),
            _safe_float(item.get("invalid_action_rate"), 1.0),
            str(item.get("episode_id") or ""),
        )
    )
    fraction_target = max(1, int(math.ceil(float(len(filtered_episodes)) * float(self_cfg.top_episode_fraction))))
    top_k_limit = min(len(filtered_episodes), max(1, int(self_cfg.top_k_episodes)))
    episode_limit = min(len(filtered_episodes), max(fraction_target, top_k_limit))
    selected_episodes = filtered_episodes[:episode_limit]
    selected_episode_ids = {str(row.get("episode_id") or "") for row in selected_episodes if str(row.get("episode_id") or "").strip()}
    max_replay_steps = int(self_cfg.max_replay_steps)
    if float(self_cfg.replay_ratio) > 0.0:
        max_replay_steps = min(
            max_replay_steps,
            max(1, int(math.ceil(float(base_step_count) * float(self_cfg.replay_ratio)))),
        )
    steps_by_episode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in steps:
        if not isinstance(row, dict):
            continue
        episode_id = str(row.get("episode_id") or "")
        if episode_id in selected_episode_ids:
            cloned = dict(row)
            info_summary = dict(cloned.get("info_summary") or {}) if isinstance(cloned.get("info_summary"), dict) else {}
            info_summary["replay_source"] = "self_imitation"
            if str((stage_payload or {}).get("stage") or "").strip():
                info_summary["self_imitation_stage"] = str((stage_payload or {}).get("stage") or "")
            cloned["info_summary"] = info_summary
            cloned["replay_source"] = "self_imitation"
            steps_by_episode[episode_id].append(cloned)
    selected_steps: list[dict[str, Any]] = []
    for episode in selected_episodes:
        selected_steps.extend(steps_by_episode.get(str(episode.get("episode_id") or ""), []))
        if max_replay_steps > 0 and len(selected_steps) >= max_replay_steps:
            selected_steps = selected_steps[:max_replay_steps]
            break
    replay_ratio = _safe_ratio(float(len(selected_steps)), float(base_step_count))
    return {
        "enabled": True,
        "status": "ok" if selected_steps else "stub",
        "selected_episode_count": len(selected_episodes),
        "selected_step_count": len(selected_steps),
        "replay_ratio": replay_ratio,
        "stage": str((stage_payload or {}).get("stage") or ""),
        "stage_min": str(self_cfg.stage_min or ""),
        "quality_threshold": float(self_cfg.quality_threshold),
        "bucket_allowlist": list(self_cfg.bucket_allowlist or []),
        "slice_allowlist": list(self_cfg.slice_allowlist or []),
        "phase_allowlist": list(self_cfg.phase_allowlist or []),
        "action_type_allowlist": list(self_cfg.action_type_allowlist or []),
        "bucket_filter_applied": False,
        "selected_episodes": selected_episodes,
        "steps": selected_steps,
    }


def _select_prior_rollout_reuse_payload(
    *,
    current_step_count: int,
    rollout_history: list[dict[str, Any]],
    cfg: PPOConfig,
) -> dict[str, Any]:
    disabled_payload = {
        "enabled": False,
        "status": "disabled",
        "selected_step_count": 0,
        "replay_ratio": 0.0,
        "selected_source_update_count": 0,
        "source_update_indices": [],
        "steps": [],
    }
    ratio = max(0.0, float(cfg.train.rollout_reuse_ratio))
    max_updates = max(0, int(cfg.train.rollout_reuse_updates))
    max_steps = max(0, int(cfg.train.rollout_reuse_max_steps))
    if ratio <= 0.0 or max_updates <= 0 or max_steps <= 0 or current_step_count <= 0:
        return disabled_payload
    recent_history = [row for row in rollout_history[-max_updates:] if isinstance(row, dict)]
    if not recent_history:
        disabled_payload["status"] = "stub"
        disabled_payload["reason"] = "no_prior_rollouts"
        return disabled_payload

    selected_limit = min(max_steps, max(1, int(math.ceil(float(current_step_count) * ratio))))
    candidate_steps: list[dict[str, Any]] = []
    source_update_indices: list[int] = []
    for history_row in reversed(recent_history):
        source_update = max(0, _safe_int(history_row.get("update_index"), 0))
        source_update_indices.append(source_update)
        history_steps = history_row.get("steps") if isinstance(history_row.get("steps"), list) else []
        for step in history_steps:
            if not isinstance(step, dict):
                continue
            cloned = dict(step)
            info_summary = dict(cloned.get("info_summary") or {}) if isinstance(cloned.get("info_summary"), dict) else {}
            info_summary["replay_source"] = "prior_rollout_reuse"
            info_summary["rollout_source_update"] = int(source_update)
            cloned["info_summary"] = info_summary
            cloned["replay_source"] = "prior_rollout_reuse"
            cloned["rollout_source_update"] = int(source_update)
            candidate_steps.append(cloned)
            if len(candidate_steps) >= selected_limit:
                break
        if len(candidate_steps) >= selected_limit:
            break
    if not candidate_steps:
        disabled_payload["status"] = "stub"
        disabled_payload["reason"] = "history_present_but_no_steps"
        disabled_payload["source_update_indices"] = [int(idx) for idx in source_update_indices]
        disabled_payload["selected_source_update_count"] = len(source_update_indices)
        return disabled_payload
    replay_ratio = _safe_ratio(float(len(candidate_steps)), float(current_step_count))
    return {
        "enabled": True,
        "status": "ok",
        "selected_step_count": len(candidate_steps),
        "replay_ratio": replay_ratio,
        "selected_source_update_count": len(source_update_indices),
        "source_update_indices": [int(idx) for idx in source_update_indices],
        "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
        "steps": candidate_steps,
    }


def _runtime_profile_name(runtime_profile: dict[str, Any]) -> str:
    if not isinstance(runtime_profile, dict):
        return ""
    return str(
        runtime_profile.get("profile_name")
        or runtime_profile.get("requested_profile")
        or ((runtime_profile.get("resolved_profile") or {}).get("profile_name") if isinstance(runtime_profile.get("resolved_profile"), dict) else "")
        or ""
    )


def _ppo_update_from_steps(
    *,
    model: Any,
    optimizer: Any,
    steps: list[dict[str, Any]],
    cfg: PPOConfig,
    device: Any,
) -> tuple[str, dict[str, Any]]:
    torch, F, Categorical = _require_torch()

    obs_list = [list(row.get("obs_vector") or []) for row in steps if isinstance(row, dict)]
    action_list = [_safe_int(row.get("action"), 0) for row in steps if isinstance(row, dict)]
    reward_list = [_safe_float(row.get("reward"), 0.0) for row in steps if isinstance(row, dict)]
    done_list = [
        bool(row.get("terminated") or row.get("truncated"))
        for row in steps
        if isinstance(row, dict)
    ]
    old_logprob_list = [_safe_float(row.get("action_logprob"), 0.0) for row in steps if isinstance(row, dict)]
    old_value_list = [_safe_float(row.get("value_pred"), 0.0) for row in steps if isinstance(row, dict)]
    legal_ids_list = [
        [int(x) for x in (row.get("legal_action_ids") if isinstance(row.get("legal_action_ids"), list) else [])]
        for row in steps
        if isinstance(row, dict)
    ]
    if not obs_list:
        return "stub", {}

    obs_t = torch.tensor(obs_list, dtype=torch.float32, device=device)
    actions_t = torch.tensor(action_list, dtype=torch.long, device=device)
    old_logprob_t = torch.tensor(old_logprob_list, dtype=torch.float32, device=device)

    advantages, returns = _compute_gae(
        rewards=reward_list,
        dones=done_list,
        values=old_value_list,
        gamma=float(cfg.train.gamma),
        gae_lambda=float(cfg.train.gae_lambda),
    )
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    if bool(cfg.train.normalize_advantage) and adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-6)

    masks_t = _build_masks_tensor(
        torch_mod=torch,
        legal_action_ids=legal_ids_list,
        action_dim=int(model.action_dim),
        device=device,
    )

    n = obs_t.shape[0]
    batch_size = max(1, min(int(cfg.train.minibatch_size), n))
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []

    for _epoch in range(int(cfg.train.ppo_epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits, values = model(obs_t[idx])
            values = values.squeeze(-1)
            masked = _masked_logits(logits, masks_t[idx], torch)
            dist = Categorical(logits=masked)
            new_logprob = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprob - old_logprob_t[idx])
            unclipped = ratio * adv_t[idx]
            clipped = torch.clamp(
                ratio,
                1.0 - float(cfg.train.clip_range),
                1.0 + float(cfg.train.clip_range),
            ) * adv_t[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, ret_t[idx])
            total_loss = (
                policy_loss
                + float(cfg.train.value_coef) * value_loss
                - float(cfg.train.entropy_coef) * entropy
            )
            if bool(cfg.train.nan_fail_fast) and (not torch.isfinite(total_loss)):
                return "failed", {"reason": "nan_loss"}
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if float(cfg.train.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            optimizer.step()
            approx_kl = torch.mean(old_logprob_t[idx] - new_logprob).item()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            approx_kls.append(float(approx_kl))

    return "ok", {
        "policy_loss": float(statistics.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(statistics.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(statistics.mean(entropies)) if entropies else 0.0,
        "kl_divergence": float(statistics.mean(approx_kls)) if approx_kls else 0.0,
        "return_mean": float(statistics.mean(returns)) if returns else 0.0,
    }


def _train_one_seed(
    *,
    seed: str,
    seed_index: int,
    seed_total: int,
    cfg: PPOConfig,
    cfg_raw: dict[str, Any],
    scheduler: Any,
    run_dir: Path,
    progress_path: Path,
    unified_progress_path: Path,
    warnings_log_path: Path,
    curriculum_applied_path: Path,
    runtime_profile: dict[str, Any],
    hard_case_plan: dict[str, Any],
) -> dict[str, Any]:
    torch, _, _ = _require_torch()
    from trainer.rl.policy_value_model import PolicyValueModel

    seed_int = int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:8], 16)
    torch.manual_seed(seed_int)
    learner_device, rollout_device = _resolved_runtime_devices(runtime_profile)
    resolved_profile = runtime_profile.get("resolved_profile") if isinstance(runtime_profile.get("resolved_profile"), dict) else {}
    resolved_runtime = runtime_profile.get("resolved") if isinstance(runtime_profile.get("resolved"), dict) else {}
    if not resolved_runtime and isinstance(resolved_profile.get("resolved"), dict):
        resolved_runtime = dict(resolved_profile.get("resolved") or {})
    try:
        torch.set_float32_matmul_precision(str(resolved_runtime.get("matmul_precision") or "high"))
    except Exception:
        pass
    device = torch.device(str(learner_device or "cpu"))
    obs_dim, action_dim = _resolve_env_dimensions(cfg)
    model = PolicyValueModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    seed_dir = run_dir / "seed_runs" / f"seed_{seed_index:03d}_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = seed_dir / "best.pt"
    last_ckpt = seed_dir / "last.pt"
    rollout_root = seed_dir / "rollout_updates"

    rewards_hist: list[float] = []
    scores_hist: list[float] = []
    invalid_hist: list[float] = []
    policy_loss_hist: list[float] = []
    value_loss_hist: list[float] = []
    entropy_hist: list[float] = []
    kl_hist: list[float] = []
    rollout_buffers: list[str] = []
    rollout_manifests: list[str] = []
    best_reward = -1e18
    status = "ok"
    warnings: list[str] = []
    learner_updates_per_sec_hist: list[float] = []
    rollout_steps_per_sec_hist: list[float] = []
    buffer_backlog_hist: list[float] = []
    gpu_mem_hist: list[float] = []
    weight_sync_count = 0
    oom_restart_count = 0
    component_name = _component_name_for_schema(cfg.schema)
    hard_case_updates: list[dict[str, Any]] = []
    self_imitation_updates: list[dict[str, Any]] = []
    self_imitation_ratio_hist: list[float] = []
    self_imitation_episode_hist: list[int] = []
    rollout_reuse_updates: list[dict[str, Any]] = []
    rollout_reuse_ratio_hist: list[float] = []
    rollout_reuse_step_hist: list[int] = []
    rollout_reuse_source_update_hist: list[int] = []
    actor_policy_lag_hist: list[int] = []
    cached_rollout_snapshot: dict[str, Any] | None = None
    cached_rollout_snapshot_update = 0
    prior_rollout_history: list[dict[str, Any]] = []

    for update_idx in range(1, int(cfg.train.max_updates) + 1):
        applied_cfg_dict, stage_payload = scheduler.apply_to_config(cfg_raw, training_iteration=update_idx)
        applied_cfg = PPOConfig.from_mapping(applied_cfg_dict)
        resolved_batch = max(
            8,
            _safe_int(
                resolved_runtime.get("micro_batch_size")
                or resolved_runtime.get("batch_size"),
                int(applied_cfg.train.minibatch_size),
            ),
        )
        applied_cfg.train.minibatch_size = int(resolved_batch)
        _append_jsonl(
            curriculum_applied_path,
            {
                "schema": "p44_curriculum_applied_v1",
                "ts": _now_iso(),
                "seed": str(seed),
                **stage_payload,
            },
        )

        distributed_seeds = list(applied_cfg.distributed.seeds) if applied_cfg.distributed.seeds else [str(seed)]
        rollout_seed_schedule = _build_rollout_seed_schedule(
            base_seeds=distributed_seeds,
            hard_case_plan=hard_case_plan,
        )
        rollout_episodes_per_worker = max(
            int(applied_cfg.distributed.episodes_per_worker),
            int(math.ceil(float(len(rollout_seed_schedule)) / float(max(1, int(applied_cfg.distributed.num_workers))))),
        )
        hard_case_update_payload = {
            "update_index": int(update_idx),
            "base_seeds": list(distributed_seeds),
            "rollout_seed_schedule": list(rollout_seed_schedule),
            "rollout_seed_count": len(rollout_seed_schedule),
            "episodes_per_worker": int(rollout_episodes_per_worker),
            "hard_case_status": str(hard_case_plan.get("status") or "disabled"),
            "selected_failure_count": int(hard_case_plan.get("selected_failure_count") or 0),
            "selected_failure_seeds": list(hard_case_plan.get("selected_failure_seeds") or []),
            "preserve_seed_identity": bool(hard_case_plan.get("preserve_seed_identity", False)),
        }
        hard_case_updates.append(hard_case_update_payload)
        update_started = time.time()
        actor_refresh_interval = max(1, int(applied_cfg.train.actor_refresh_interval_updates))
        if cached_rollout_snapshot is None or (update_idx - cached_rollout_snapshot_update) >= actor_refresh_interval:
            cached_rollout_snapshot = _cpu_model_snapshot(model)
            cached_rollout_snapshot_update = int(update_idx)
        policy_lag_updates = max(0, int(update_idx) - int(cached_rollout_snapshot_update))
        rollout_summary = run_distributed_rollout(
            policy_snapshot=cached_rollout_snapshot,
            policy_id="ppo_lite",
            num_workers=int(applied_cfg.distributed.num_workers),
            seeds=rollout_seed_schedule,
            episodes_per_worker=int(rollout_episodes_per_worker),
            max_steps_per_episode=int(applied_cfg.distributed.max_steps_per_episode),
            total_steps_cap=int(applied_cfg.rollout.total_steps_cap),
            run_id=f"{seed}-u{update_idx:03d}",
            out_dir=rollout_root / f"update_{update_idx:03d}",
            backend=str(applied_cfg.env.backend),
            reward_config=applied_cfg.env.reward,
            env_config={
                "timeout_sec": float(applied_cfg.env.timeout_sec),
                "max_steps_per_episode": int(applied_cfg.env.max_steps_per_episode),
                "max_auto_steps": int(applied_cfg.env.max_auto_steps),
                "max_ante": int(applied_cfg.env.max_ante),
                "auto_advance": bool(applied_cfg.env.auto_advance),
            },
            rollout_device=rollout_device,
            runtime_profile=runtime_profile,
            progress_path=unified_progress_path,
            include_steps_in_result=True,
            preserve_seed_identity=bool(hard_case_plan.get("preserve_seed_identity", False)),
        )
        rollout_buffers.append(str(rollout_summary.get("rollout_buffer_jsonl") or ""))
        rollout_manifests.append(str(rollout_summary.get("rollout_manifest") or ""))
        steps = rollout_summary.get("steps") if isinstance(rollout_summary.get("steps"), list) else []
        if not steps:
            status = "stub"
            warnings.append(f"update_{update_idx}:empty_rollout_steps")
            break
        self_imitation_payload = _select_self_imitation_payload(
            steps=steps,
            worker_stats_path=Path(str(rollout_summary.get("worker_stats_json") or "")) if str(rollout_summary.get("worker_stats_json") or "").strip() else None,
            cfg=applied_cfg,
            stage_payload=stage_payload,
        )
        rollout_reuse_payload = _select_prior_rollout_reuse_payload(
            current_step_count=len(steps),
            rollout_history=prior_rollout_history,
            cfg=applied_cfg,
        )
        reuse_steps = rollout_reuse_payload.get("steps") if isinstance(rollout_reuse_payload.get("steps"), list) else []
        replay_steps = self_imitation_payload.get("steps") if isinstance(self_imitation_payload.get("steps"), list) else []
        train_steps = list(steps) + list(reuse_steps) + list(replay_steps)
        self_imitation_update_payload = {
            "update_index": int(update_idx),
            "status": str(self_imitation_payload.get("status") or "disabled"),
            "selected_episode_count": int(self_imitation_payload.get("selected_episode_count") or 0),
            "selected_step_count": int(self_imitation_payload.get("selected_step_count") or 0),
            "replay_ratio": _safe_float(self_imitation_payload.get("replay_ratio"), 0.0),
            "stage": str(self_imitation_payload.get("stage") or ""),
            "stage_min": str(self_imitation_payload.get("stage_min") or ""),
            "quality_threshold": _safe_float(self_imitation_payload.get("quality_threshold"), 0.0),
            "bucket_allowlist": list(self_imitation_payload.get("bucket_allowlist") or []),
            "slice_allowlist": list(self_imitation_payload.get("slice_allowlist") or []),
            "phase_allowlist": list(self_imitation_payload.get("phase_allowlist") or []),
            "action_type_allowlist": list(self_imitation_payload.get("action_type_allowlist") or []),
            "selected_episodes": list(self_imitation_payload.get("selected_episodes") or [])[:8],
        }
        self_imitation_updates.append(self_imitation_update_payload)
        rollout_reuse_update_payload = {
            "update_index": int(update_idx),
            "status": str(rollout_reuse_payload.get("status") or "disabled"),
            "selected_step_count": int(rollout_reuse_payload.get("selected_step_count") or 0),
            "replay_ratio": _safe_float(rollout_reuse_payload.get("replay_ratio"), 0.0),
            "selected_source_update_count": int(rollout_reuse_payload.get("selected_source_update_count") or 0),
            "source_update_indices": list(rollout_reuse_payload.get("source_update_indices") or []),
            "actor_refresh_interval_updates": actor_refresh_interval,
            "policy_lag_updates": int(policy_lag_updates),
        }
        rollout_reuse_updates.append(rollout_reuse_update_payload)

        oom_policy = str(resolved_runtime.get("oom_fallback_policy") or "reduce_batch").strip().lower()
        attempt_cfg = applied_cfg
        active_minibatch = max(8, int(attempt_cfg.train.minibatch_size))
        while True:
            attempt_cfg.train.minibatch_size = int(active_minibatch)
            try:
                update_status, update_metrics = _ppo_update_from_steps(
                    model=model,
                    optimizer=optimizer,
                    steps=train_steps,
                    cfg=attempt_cfg,
                    device=device,
                )
                break
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                oom_restart_count += 1
                warnings.append(f"update_{update_idx}:oom:minibatch={active_minibatch}")
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if oom_policy == "reduce_batch" and active_minibatch > 8:
                    active_minibatch = max(8, active_minibatch // 2)
                    continue
                if oom_policy == "cpu_fallback" and str(device).startswith("cuda"):
                    device = torch.device("cpu")
                    learner_device = "cpu"
                    model = model.to(device)
                    continue
                raise
        if update_status != "ok":
            status = update_status
            warnings.append(
                f"update_{update_idx}:{str(update_metrics.get('reason') or 'ppo_update_failed')}"
            )
            break

        reward_values = [_safe_float(row.get("reward"), 0.0) for row in steps]
        rollout_stats_payload = _read_json(Path(str(rollout_summary.get("rollout_stats_json") or ""))) or {}
        mean_reward = float(statistics.mean(reward_values)) if reward_values else 0.0
        mean_score = _safe_float(rollout_stats_payload.get("mean_score"), mean_reward)
        invalid_rate = _safe_float(rollout_summary.get("invalid_action_rate"), 0.0)
        rollout_steps_per_sec = _safe_float(rollout_summary.get("rollout_steps_per_sec"), 0.0)
        learner_elapsed = max(1e-6, time.time() - update_started)
        learner_updates_per_sec = 1.0 / learner_elapsed
        backlog_steps = max(
            0.0,
            float(int(rollout_summary.get("step_count") or 0) - int(attempt_cfg.train.minibatch_size)),
        )
        gpu_mem_mb = get_gpu_mem_mb(device)

        rewards_hist.append(mean_reward)
        scores_hist.append(mean_score)
        invalid_hist.append(invalid_rate)
        self_imitation_ratio_hist.append(_safe_float(self_imitation_update_payload.get("replay_ratio"), 0.0))
        self_imitation_episode_hist.append(int(self_imitation_update_payload.get("selected_episode_count") or 0))
        rollout_reuse_ratio_hist.append(_safe_float(rollout_reuse_update_payload.get("replay_ratio"), 0.0))
        rollout_reuse_step_hist.append(int(rollout_reuse_update_payload.get("selected_step_count") or 0))
        rollout_reuse_source_update_hist.append(int(rollout_reuse_update_payload.get("selected_source_update_count") or 0))
        actor_policy_lag_hist.append(int(policy_lag_updates))
        policy_loss_hist.append(_safe_float(update_metrics.get("policy_loss"), 0.0))
        value_loss_hist.append(_safe_float(update_metrics.get("value_loss"), 0.0))
        entropy_hist.append(_safe_float(update_metrics.get("entropy"), 0.0))
        kl_hist.append(_safe_float(update_metrics.get("kl_divergence"), 0.0))
        rollout_steps_per_sec_hist.append(rollout_steps_per_sec)
        learner_updates_per_sec_hist.append(learner_updates_per_sec)
        buffer_backlog_hist.append(backlog_steps)
        weight_sync_count += 1
        if gpu_mem_mb is not None:
            gpu_mem_hist.append(float(gpu_mem_mb))

        if invalid_rate > float(cfg.train.invalid_action_warn_threshold):
            warnings.append(
                "update_{idx}:invalid_action_rate={value:.4f}>threshold={thr:.4f}".format(
                    idx=update_idx,
                    value=invalid_rate,
                    thr=float(cfg.train.invalid_action_warn_threshold),
                )
            )
        if abs(_safe_float(update_metrics.get("kl_divergence"), 0.0)) > float(cfg.train.max_kl_warn):
            warnings.append(
                "update_{idx}:kl_divergence={value:.4f}>max_kl_warn={thr:.4f}".format(
                    idx=update_idx,
                    value=_safe_float(update_metrics.get("kl_divergence"), 0.0),
                    thr=float(cfg.train.max_kl_warn),
                )
            )

        checkpoint_payload = {
            "schema": "p44_rl_checkpoint_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "update_index": int(update_idx),
            "model": _cpu_model_snapshot(model),
            "optimizer_state": optimizer.state_dict(),
            "train_cfg": applied_cfg.to_dict(),
            "curriculum_stage": stage_payload,
            "metrics": {
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "invalid_action_rate": invalid_rate,
                "policy_loss": _safe_float(update_metrics.get("policy_loss"), 0.0),
                "value_loss": _safe_float(update_metrics.get("value_loss"), 0.0),
                "entropy": _safe_float(update_metrics.get("entropy"), 0.0),
                "kl_divergence": _safe_float(update_metrics.get("kl_divergence"), 0.0),
                "rollout_steps_per_sec": rollout_steps_per_sec,
                "learner_updates_per_sec": learner_updates_per_sec,
                "buffer_backlog_steps": backlog_steps,
                "weight_sync_count": weight_sync_count,
                "oom_restart_count": oom_restart_count,
                "self_imitation_selected_episodes": int(self_imitation_update_payload.get("selected_episode_count") or 0),
                "self_imitation_replay_ratio": _safe_float(self_imitation_update_payload.get("replay_ratio"), 0.0),
                "rollout_reuse_selected_steps": int(rollout_reuse_update_payload.get("selected_step_count") or 0),
                "rollout_reuse_ratio": _safe_float(rollout_reuse_update_payload.get("replay_ratio"), 0.0),
                "rollout_reuse_source_updates": int(rollout_reuse_update_payload.get("selected_source_update_count") or 0),
                "actor_refresh_interval_updates": int(actor_refresh_interval),
                "policy_lag_updates": int(policy_lag_updates),
            },
            "runtime_profile": runtime_profile,
            "hard_case_sampling": hard_case_update_payload,
            "self_imitation": self_imitation_update_payload,
            "rollout_reuse": rollout_reuse_update_payload,
        }
        save_torch_checkpoint(last_ckpt, checkpoint_payload)
        if mean_reward >= best_reward:
            best_reward = mean_reward
            save_torch_checkpoint(best_ckpt, checkpoint_payload)

        _append_jsonl(
            progress_path,
            {
                "schema": "p44_rl_progress_v1",
                "ts": _now_iso(),
                "seed": str(seed),
                "seed_index": int(seed_index),
                "seed_total": int(seed_total),
                "update": int(update_idx),
                "status": status,
                "curriculum_stage": str(stage_payload.get("stage") or ""),
                "phase_index": int(stage_payload.get("phase_index") or 0),
                "mean_reward": mean_reward,
                "mean_score": mean_score,
                "policy_loss": _safe_float(update_metrics.get("policy_loss"), 0.0),
                "value_loss": _safe_float(update_metrics.get("value_loss"), 0.0),
                "entropy": _safe_float(update_metrics.get("entropy"), 0.0),
                "kl_divergence": _safe_float(update_metrics.get("kl_divergence"), 0.0),
                "invalid_action_rate": invalid_rate,
                "rollout_steps": int(rollout_summary.get("step_count") or 0),
                "rollout_dir": str(rollout_summary.get("run_dir") or ""),
                "rollout_steps_per_sec": rollout_steps_per_sec,
                "learner_updates_per_sec": learner_updates_per_sec,
                "buffer_backlog_steps": backlog_steps,
                "weight_sync_count": weight_sync_count,
                "oom_restart_count": oom_restart_count,
                "hard_case_seed_count": int(len(rollout_seed_schedule)),
                "hard_case_selected_failures": int(hard_case_plan.get("selected_failure_count") or 0),
                "self_imitation_selected_episodes": int(self_imitation_update_payload.get("selected_episode_count") or 0),
                "self_imitation_replay_ratio": _safe_float(self_imitation_update_payload.get("replay_ratio"), 0.0),
                "rollout_reuse_selected_steps": int(rollout_reuse_update_payload.get("selected_step_count") or 0),
                "rollout_reuse_ratio": _safe_float(rollout_reuse_update_payload.get("replay_ratio"), 0.0),
                "rollout_reuse_source_updates": int(rollout_reuse_update_payload.get("selected_source_update_count") or 0),
                "actor_refresh_interval_updates": int(actor_refresh_interval),
                "policy_lag_updates": int(policy_lag_updates),
                "learner_device": learner_device,
                "rollout_device": rollout_device,
                "gpu_mem_mb": gpu_mem_mb,
            },
        )
        append_unified_progress_event(
            unified_progress_path,
            build_progress_event(
                run_id=run_dir.name,
                component=component_name,
                phase="train",
                status=status,
                step=update_idx,
                epoch_or_iter=update_idx,
                seed=str(seed),
                metrics={
                    "mean_reward": mean_reward,
                    "mean_score": mean_score,
                    "policy_loss": _safe_float(update_metrics.get("policy_loss"), 0.0),
                    "value_loss": _safe_float(update_metrics.get("value_loss"), 0.0),
                    "entropy": _safe_float(update_metrics.get("entropy"), 0.0),
                    "kl_divergence": _safe_float(update_metrics.get("kl_divergence"), 0.0),
                    "invalid_action_rate": invalid_rate,
                    "rollout_steps": int(rollout_summary.get("step_count") or 0),
                    "rollout_steps_per_sec": rollout_steps_per_sec,
                    "learner_updates_per_sec": learner_updates_per_sec,
                    "buffer_backlog_steps": backlog_steps,
                    "weight_sync_count": weight_sync_count,
                    "oom_restart_count": oom_restart_count,
                    "hard_case_seed_count": int(len(rollout_seed_schedule)),
                    "hard_case_selected_failures": int(hard_case_plan.get("selected_failure_count") or 0),
                    "self_imitation_selected_episodes": int(self_imitation_update_payload.get("selected_episode_count") or 0),
                    "self_imitation_replay_ratio": _safe_float(self_imitation_update_payload.get("replay_ratio"), 0.0),
                    "rollout_reuse_selected_steps": int(rollout_reuse_update_payload.get("selected_step_count") or 0),
                    "rollout_reuse_ratio": _safe_float(rollout_reuse_update_payload.get("replay_ratio"), 0.0),
                    "rollout_reuse_source_updates": int(rollout_reuse_update_payload.get("selected_source_update_count") or 0),
                    "actor_refresh_interval_updates": int(actor_refresh_interval),
                    "policy_lag_updates": int(policy_lag_updates),
                },
                device_profile=runtime_profile,
                learner_device=learner_device,
                rollout_device=rollout_device,
                throughput=rollout_steps_per_sec,
                gpu_mem_mb=gpu_mem_mb,
                warning=(warnings[-1] if warnings else ""),
            ),
        )
        history_step_cap = max(
            max(1, int(applied_cfg.train.rollout_reuse_max_steps or 0)),
            min(len(steps), max(1, int(math.ceil(float(len(steps)) * max(float(applied_cfg.train.rollout_reuse_ratio), 0.0))))),
        )
        prior_rollout_history.append(
            {
                "update_index": int(update_idx),
                "steps": [dict(row) for row in steps[:history_step_cap] if isinstance(row, dict)],
            }
        )
        history_keep = max(0, int(applied_cfg.train.rollout_reuse_updates or 0))
        if history_keep > 0 and len(prior_rollout_history) > history_keep:
            prior_rollout_history = prior_rollout_history[-history_keep:]

    if warnings:
        with warnings_log_path.open("a", encoding="utf-8", newline="\n") as fp:
            for line in warnings:
                fp.write(line + "\n")

    metrics = {
        "schema": "p44_rl_seed_metrics_v1",
        "generated_at": _now_iso(),
        "seed": str(seed),
        "status": status,
        "updates_completed": len(rewards_hist),
        "mean_reward": float(statistics.mean(rewards_hist)) if rewards_hist else 0.0,
        "mean_score": float(statistics.mean(scores_hist)) if scores_hist else 0.0,
        "mean_invalid_action_rate": float(statistics.mean(invalid_hist)) if invalid_hist else 0.0,
        "policy_loss": float(statistics.mean(policy_loss_hist)) if policy_loss_hist else 0.0,
        "value_loss": float(statistics.mean(value_loss_hist)) if value_loss_hist else 0.0,
        "entropy": float(statistics.mean(entropy_hist)) if entropy_hist else 0.0,
        "kl_divergence": float(statistics.mean(kl_hist)) if kl_hist else 0.0,
        "rollout_steps_per_sec": float(statistics.mean(rollout_steps_per_sec_hist)) if rollout_steps_per_sec_hist else 0.0,
        "learner_updates_per_sec": float(statistics.mean(learner_updates_per_sec_hist)) if learner_updates_per_sec_hist else 0.0,
        "buffer_backlog_steps": float(statistics.mean(buffer_backlog_hist)) if buffer_backlog_hist else 0.0,
        "weight_sync_count": int(weight_sync_count),
        "oom_restart_count": int(oom_restart_count),
        "gpu_mem_mb": float(statistics.mean(gpu_mem_hist)) if gpu_mem_hist else None,
        "best_checkpoint": str(best_ckpt if best_ckpt.exists() else ""),
        "last_checkpoint": str(last_ckpt if last_ckpt.exists() else ""),
        "runtime_profile": runtime_profile,
        "hard_case_sampling_status": str(hard_case_plan.get("status") or "disabled"),
        "hard_case_selected_failures": int(hard_case_plan.get("selected_failure_count") or 0),
        "hard_case_seed_count": int(len(hard_case_plan.get("selected_failure_seeds") or [])),
        "hard_case_failure_type_count": int(hard_case_plan.get("failure_type_coverage") or 0),
        "hard_case_mean_replay_weight": _safe_float(hard_case_plan.get("mean_replay_weight"), 0.0),
        "self_imitation_selected_episodes": int(sum(self_imitation_episode_hist)),
        "self_imitation_replay_ratio": float(statistics.mean(self_imitation_ratio_hist)) if self_imitation_ratio_hist else 0.0,
        "rollout_reuse_selected_steps": int(sum(rollout_reuse_step_hist)),
        "rollout_reuse_replay_ratio": float(statistics.mean(rollout_reuse_ratio_hist)) if rollout_reuse_ratio_hist else 0.0,
        "rollout_reuse_source_updates": float(statistics.mean(rollout_reuse_source_update_hist)) if rollout_reuse_source_update_hist else 0.0,
        "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
        "mean_policy_lag_updates": float(statistics.mean(actor_policy_lag_hist)) if actor_policy_lag_hist else 0.0,
    }
    hard_case_schedule_path = seed_dir / "hard_case_schedule.json"
    _write_json(
        hard_case_schedule_path,
        {
            "schema": "p44_hard_case_schedule_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "hard_case_plan": hard_case_plan,
            "updates": hard_case_updates,
        },
    )
    self_imitation_path = seed_dir / "self_imitation_stats.json"
    _write_json(
        self_imitation_path,
        {
            "schema": "p44_self_imitation_stats_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "enabled": any(str(row.get("status") or "") == "ok" for row in self_imitation_updates),
            "configured_enabled": bool(cfg.self_imitation.enabled),
            "stage_min": str(cfg.self_imitation.stage_min or ""),
            "bucket_allowlist": list(cfg.self_imitation.bucket_allowlist or []),
            "slice_allowlist": list(cfg.self_imitation.slice_allowlist or []),
            "phase_allowlist": list(cfg.self_imitation.phase_allowlist or []),
            "action_type_allowlist": list(cfg.self_imitation.action_type_allowlist or []),
            "quality_threshold": float(cfg.self_imitation.quality_threshold),
            "updates": self_imitation_updates,
            "selected_episode_total": int(sum(self_imitation_episode_hist)),
            "mean_replay_ratio": float(statistics.mean(self_imitation_ratio_hist)) if self_imitation_ratio_hist else 0.0,
        },
    )
    rollout_reuse_path = seed_dir / "rollout_reuse_stats.json"
    _write_json(
        rollout_reuse_path,
        {
            "schema": "p44_rollout_reuse_stats_v1",
            "generated_at": _now_iso(),
            "seed": str(seed),
            "enabled": any(str(row.get("status") or "") == "ok" for row in rollout_reuse_updates),
            "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
            "rollout_reuse_ratio": float(cfg.train.rollout_reuse_ratio),
            "rollout_reuse_updates": int(cfg.train.rollout_reuse_updates),
            "rollout_reuse_max_steps": int(cfg.train.rollout_reuse_max_steps),
            "updates": rollout_reuse_updates,
            "selected_step_total": int(sum(rollout_reuse_step_hist)),
            "mean_replay_ratio": float(statistics.mean(rollout_reuse_ratio_hist)) if rollout_reuse_ratio_hist else 0.0,
            "mean_policy_lag_updates": float(statistics.mean(actor_policy_lag_hist)) if actor_policy_lag_hist else 0.0,
        },
    )
    _write_json(seed_dir / "metrics.json", metrics)
    return {
        "seed": str(seed),
        "status": status,
        "run_dir": str(seed_dir),
        "metrics": metrics,
        "best_checkpoint": str(best_ckpt if best_ckpt.exists() else ""),
        "last_checkpoint": str(last_ckpt if last_ckpt.exists() else ""),
        "rollout_buffers": rollout_buffers,
        "rollout_manifests": rollout_manifests,
        "runtime_profile": runtime_profile,
        "hard_case_schedule_json": str(hard_case_schedule_path),
        "self_imitation_stats_json": str(self_imitation_path),
        "rollout_reuse_stats_json": str(rollout_reuse_path),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def run_ppo_lite_training(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_repo_root()
    if isinstance(config, dict):
        cfg_raw = dict(config)
    elif config_path is not None:
        cfg_raw = _read_yaml_or_json(Path(config_path))
    else:
        cfg_raw = {}
    cfg = PPOConfig.from_mapping(cfg_raw)

    if quick:
        cfg.distributed.num_workers = min(cfg.distributed.num_workers, 2)
        cfg.distributed.episodes_per_worker = min(cfg.distributed.episodes_per_worker, 2)
        cfg.distributed.max_steps_per_episode = min(cfg.distributed.max_steps_per_episode, 80)
        cfg.rollout.total_steps_cap = min(cfg.rollout.total_steps_cap, 320)
        cfg.train.max_updates = min(cfg.train.max_updates, 2)
        cfg.train.ppo_epochs = min(cfg.train.ppo_epochs, 1)
        cfg.train.minibatch_size = min(cfg.train.minibatch_size, 64)
        cfg.evaluation.seeds = cfg.evaluation.seeds[:2]
        cfg.evaluation.episodes_per_seed = min(cfg.evaluation.episodes_per_seed, 1)
        cfg.evaluation.max_steps_per_episode = min(cfg.evaluation.max_steps_per_episode, 80)

    seeds = [str(s).strip() for s in (seeds_override if seeds_override else cfg.seeds) if str(s).strip()]
    if not seeds:
        seeds = ["AAAAAAA", "BBBBBBB"]
    if quick and len(seeds) > 2:
        seeds = seeds[:2]
    cfg.seeds = seeds
    cfg_raw = cfg.to_dict()
    cfg_raw["seeds"] = list(seeds)

    run_token = str(run_id or cfg.run_id or _now_stamp())
    run_dir = _resolve_path(repo_root, out_dir) if out_dir else (_resolve_path(repo_root, cfg.output_artifacts_root) / run_token)
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_profile_payload = load_runtime_profile(config=cfg_raw, component=_component_name_for_schema(cfg.schema)).to_dict()
    runtime_resolved = (
        runtime_profile_payload.get("resolved_profile", {}).get("resolved")
        if isinstance(runtime_profile_payload.get("resolved_profile"), dict)
        else {}
    )
    learner_device = str((runtime_resolved or {}).get("learner_device") or "cpu")
    rollout_device = str((runtime_resolved or {}).get("rollout_device") or "cpu")
    runtime_profile_json = run_dir / "runtime_profile.json"
    _write_json(runtime_profile_json, runtime_profile_payload)
    hard_case_plan = _resolve_hard_case_plan(cfg=cfg, repo_root=repo_root)
    hard_case_sampling_json = run_dir / "hard_case_sampling.json"
    _write_json(hard_case_sampling_json, hard_case_plan)

    seeds_payload = build_seeds_payload(seeds, seed_policy_version="p44.rl_ppo_lite")
    _write_json(run_dir / "seeds_used.json", seeds_payload)
    _write_json(run_dir / "reward_config.json", cfg.env.reward if isinstance(cfg.env.reward, dict) else {})
    progress_path = run_dir / "progress.jsonl"
    unified_progress_path = run_dir / "progress.unified.jsonl"
    warnings_log_path = run_dir / "warnings.log"

    scheduler = load_curriculum_scheduler(config_path=cfg.curriculum_config or None)
    curriculum_plan = scheduler.plan_payload(seeds=seeds)
    curriculum_plan_path = run_dir / "curriculum_plan.json"
    curriculum_applied_path = run_dir / "curriculum_applied.jsonl"
    _write_json(curriculum_plan_path, curriculum_plan)
    if not curriculum_applied_path.exists():
        curriculum_applied_path.write_text("", encoding="utf-8")

    evaluation_out_dir = _resolve_path(repo_root, cfg.evaluation.out_root) / run_token
    diagnostics_out_dir = _resolve_path(repo_root, cfg.diagnostics.out_root) / run_token

    if dry_run:
        manifest = {
            "schema": "p44_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "dry_run",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "runtime_profile": runtime_profile_payload,
            "hard_case_sampling": hard_case_plan,
            "curriculum_plan": curriculum_plan,
            "seed_results": [],
            "best_checkpoint": "",
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p44_rl_train_metrics_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "status": "dry_run",
                "seed_count": len(seeds),
                "ok_seed_count": 0,
                "learner_device": learner_device,
                "rollout_device": rollout_device,
                "hard_case_sampling_status": str(hard_case_plan.get("status") or "disabled"),
                "hard_case_selected_failures": int(hard_case_plan.get("selected_failure_count") or 0),
            },
        )
        (run_dir / "best_checkpoint.txt").write_text("\n", encoding="utf-8")
        append_unified_progress_event(
            unified_progress_path,
            build_progress_event(
                run_id=run_token,
                component=_component_name_for_schema(cfg.schema),
                phase="train",
                status="dry_run",
                device_profile=runtime_profile_payload,
                learner_device=learner_device,
                rollout_device=rollout_device,
            ),
        )
        return {
            "status": "dry_run",
            "run_id": run_token,
            "run_dir": str(run_dir),
            "train_manifest": str(run_dir / "train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": "",
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "warnings_log": str(warnings_log_path),
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
            "runtime_profile_json": str(runtime_profile_json),
            "progress_unified_jsonl": str(unified_progress_path),
        }

    try:
        _require_torch()
    except Exception as exc:
        stub_checkpoint = run_dir / "best_checkpoint_stub.json"
        _write_json(
            stub_checkpoint,
            {
                "schema": "p44_rl_checkpoint_stub_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "reason": "torch_missing",
                "note": "fallback artifact for non-torch environments",
                "config": cfg.to_dict(),
            },
        )
        with warnings_log_path.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(f"torch_missing:{exc}\n")
        manifest = {
            "schema": "p44_rl_train_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "status": "stub",
            "run_dir": str(run_dir),
            "config": cfg.to_dict(),
            "curriculum_plan": curriculum_plan,
            "seed_results": [],
            "best_checkpoint": str(stub_checkpoint),
            "reason": "torch_missing",
            "runtime_profile": runtime_profile_payload,
        }
        write_manifest(run_dir / "train_manifest.json", manifest)
        _write_json(
            run_dir / "metrics.json",
            {
                "schema": "p44_rl_train_metrics_v1",
                "generated_at": _now_iso(),
                "run_id": run_token,
                "status": "stub",
                "seed_count": len(seeds),
                "ok_seed_count": 0,
                "reason": "torch_missing",
                "candidate_checkpoint": str(stub_checkpoint),
                "learner_device": learner_device,
                "rollout_device": rollout_device,
            },
        )
        (run_dir / "best_checkpoint.txt").write_text(str(stub_checkpoint) + "\n", encoding="utf-8")
        append_unified_progress_event(
            unified_progress_path,
            build_progress_event(
                run_id=run_token,
                component=_component_name_for_schema(cfg.schema),
                phase="train",
                status="stub",
                warning="torch_missing",
                device_profile=runtime_profile_payload,
                learner_device=learner_device,
                rollout_device=rollout_device,
            ),
        )
        return {
            "status": "stub",
            "run_id": run_token,
            "run_dir": str(run_dir),
            "train_manifest": str(run_dir / "train_manifest.json"),
            "metrics": str(run_dir / "metrics.json"),
            "best_checkpoint": str(stub_checkpoint),
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "hard_case_sampling_json": str(hard_case_sampling_json),
            "warnings_log": str(warnings_log_path),
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
            "runtime_profile_json": str(runtime_profile_json),
            "progress_unified_jsonl": str(unified_progress_path),
        }

    seed_results: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds, start=1):
        seed_result = _train_one_seed(
            seed=seed,
            seed_index=idx,
            seed_total=len(seeds),
            cfg=cfg,
            cfg_raw=cfg_raw,
            scheduler=scheduler,
            run_dir=run_dir,
            progress_path=progress_path,
            unified_progress_path=unified_progress_path,
            warnings_log_path=warnings_log_path,
            curriculum_applied_path=curriculum_applied_path,
            runtime_profile=runtime_profile_payload,
            hard_case_plan=hard_case_plan,
        )
        seed_results.append(seed_result)

    self_imitation_seed_stats: list[dict[str, Any]] = []
    for row in seed_results:
        stats_path = Path(str(row.get("self_imitation_stats_json") or ""))
        payload = _read_json_payload(stats_path) if str(stats_path).strip() else None
        if isinstance(payload, dict):
            self_imitation_seed_stats.append(payload)
    rollout_reuse_seed_stats: list[dict[str, Any]] = []
    for row in seed_results:
        stats_path = Path(str(row.get("rollout_reuse_stats_json") or ""))
        payload = _read_json_payload(stats_path) if str(stats_path).strip() else None
        if isinstance(payload, dict):
            rollout_reuse_seed_stats.append(payload)

    reward_schedule_payload = {
        "schema": "p44_reward_schedule_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "enabled": bool(curriculum_plan.get("enabled")),
        "phase_count": int(curriculum_plan.get("phase_count") or 0),
        "stages": [
            {
                "stage": str(phase.get("stage") or ""),
                "phase_index": int(phase.get("phase_index") or 0),
                "reward": dict(phase.get("reward") or {}) if isinstance(phase.get("reward"), dict) else {},
                "hard_case_sampling": dict(phase.get("hard_case_sampling") or {}) if isinstance(phase.get("hard_case_sampling"), dict) else {},
                "self_imitation": dict(phase.get("self_imitation") or {}) if isinstance(phase.get("self_imitation"), dict) else {},
            }
            for phase in (curriculum_plan.get("phases") if isinstance(curriculum_plan.get("phases"), list) else [])
            if isinstance(phase, dict)
        ],
    }
    reward_schedule_json = run_dir / "reward_schedule.json"
    _write_json(reward_schedule_json, reward_schedule_payload)

    hard_case_manifest_json = run_dir / "hardcase_manifest.json"
    hard_case_stats_json = run_dir / "hardcase_stats.json"
    hard_case_stats_md = run_dir / "hardcase_stats.md"
    bucket_replay_manifest_json = run_dir / "bucket_replay_manifest.json"
    bucket_replay_stats_json = run_dir / "bucket_replay_stats.json"
    bucket_replay_stats_md = run_dir / "bucket_replay_stats.md"
    hard_case_stats_payload = {
        "schema": "p44_hardcase_stats_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": str(hard_case_plan.get("status") or "disabled"),
        "selected_failure_count": int(hard_case_plan.get("selected_failure_count") or 0),
        "selected_failure_seeds": list(hard_case_plan.get("selected_failure_seeds") or []),
        "seed_replay_counts": dict(hard_case_plan.get("seed_replay_counts") or {}),
        "failure_type_counts": dict(hard_case_plan.get("failure_type_counts") or {}),
        "failure_bucket_counts": dict(hard_case_plan.get("failure_bucket_counts") or {}),
        "failure_bucket_ratios": dict(hard_case_plan.get("bucket_selected_ratios") or {}),
        "bucket_sampling_weights": dict(hard_case_plan.get("bucket_sampling_weights") or {}),
        "bucket_quota_caps": dict(hard_case_plan.get("bucket_quota_caps") or {}),
        "bucket_minimum_counts": dict(hard_case_plan.get("bucket_minimum_counts") or {}),
        "source_type_counts": dict(hard_case_plan.get("source_type_selected_counts") or hard_case_plan.get("source_type_counts") or {}),
        "source_type_ratios": dict(hard_case_plan.get("source_type_selected_ratios") or {}),
        "source_type_sampling_weights": dict(hard_case_plan.get("source_type_sampling_weights") or {}),
        "source_type_minimum_counts": dict(hard_case_plan.get("source_type_minimum_counts") or {}),
        "source_type_quota_caps": dict(hard_case_plan.get("source_type_quota_caps") or {}),
        "source_variant_counts": dict(hard_case_plan.get("source_variant_selected_counts") or hard_case_plan.get("source_variant_counts") or {}),
        "source_variant_ratios": dict(hard_case_plan.get("source_variant_selected_ratios") or {}),
        "source_variant_sampling_weights": dict(hard_case_plan.get("source_variant_sampling_weights") or {}),
        "source_variant_minimum_counts": dict(hard_case_plan.get("source_variant_minimum_counts") or {}),
        "source_variant_quota_caps": dict(hard_case_plan.get("source_variant_quota_caps") or {}),
        "slice_tag_counts": dict(hard_case_plan.get("slice_tag_counts") or {}),
        "slice_minimum_counts": dict(hard_case_plan.get("slice_minimum_counts") or {}),
        "risk_tag_counts": dict(hard_case_plan.get("risk_tag_counts") or {}),
        "mean_replay_weight": _safe_float(hard_case_plan.get("mean_replay_weight"), 0.0),
        "failure_type_coverage": int(hard_case_plan.get("failure_type_coverage") or 0),
        "failure_bucket_coverage": int(hard_case_plan.get("failure_bucket_coverage") or 0),
        "scarce_failure_buckets": list(hard_case_plan.get("scarce_failure_buckets") or []),
    }
    _write_json(
        hard_case_manifest_json,
        {
            "schema": "p44_hardcase_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "failure_pack_manifest": str(hard_case_plan.get("failure_pack_manifest") or ""),
            "selected_failures_preview": list(hard_case_plan.get("selected_failures_preview") or []),
            "hard_case_plan": hard_case_plan,
        },
    )
    _write_json(hard_case_stats_json, hard_case_stats_payload)
    _write_json(
        bucket_replay_manifest_json,
        {
            "schema": "p44_bucket_replay_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "selected_failures_preview": list(hard_case_plan.get("selected_failures_preview") or []),
            "hard_case_plan": hard_case_plan,
        },
    )
    _write_json(bucket_replay_stats_json, hard_case_stats_payload)
    _write_markdown(
        hard_case_stats_md,
        [
            f"# Hard-case Stats ({run_token})",
            "",
            f"- status: `{hard_case_stats_payload['status']}`",
            f"- selected_failure_count: `{hard_case_stats_payload['selected_failure_count']}`",
            f"- failure_type_coverage: `{hard_case_stats_payload['failure_type_coverage']}`",
            f"- mean_replay_weight: `{hard_case_stats_payload['mean_replay_weight']}`",
            "",
            "## Failure Buckets",
            *(
                [f"- {k}: {v}" for k, v in sorted(hard_case_stats_payload["failure_bucket_counts"].items(), key=lambda item: (-item[1], item[0]))]
                if hard_case_stats_payload["failure_bucket_counts"]
                else ["- none"]
            ),
        ],
    )
    _write_markdown(
        bucket_replay_stats_md,
        [
            f"# Bucket-aware Replay Stats ({run_token})",
            "",
            f"- status: `{hard_case_stats_payload['status']}`",
            f"- selected_failure_count: `{hard_case_stats_payload['selected_failure_count']}`",
            f"- failure_bucket_coverage: `{hard_case_stats_payload['failure_bucket_coverage']}`",
            f"- scarce_failure_buckets: `{', '.join(hard_case_stats_payload['scarce_failure_buckets'])}`",
            "",
            "## Bucket Ratios",
            *(
                [
                    f"- {k}: count=`{hard_case_stats_payload['failure_bucket_counts'].get(k)}` ratio=`{hard_case_stats_payload['failure_bucket_ratios'].get(k)}` weight=`{hard_case_stats_payload['bucket_sampling_weights'].get(k, 1.0)}`"
                    for k in sorted(hard_case_stats_payload["failure_bucket_counts"])
                ]
                if hard_case_stats_payload["failure_bucket_counts"]
                else ["- none"]
            ),
        ],
    )

    best_trajectory_manifest_json = run_dir / "best_trajectory_manifest.json"
    best_trajectory_stats_json = run_dir / "best_trajectory_stats.json"
    best_trajectory_stats_md = run_dir / "best_trajectory_stats.md"
    best_episode_rows = [
        episode
        for seed_payload in self_imitation_seed_stats
        for episode in (seed_payload.get("updates") if isinstance(seed_payload.get("updates"), list) else [])
        if isinstance(episode, dict)
    ]
    best_trajectory_stats_payload = {
        "schema": "p44_best_trajectory_stats_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "enabled": any(bool(row.get("enabled")) for row in self_imitation_seed_stats),
        "configured_enabled": bool(cfg.self_imitation.enabled),
        "selected_episode_total": int(sum(int(row.get("selected_episode_count") or 0) for row in best_episode_rows)),
        "selected_step_total": int(sum(int(row.get("selected_step_count") or 0) for row in best_episode_rows)),
        "mean_replay_ratio": (
            float(statistics.mean([_safe_float(row.get("replay_ratio"), 0.0) for row in best_episode_rows]))
            if best_episode_rows
            else 0.0
        ),
        "stage_min": next((str(row.get("stage_min") or "") for row in self_imitation_seed_stats if str(row.get("stage_min") or "").strip()), str(cfg.self_imitation.stage_min or "")),
        "bucket_allowlist": next((list(row.get("bucket_allowlist") or []) for row in self_imitation_seed_stats if isinstance(row.get("bucket_allowlist"), list) and row.get("bucket_allowlist")), list(cfg.self_imitation.bucket_allowlist or [])),
        "slice_allowlist": next((list(row.get("slice_allowlist") or []) for row in self_imitation_seed_stats if isinstance(row.get("slice_allowlist"), list) and row.get("slice_allowlist")), list(cfg.self_imitation.slice_allowlist or [])),
        "phase_allowlist": next((list(row.get("phase_allowlist") or []) for row in self_imitation_seed_stats if isinstance(row.get("phase_allowlist"), list) and row.get("phase_allowlist")), list(cfg.self_imitation.phase_allowlist or [])),
        "action_type_allowlist": next((list(row.get("action_type_allowlist") or []) for row in self_imitation_seed_stats if isinstance(row.get("action_type_allowlist"), list) and row.get("action_type_allowlist")), list(cfg.self_imitation.action_type_allowlist or [])),
        "quality_threshold": next((_safe_float(row.get("quality_threshold"), 0.0) for row in self_imitation_seed_stats if _safe_float(row.get("quality_threshold"), 0.0) > 0.0), float(cfg.self_imitation.quality_threshold)),
    }
    _write_json(
        best_trajectory_manifest_json,
        {
            "schema": "p44_best_trajectory_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "self_imitation_seed_stats": [str(row.get("seed") or "") for row in self_imitation_seed_stats],
            "updates_preview": best_episode_rows[:16],
        },
    )
    _write_json(best_trajectory_stats_json, best_trajectory_stats_payload)
    _write_markdown(
        best_trajectory_stats_md,
        [
            f"# Best Trajectory Replay ({run_token})",
            "",
            f"- enabled: `{cfg.self_imitation.enabled}`",
            f"- selected_episode_total: `{best_trajectory_stats_payload['selected_episode_total']}`",
            f"- selected_step_total: `{best_trajectory_stats_payload['selected_step_total']}`",
            f"- mean_replay_ratio: `{best_trajectory_stats_payload['mean_replay_ratio']}`",
        ],
    )
    rollout_reuse_manifest_json = run_dir / "rollout_reuse_manifest.json"
    rollout_reuse_stats_json = run_dir / "rollout_reuse_stats.json"
    rollout_reuse_stats_md = run_dir / "rollout_reuse_stats.md"
    rollout_reuse_stats_payload = {
        "schema": "p44_rollout_reuse_run_stats_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "enabled": any(bool(row.get("enabled")) for row in rollout_reuse_seed_stats),
        "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
        "rollout_reuse_ratio": float(cfg.train.rollout_reuse_ratio),
        "rollout_reuse_updates": int(cfg.train.rollout_reuse_updates),
        "rollout_reuse_max_steps": int(cfg.train.rollout_reuse_max_steps),
        "selected_step_total": int(sum(int(row.get("selected_step_total") or 0) for row in rollout_reuse_seed_stats)),
        "mean_replay_ratio": (
            float(statistics.mean([_safe_float(row.get("mean_replay_ratio"), 0.0) for row in rollout_reuse_seed_stats]))
            if rollout_reuse_seed_stats
            else 0.0
        ),
        "mean_policy_lag_updates": (
            float(statistics.mean([_safe_float(row.get("mean_policy_lag_updates"), 0.0) for row in rollout_reuse_seed_stats]))
            if rollout_reuse_seed_stats
            else 0.0
        ),
        "seed_stats": rollout_reuse_seed_stats,
    }
    _write_json(
        rollout_reuse_manifest_json,
        {
            "schema": "p44_rollout_reuse_manifest_v1",
            "generated_at": _now_iso(),
            "run_id": run_token,
            "seed_stats": [str(row.get("seed") or "") for row in rollout_reuse_seed_stats],
        },
    )
    _write_json(rollout_reuse_stats_json, rollout_reuse_stats_payload)
    _write_markdown(
        rollout_reuse_stats_md,
        [
            f"# Rollout Reuse Stats ({run_token})",
            "",
            f"- enabled: `{rollout_reuse_stats_payload['enabled']}`",
            f"- actor_refresh_interval_updates: `{rollout_reuse_stats_payload['actor_refresh_interval_updates']}`",
            f"- rollout_reuse_ratio: `{rollout_reuse_stats_payload['rollout_reuse_ratio']}`",
            f"- rollout_reuse_updates: `{rollout_reuse_stats_payload['rollout_reuse_updates']}`",
            f"- rollout_reuse_max_steps: `{rollout_reuse_stats_payload['rollout_reuse_max_steps']}`",
            f"- selected_step_total: `{rollout_reuse_stats_payload['selected_step_total']}`",
            f"- mean_replay_ratio: `{rollout_reuse_stats_payload['mean_replay_ratio']}`",
            f"- mean_policy_lag_updates: `{rollout_reuse_stats_payload['mean_policy_lag_updates']}`",
        ],
    )

    ok_rows = [row for row in seed_results if str(row.get("status")) == "ok"]
    checkpoint_candidates = [
        str(row.get("best_checkpoint") or "")
        for row in ok_rows
        if str(row.get("best_checkpoint") or "").strip()
    ]
    if checkpoint_candidates:
        eval_summary = run_multi_seed_evaluation(
            checkpoint_paths=checkpoint_candidates,
            seeds=cfg.evaluation.seeds,
            episodes_per_seed=int(cfg.evaluation.episodes_per_seed),
            max_steps_per_episode=int(cfg.evaluation.max_steps_per_episode),
            backend=str(cfg.env.backend),
            reward_config=cfg.env.reward,
            env_config={
                "timeout_sec": float(cfg.env.timeout_sec),
                "max_steps_per_episode": int(cfg.env.max_steps_per_episode),
                "max_auto_steps": int(cfg.env.max_auto_steps),
                "max_ante": int(cfg.env.max_ante),
                "auto_advance": bool(cfg.env.auto_advance),
            },
            greedy=bool(cfg.evaluation.greedy),
            run_id=run_token,
            out_dir=evaluation_out_dir,
        )
    else:
        eval_summary = {
            "status": "stub",
            "run_id": run_token,
            "run_dir": str(evaluation_out_dir),
            "seed_results_json": "",
            "best_checkpoint": "",
            "checkpoint_results": [],
        }

    best_checkpoint = str(eval_summary.get("best_checkpoint") or "")
    if not best_checkpoint and checkpoint_candidates:
        best_checkpoint = checkpoint_candidates[0]
    status = "ok" if ok_rows else "stub"
    selected_eval_row = (
        (eval_summary.get("checkpoint_results") or [])[0]
        if isinstance(eval_summary.get("checkpoint_results"), list) and (eval_summary.get("checkpoint_results") or [])
        else {}
    )

    rollout_buffers = [
        str(path)
        for row in seed_results
        for path in (row.get("rollout_buffers") or [])
        if str(path).strip()
    ]
    diagnostics_summary = run_diagnostics(
        progress_jsonl=progress_path,
        rollout_buffers=rollout_buffers,
        out_dir=diagnostics_out_dir,
        action_topk=int(cfg.diagnostics.action_topk),
    )

    metrics_payload = {
        "schema": "p44_rl_train_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "seed_count": len(seeds),
        "ok_seed_count": len(ok_rows),
        "mean_reward": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("mean_reward"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "mean_score": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("mean_score"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "policy_loss": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("policy_loss"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "value_loss": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("value_loss"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "entropy": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("entropy"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "kl_divergence": (
            float(statistics.mean([_safe_float((row.get("metrics") or {}).get("kl_divergence"), 0.0) for row in ok_rows]))
            if ok_rows
            else 0.0
        ),
        "invalid_action_rate": (
            float(
                statistics.mean(
                    [_safe_float((row.get("metrics") or {}).get("mean_invalid_action_rate"), 0.0) for row in ok_rows]
                )
            )
            if ok_rows
            else 0.0
        ),
        "eval_mean_score": _safe_float(selected_eval_row.get("mean_score"), 0.0),
        "eval_std_score": _safe_float(selected_eval_row.get("std_score"), 0.0),
        "candidate_checkpoint": best_checkpoint,
        "hard_case_sampling_status": str(hard_case_plan.get("status") or "disabled"),
        "hard_case_selected_failures": int(hard_case_plan.get("selected_failure_count") or 0),
        "hard_case_seed_count": int(len(hard_case_plan.get("selected_failure_seeds") or [])),
        "hard_case_failure_type_count": int(hard_case_plan.get("failure_type_coverage") or 0),
        "hard_case_mean_replay_weight": _safe_float(hard_case_plan.get("mean_replay_weight"), 0.0),
        "self_imitation_selected_episodes": int(best_trajectory_stats_payload.get("selected_episode_total") or 0),
        "self_imitation_replay_ratio": _safe_float(best_trajectory_stats_payload.get("mean_replay_ratio"), 0.0),
        "self_imitation_stage_min": str(best_trajectory_stats_payload.get("stage_min") or ""),
        "self_imitation_quality_threshold": _safe_float(best_trajectory_stats_payload.get("quality_threshold"), 0.0),
        "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
        "rollout_reuse_ratio": float(cfg.train.rollout_reuse_ratio),
        "rollout_reuse_updates": int(cfg.train.rollout_reuse_updates),
        "rollout_reuse_max_steps": int(cfg.train.rollout_reuse_max_steps),
        "rollout_reuse_selected_steps": int(
            sum(int(row.get("selected_step_total") or 0) for row in rollout_reuse_seed_stats)
        ),
        "rollout_reuse_replay_ratio": (
            float(statistics.mean([_safe_float(row.get("mean_replay_ratio"), 0.0) for row in rollout_reuse_seed_stats]))
            if rollout_reuse_seed_stats
            else 0.0
        ),
        "mean_policy_lag_updates": (
            float(statistics.mean([_safe_float(row.get("mean_policy_lag_updates"), 0.0) for row in rollout_reuse_seed_stats]))
            if rollout_reuse_seed_stats
            else 0.0
        ),
        "certification_status": "fast_pass",
    }
    bucket_target_weights = {
        str(bucket): max(0.0, _safe_float(weight, 0.0))
        for bucket, weight in (hard_case_plan.get("bucket_sampling_weights") or {}).items()
        if str(bucket).strip()
    }
    slice_target_weights = {
        str(tag): max(0.0, _safe_float(weight, 0.0))
        for tag, weight in (hard_case_plan.get("slice_sampling_weights") or {}).items()
        if str(tag).strip()
    }
    risk_target_weights = {
        str(tag): max(0.0, _safe_float(weight, 0.0))
        for tag, weight in (hard_case_plan.get("risk_tag_sampling_weights") or {}).items()
        if str(tag).strip()
    }
    source_variant_target_weights = {
        str(tag): max(0.0, _safe_float(weight, 0.0))
        for tag, weight in (hard_case_plan.get("source_variant_sampling_weights") or {}).items()
        if str(tag).strip()
    }
    bucket_mix_delta_vs_target: dict[str, float] = {}
    bucket_ratios = dict(hard_case_plan.get("bucket_selected_ratios") or {})
    if bucket_target_weights:
        weight_sum = sum(bucket_target_weights.values())
        normalized_targets = {
            bucket: round(_safe_ratio(weight, weight_sum), 6) for bucket, weight in bucket_target_weights.items()
        } if weight_sum > 0.0 else {}
        for bucket in sorted(set(bucket_ratios) | set(normalized_targets)):
            bucket_mix_delta_vs_target[bucket] = round(
                _safe_float(bucket_ratios.get(bucket), 0.0) - _safe_float(normalized_targets.get(bucket), 0.0),
                6,
            )
    slice_mix_delta_vs_target: dict[str, float] = {}
    slice_ratios = dict(hard_case_plan.get("slice_selected_ratios") or {})
    if slice_target_weights:
        weight_sum = sum(slice_target_weights.values())
        normalized_targets = {
            tag: round(_safe_ratio(weight, weight_sum), 6) for tag, weight in slice_target_weights.items()
        } if weight_sum > 0.0 else {}
        for tag in sorted(set(slice_ratios) | set(normalized_targets)):
            slice_mix_delta_vs_target[tag] = round(
                _safe_float(slice_ratios.get(tag), 0.0) - _safe_float(normalized_targets.get(tag), 0.0),
                6,
            )
    risk_mix_delta_vs_target: dict[str, float] = {}
    risk_ratios = dict(hard_case_plan.get("risk_selected_ratios") or {})
    if risk_target_weights:
        weight_sum = sum(risk_target_weights.values())
        normalized_targets = {
            tag: round(_safe_ratio(weight, weight_sum), 6) for tag, weight in risk_target_weights.items()
        } if weight_sum > 0.0 else {}
        for tag in sorted(set(risk_ratios) | set(normalized_targets)):
            risk_mix_delta_vs_target[tag] = round(
                _safe_float(risk_ratios.get(tag), 0.0) - _safe_float(normalized_targets.get(tag), 0.0),
                6,
            )
    source_variant_mix_delta_vs_target: dict[str, float] = {}
    source_variant_ratios = dict(hard_case_plan.get("source_variant_selected_ratios") or {})
    if source_variant_target_weights:
        weight_sum = sum(source_variant_target_weights.values())
        normalized_targets = {
            tag: round(_safe_ratio(weight, weight_sum), 6) for tag, weight in source_variant_target_weights.items()
        } if weight_sum > 0.0 else {}
        for tag in sorted(set(source_variant_ratios) | set(normalized_targets)):
            source_variant_mix_delta_vs_target[tag] = round(
                _safe_float(source_variant_ratios.get(tag), 0.0) - _safe_float(normalized_targets.get(tag), 0.0),
                6,
            )
    bucket_metrics_json = run_dir / "bucket_metrics.json"
    slice_metrics_json = run_dir / "slice_metrics.json"
    training_summary_md = run_dir / "training_summary.md"
    bucket_metrics_payload = {
        "schema": "p44_bucket_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "failure_bucket_counts": dict(hard_case_plan.get("failure_bucket_counts") or {}),
        "failure_bucket_ratios": bucket_ratios,
        "bucket_sampling_weights": bucket_target_weights,
        "bucket_quota_caps": dict(hard_case_plan.get("bucket_quota_caps") or {}),
        "bucket_minimum_counts": dict(hard_case_plan.get("bucket_minimum_counts") or {}),
        "bucket_mix_delta_vs_target": bucket_mix_delta_vs_target,
        "source_type_counts": dict(hard_case_plan.get("source_type_selected_counts") or hard_case_plan.get("source_type_counts") or {}),
        "source_type_sampling_weights": dict(hard_case_plan.get("source_type_sampling_weights") or {}),
        "source_type_minimum_counts": dict(hard_case_plan.get("source_type_minimum_counts") or {}),
        "source_type_quota_caps": dict(hard_case_plan.get("source_type_quota_caps") or {}),
        "source_variant_counts": dict(hard_case_plan.get("source_variant_selected_counts") or hard_case_plan.get("source_variant_counts") or {}),
        "source_variant_ratios": source_variant_ratios,
        "source_variant_sampling_weights": source_variant_target_weights,
        "source_variant_minimum_counts": dict(hard_case_plan.get("source_variant_minimum_counts") or {}),
        "source_variant_quota_caps": dict(hard_case_plan.get("source_variant_quota_caps") or {}),
        "source_variant_mix_delta_vs_target": source_variant_mix_delta_vs_target,
        "slice_sampling_weights": dict(hard_case_plan.get("slice_sampling_weights") or {}),
        "slice_minimum_counts": dict(hard_case_plan.get("slice_minimum_counts") or {}),
        "slice_quota_caps": dict(hard_case_plan.get("slice_quota_caps") or {}),
        "slice_selected_counts": dict(hard_case_plan.get("slice_selected_counts") or {}),
        "slice_selected_ratios": slice_ratios,
        "slice_mix_delta_vs_target": slice_mix_delta_vs_target,
        "risk_tag_sampling_weights": dict(hard_case_plan.get("risk_tag_sampling_weights") or {}),
        "risk_tag_quota_caps": dict(hard_case_plan.get("risk_tag_quota_caps") or {}),
        "risk_selected_counts": dict(hard_case_plan.get("risk_selected_counts") or {}),
        "risk_selected_ratios": risk_ratios,
        "risk_mix_delta_vs_target": risk_mix_delta_vs_target,
        "scarce_failure_buckets": list(hard_case_plan.get("scarce_failure_buckets") or []),
        "failure_bucket_coverage": int(hard_case_plan.get("failure_bucket_coverage") or 0),
        "known_failure_buckets": list(hard_case_plan.get("known_failure_buckets") or KNOWN_FAILURE_BUCKETS),
        "self_imitation_stage_min": str(cfg.self_imitation.stage_min or ""),
        "actor_refresh_interval_updates": int(cfg.train.actor_refresh_interval_updates),
        "rollout_reuse_ratio": float(cfg.train.rollout_reuse_ratio),
        "rollout_reuse_updates": int(cfg.train.rollout_reuse_updates),
        "rollout_reuse_max_steps": int(cfg.train.rollout_reuse_max_steps),
        "certification_status": "fast_pass",
    }
    slice_metrics_payload = {
        "schema": "p44_slice_metrics_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "slice_tag_counts": dict(hard_case_plan.get("slice_tag_counts") or {}),
        "slice_tag_ratios": _counter_ratios(hard_case_plan.get("slice_tag_counts") or {}),
        "risk_tag_counts": dict(hard_case_plan.get("risk_tag_counts") or {}),
        "risk_tag_ratios": _counter_ratios(hard_case_plan.get("risk_tag_counts") or {}),
        "source_type_counts": dict(hard_case_plan.get("source_type_selected_counts") or hard_case_plan.get("source_type_counts") or {}),
        "source_type_minimum_counts": dict(hard_case_plan.get("source_type_minimum_counts") or {}),
        "source_variant_counts": dict(hard_case_plan.get("source_variant_selected_counts") or hard_case_plan.get("source_variant_counts") or {}),
        "source_variant_ratios": source_variant_ratios,
        "source_variant_sampling_weights": source_variant_target_weights,
        "source_variant_minimum_counts": dict(hard_case_plan.get("source_variant_minimum_counts") or {}),
        "source_variant_quota_caps": dict(hard_case_plan.get("source_variant_quota_caps") or {}),
        "source_variant_mix_delta_vs_target": source_variant_mix_delta_vs_target,
        "slice_sampling_weights": dict(hard_case_plan.get("slice_sampling_weights") or {}),
        "slice_selected_counts": dict(hard_case_plan.get("slice_selected_counts") or {}),
        "slice_selected_ratios": slice_ratios,
        "slice_mix_delta_vs_target": slice_mix_delta_vs_target,
        "risk_tag_sampling_weights": dict(hard_case_plan.get("risk_tag_sampling_weights") or {}),
        "risk_selected_counts": dict(hard_case_plan.get("risk_selected_counts") or {}),
        "risk_selected_ratios": risk_ratios,
        "risk_mix_delta_vs_target": risk_mix_delta_vs_target,
        "curriculum_phase_count": int(curriculum_plan.get("phase_count") or 0),
        "certification_status": "fast_pass",
    }
    _write_json(bucket_metrics_json, bucket_metrics_payload)
    _write_json(slice_metrics_json, slice_metrics_payload)
    manifest = {
        "schema": "p44_rl_train_manifest_v1",
        "generated_at": _now_iso(),
        "run_id": run_token,
        "status": status,
        "run_dir": str(run_dir),
        "config": cfg.to_dict(),
        "hard_case_sampling": hard_case_plan,
        "curriculum_plan": curriculum_plan,
        "paths": {
            "metrics": str(run_dir / "metrics.json"),
            "progress_jsonl": str(progress_path),
            "warnings_log": str(warnings_log_path),
            "seeds_used": str(run_dir / "seeds_used.json"),
            "reward_config": str(run_dir / "reward_config.json"),
            "hard_case_sampling_json": str(hard_case_sampling_json),
            "hardcase_manifest_json": str(hard_case_manifest_json),
            "hardcase_stats_json": str(hard_case_stats_json),
            "bucket_replay_manifest_json": str(bucket_replay_manifest_json),
            "bucket_replay_stats_json": str(bucket_replay_stats_json),
            "best_trajectory_manifest_json": str(best_trajectory_manifest_json),
            "best_trajectory_stats_json": str(best_trajectory_stats_json),
            "rollout_reuse_manifest_json": str(rollout_reuse_manifest_json),
            "rollout_reuse_stats_json": str(rollout_reuse_stats_json),
            "curriculum_plan": str(curriculum_plan_path),
            "curriculum_applied": str(curriculum_applied_path),
            "reward_schedule_json": str(reward_schedule_json),
            "bucket_metrics_json": str(bucket_metrics_json),
            "slice_metrics_json": str(slice_metrics_json),
            "training_summary_md": str(training_summary_md),
            "multi_seed_eval": str(eval_summary.get("seed_results_json") or ""),
            "diagnostics_json": str(diagnostics_summary.get("diagnostics_json") or ""),
            "diagnostics_report_md": str(diagnostics_summary.get("diagnostics_report_md") or ""),
        },
        "seed_results": seed_results,
        "best_checkpoint": best_checkpoint,
        "multi_seed_eval": eval_summary,
        "diagnostics": diagnostics_summary,
        "runtime_profile": runtime_profile_payload,
    }
    checkpoint_registry_entry: dict[str, Any] = {}
    if str(best_checkpoint).strip():
        checkpoint_registry_entry = register_checkpoint(
            {
                "family": "rl_policy",
                "training_mode": ("p42_rl_candidate" if str(cfg.schema).startswith("p42_") else "p44_distributed_rl"),
                "training_mode_category": "mainline",
                "source_run_id": run_token,
                "source_experiment_id": (Path(config_path).stem if config_path is not None else str(cfg.schema or "")),
                "seed_or_seed_group": ",".join(seeds[:4]) + (f"+{max(0, len(seeds) - 4)}" if len(seeds) > 4 else ""),
                "device_profile": _runtime_profile_name(runtime_profile_payload),
                "training_python": sys.executable,
                "artifact_path": best_checkpoint,
                "status": "draft",
                "metrics_ref": str((run_dir / "metrics.json").resolve()),
                "lineage_refs": {
                    "manifest_path": str((run_dir / "train_manifest.json").resolve()),
                    "seeds_used_json": str((run_dir / "seeds_used.json").resolve()),
                    "runtime_profile_json": str(runtime_profile_json.resolve()),
                    "eval_seed_results": str(eval_summary.get("seed_results_json") or ""),
                    "diagnostics_json": str(diagnostics_summary.get("diagnostics_json") or ""),
                    "progress_unified_jsonl": str(unified_progress_path.resolve()),
                    "hardcase_stats_json": str(hard_case_stats_json.resolve()),
                    "bucket_metrics_json": str(bucket_metrics_json.resolve()),
                    "slice_metrics_json": str(slice_metrics_json.resolve()),
                    "best_trajectory_stats_json": str(best_trajectory_stats_json.resolve()),
                    "rollout_reuse_stats_json": str(rollout_reuse_stats_json.resolve()),
                    "reward_schedule_json": str(reward_schedule_json.resolve()),
                },
                "curriculum_profile": str(curriculum_plan_path.resolve()),
                "git_commit": _git_commit(repo_root),
                "notes": "auto_registered_from_p44_ppo_lite",
            }
        )
        metrics_payload["checkpoint_id"] = str(checkpoint_registry_entry.get("checkpoint_id") or "")
        manifest["checkpoint_registry"] = {
            "checkpoint_id": str(checkpoint_registry_entry.get("checkpoint_id") or ""),
            "registry_path": str((repo_root / "docs/artifacts/registry/checkpoints_registry.json").resolve()),
        }
    _write_json(run_dir / "metrics.json", metrics_payload)
    write_manifest(run_dir / "train_manifest.json", manifest)
    _write_markdown(
        run_dir / "summary.md",
        [
            f"# PPO Training Summary ({run_token})",
            "",
            f"- status: `{status}`",
            f"- candidate_checkpoint: `{best_checkpoint}`",
            f"- mean_reward: `{metrics_payload.get('mean_reward')}`",
            f"- mean_score: `{metrics_payload.get('mean_score')}`",
            f"- invalid_action_rate: `{metrics_payload.get('invalid_action_rate')}`",
            f"- hard_case_sampling_status: `{metrics_payload.get('hard_case_sampling_status')}`",
            f"- hard_case_selected_failures: `{metrics_payload.get('hard_case_selected_failures')}`",
            f"- self_imitation_selected_episodes: `{metrics_payload.get('self_imitation_selected_episodes')}`",
            f"- self_imitation_replay_ratio: `{metrics_payload.get('self_imitation_replay_ratio')}`",
            f"- actor_refresh_interval_updates: `{metrics_payload.get('actor_refresh_interval_updates')}`",
            f"- rollout_reuse_ratio: `{metrics_payload.get('rollout_reuse_ratio')}`",
            f"- rollout_reuse_selected_steps: `{metrics_payload.get('rollout_reuse_selected_steps')}`",
            f"- mean_policy_lag_updates: `{metrics_payload.get('mean_policy_lag_updates')}`",
            f"- certification_status: `{metrics_payload.get('certification_status')}`",
        ],
    )
    _write_markdown(
        training_summary_md,
        [
            f"# R2-S2 Training Summary ({run_token})",
            "",
            f"- status: `{status}`",
            f"- candidate_checkpoint: `{best_checkpoint}`",
            f"- mean_score: `{metrics_payload.get('mean_score')}`",
            f"- eval_mean_score: `{metrics_payload.get('eval_mean_score')}`",
            f"- invalid_action_rate: `{metrics_payload.get('invalid_action_rate')}`",
            f"- hard_case_selected_failures: `{metrics_payload.get('hard_case_selected_failures')}`",
            f"- hard_case_failure_type_count: `{metrics_payload.get('hard_case_failure_type_count')}`",
            f"- self_imitation_replay_ratio: `{metrics_payload.get('self_imitation_replay_ratio')}`",
            f"- rollout_reuse_replay_ratio: `{metrics_payload.get('rollout_reuse_replay_ratio')}`",
            f"- rollout_reuse_selected_steps: `{metrics_payload.get('rollout_reuse_selected_steps')}`",
            f"- mean_policy_lag_updates: `{metrics_payload.get('mean_policy_lag_updates')}`",
            "",
            "## Bucket Replay Coverage",
            *(
                [
                    f"- {bucket}: count=`{bucket_metrics_payload['failure_bucket_counts'].get(bucket)}` ratio=`{bucket_metrics_payload['failure_bucket_ratios'].get(bucket)}` delta_vs_target=`{bucket_metrics_payload['bucket_mix_delta_vs_target'].get(bucket, 0.0)}`"
                    for bucket in sorted(bucket_metrics_payload["failure_bucket_counts"])
                ]
                if bucket_metrics_payload["failure_bucket_counts"]
                else ["- none"]
            ),
            "",
            "## Slice Coverage",
            *(
                [f"- {tag}: `{count}`" for tag, count in sorted(slice_metrics_payload["slice_tag_counts"].items(), key=lambda item: (-item[1], item[0]))[:12]]
                if slice_metrics_payload["slice_tag_counts"]
                else ["- none"]
            ),
        ],
    )
    (run_dir / "best_checkpoint.txt").write_text(str(best_checkpoint).strip() + "\n", encoding="utf-8")
    append_unified_progress_event(
        unified_progress_path,
        build_progress_event(
            run_id=run_token,
            component=_component_name_for_schema(cfg.schema),
            phase="train",
            status=status,
            step=len(seed_results),
            epoch_or_iter=len(seed_results),
            metrics={
                "mean_reward": metrics_payload.get("mean_reward"),
                "mean_score": metrics_payload.get("mean_score"),
                "policy_loss": metrics_payload.get("policy_loss"),
                "value_loss": metrics_payload.get("value_loss"),
                "entropy": metrics_payload.get("entropy"),
                "kl_divergence": metrics_payload.get("kl_divergence"),
                "invalid_action_rate": metrics_payload.get("invalid_action_rate"),
                "rollout_reuse_replay_ratio": metrics_payload.get("rollout_reuse_replay_ratio"),
                "mean_policy_lag_updates": metrics_payload.get("mean_policy_lag_updates"),
                "candidate_checkpoint": best_checkpoint,
            },
            device_profile=runtime_profile_payload,
            learner_device=learner_device,
            rollout_device=rollout_device,
            gpu_mem_mb=get_gpu_mem_mb(learner_device),
        ),
    )

    return {
        "status": status,
        "run_id": run_token,
        "run_dir": str(run_dir),
        "train_manifest": str(run_dir / "train_manifest.json"),
        "metrics": str(run_dir / "metrics.json"),
        "best_checkpoint": str(best_checkpoint),
        "seeds_used": str(run_dir / "seeds_used.json"),
        "reward_config": str(run_dir / "reward_config.json"),
        "hard_case_sampling_json": str(hard_case_sampling_json),
        "hardcase_manifest_json": str(hard_case_manifest_json),
        "hardcase_stats_json": str(hard_case_stats_json),
        "bucket_replay_manifest_json": str(bucket_replay_manifest_json),
        "bucket_replay_stats_json": str(bucket_replay_stats_json),
        "warnings_log": str(warnings_log_path),
        "curriculum_plan": str(curriculum_plan_path),
        "curriculum_applied": str(curriculum_applied_path),
        "reward_schedule_json": str(reward_schedule_json),
        "bucket_metrics_json": str(bucket_metrics_json),
        "slice_metrics_json": str(slice_metrics_json),
        "best_trajectory_manifest_json": str(best_trajectory_manifest_json),
        "best_trajectory_stats_json": str(best_trajectory_stats_json),
        "rollout_reuse_manifest_json": str(rollout_reuse_manifest_json),
        "rollout_reuse_stats_json": str(rollout_reuse_stats_json),
        "summary_md": str(run_dir / "summary.md"),
        "training_summary_md": str(training_summary_md),
        "eval_seed_results": str(eval_summary.get("seed_results_json") or ""),
        "diagnostics_json": str(diagnostics_summary.get("diagnostics_json") or ""),
        "diagnostics_report_md": str(diagnostics_summary.get("diagnostics_report_md") or ""),
        "seed_results": seed_results,
        "runtime_profile_json": str(runtime_profile_json),
        "progress_unified_jsonl": str(unified_progress_path),
        "checkpoint_id": str(checkpoint_registry_entry.get("checkpoint_id") or ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 PPO-lite RL trainer.")
    parser.add_argument("--config", default="configs/experiments/p42_rl_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed override")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_override = [seed.strip() for seed in str(args.seeds).split(",") if seed.strip()]
    summary = run_ppo_lite_training(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        seeds_override=seed_override or None,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
