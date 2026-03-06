from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from trainer import action_space, action_space_shop
from trainer.closed_loop.replay_manifest import (
    build_seeds_payload,
    now_iso,
    now_stamp,
    read_json,
    to_abs_path,
    write_json,
    write_markdown,
)
from trainer.common.slices import compute_slice_labels
from trainer.features_shop import SHOP_CONTEXT_DIM
from trainer.monitoring.progress_schema import append_progress_event, build_progress_event, get_gpu_mem_mb
from trainer.runtime.runtime_profile import load_runtime_profile
from trainer.world_model.imagination_schema import (
    IMAGINED_SOURCE_TYPE,
    apply_imagination_metadata,
    imagination_schema_markdown,
    make_root_sample_id,
)
from trainer.world_model.model import load_world_model_from_checkpoint
from trainer.world_model.planning_hook import PHASE_TO_ID
from trainer.world_model.schema import action_token_from_parts, stable_hash_int


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for trainer.world_model.imagination_rollout") from exc
    return torch


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise RuntimeError(f"PyYAML unavailable and no sidecar JSON for: {path}")
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be mapping: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def _find_latest_world_model_checkpoint(repo_root: Path) -> str:
    root = (repo_root / "docs/artifacts/p45/wm_train").resolve()
    if not root.exists():
        return ""
    best_candidates = sorted(root.glob("**/best.pt"), key=lambda path: str(path))
    if not best_candidates:
        return ""
    return str(best_candidates[-1].resolve())


def _resolve_checkpoint(repo_root: Path, cfg: dict[str, Any]) -> str:
    block = cfg.get("world_model") if isinstance(cfg.get("world_model"), dict) else {}
    raw = str(block.get("checkpoint") or cfg.get("checkpoint") or "").strip()
    if raw:
        return str(to_abs_path(repo_root, raw))
    return _find_latest_world_model_checkpoint(repo_root)


def _glob_files(root: Path, patterns: list[str]) -> list[Path]:
    if root.is_file():
        return [root.resolve()]
    if not root.exists():
        return []
    hits: list[Path] = []
    for pattern in patterns:
        for hit in root.glob(pattern):
            if hit.is_file():
                hits.append(hit.resolve())
    return sorted(set(hits), key=lambda path: str(path), reverse=True)


def _load_manifest_backed_files(repo_root: Path, manifest_path: Path) -> list[tuple[Path, str]]:
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        return []
    entries = payload.get("selected_entries") if isinstance(payload.get("selected_entries"), list) else []
    out: list[tuple[Path, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        format_hint = str(entry.get("format_hint") or "").strip().lower()
        path = Path(str(entry.get("path") or ""))
        if not path.is_absolute():
            path = to_abs_path(repo_root, path)
        if not path.exists():
            continue
        if format_hint != "bc_record_v1":
            continue
        out.append((path.resolve(), str(entry.get("source_type") or "replay_manifest")))
    return out


def _resolve_root_files(repo_root: Path, cfg: dict[str, Any], *, quick: bool) -> list[tuple[Path, str]]:
    imag_cfg = cfg.get("imagination") if isinstance(cfg.get("imagination"), dict) else {}
    root_cfg = imag_cfg.get("root_source") if isinstance(imag_cfg.get("root_source"), dict) else {}
    files: list[tuple[Path, str]] = []

    manifest_raw = str(root_cfg.get("replay_manifest") or imag_cfg.get("replay_manifest") or "").strip()
    if manifest_raw:
        files.extend(_load_manifest_backed_files(repo_root, to_abs_path(repo_root, manifest_raw)))

    raw_path = str(root_cfg.get("path") or imag_cfg.get("root_path") or "docs/artifacts/p16").strip()
    path = to_abs_path(repo_root, raw_path)
    patterns = root_cfg.get("patterns") if isinstance(root_cfg.get("patterns"), list) else []
    if not patterns:
        patterns = ["**/*dagger*.jsonl", "**/dataset*.jsonl"]
    for file_path in _glob_files(path, [str(pattern) for pattern in patterns]):
        files.append((file_path, str(root_cfg.get("source_type") or "p13_dagger_or_real")))

    unique: dict[str, tuple[Path, str]] = {}
    for file_path, source_type in files:
        unique[str(file_path)] = (file_path, source_type)
    resolved = list(unique.values())
    max_files = max(1, _safe_int(root_cfg.get("max_files"), 1 if quick else 3))
    return resolved[:max_files]


def _iter_bc_records(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            phase = str(obj.get("phase") or "")
            has_hand = obj.get("expert_action_id") is not None and phase == "SELECTING_HAND"
            has_shop = obj.get("shop_expert_action_id") is not None and phase in {"SHOP", "SMODS_BOOSTER_OPENED"}
            if not (has_hand or has_shop):
                continue
            out.append(obj)
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def _vectorize_record(record: dict[str, Any], *, input_dim: int) -> list[float]:
    phase = str(record.get("phase") or "OTHER").upper()
    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    shop_features = record.get("shop_features") if isinstance(record.get("shop_features"), dict) else {}

    context = list(features.get("context") or [0.0] * 12)
    if len(context) < 12:
        context = (context + [0.0] * 12)[:12]
    shop_context = list(shop_features.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
    if len(shop_context) < SHOP_CONTEXT_DIM:
        shop_context = (shop_context + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]

    hand_size = max(0, _safe_int(record.get("hand_size"), 0))
    legal_ids = list(record.get("legal_action_ids") or [])
    shop_legal_ids = list(record.get("shop_legal_action_ids") or [])
    legal_count = len(shop_legal_ids) if phase in action_space_shop.SHOP_PHASES else len(legal_ids)
    ranks = list(features.get("card_rank_ids") or [])
    chips = list(features.get("card_chip_hint") or [])
    enh = list(features.get("card_has_enhancement") or [])
    edt = list(features.get("card_has_edition") or [])
    seal = list(features.get("card_has_seal") or [])
    divisor = max(1, hand_size)
    phase_onehot = [0.0] * len(PHASE_TO_ID)
    phase_onehot[min(PHASE_TO_ID.get(phase, PHASE_TO_ID["OTHER"]), len(phase_onehot) - 1)] = 1.0

    vector = [
        float(context[0]) / 5.0,
        float(context[1]) / 5.0,
        float(context[2]) / 5000.0,
        float(context[3]) / 100.0,
        float(context[4]) / 20.0,
        float(context[5]) / 5000.0,
        float(context[6]) / 5000.0,
        float(context[7]) / 5000.0,
        float(context[8]) / 10.0,
        float(context[9]) / 4.0,
        float(context[10]) / 8.0,
        float(context[11]) / 20.0,
        *shop_context[:16],
        (sum(float(x) for x in ranks[: action_space.MAX_HAND]) / divisor) / 14.0 if ranks else 0.0,
        (max([float(x) for x in ranks[: action_space.MAX_HAND]] or [0.0])) / 14.0,
        (sum(float(x) for x in chips[: action_space.MAX_HAND]) / divisor) / 20.0 if chips else 0.0,
        (sum(float(x) for x in enh[: action_space.MAX_HAND]) / divisor) if enh else 0.0,
        (sum(float(x) for x in edt[: action_space.MAX_HAND]) / divisor) if edt else 0.0,
        (sum(float(x) for x in seal[: action_space.MAX_HAND]) / divisor) if seal else 0.0,
        *phase_onehot,
        float(hand_size) / float(max(1, action_space.MAX_HAND)),
        float(legal_count) / float(max(1, max(action_space.max_actions(), action_space_shop.max_actions()))),
        float(context[10]) / 8.0,
        float(context[11]) / 20.0,
        float(context[0]) / 5.0,
        float(context[1]) / 5.0,
    ]
    if len(vector) < input_dim:
        vector.extend([0.0] * (input_dim - len(vector)))
    return vector[:input_dim]


def _hand_action_dict(hand_size: int, action_id: int) -> dict[str, Any]:
    action_type, mask = action_space.decode(max(1, hand_size), int(action_id))
    return {
        "action_type": action_type,
        "indices": action_space.mask_to_indices(mask, max(1, hand_size)),
        "id": int(action_id),
    }


def _shop_action_dict(action_id: int) -> dict[str, Any]:
    action_type, params = action_space_shop.decode(int(action_id))
    payload: dict[str, Any] = {"action_type": action_type, "id": int(action_id)}
    if params:
        payload["params"] = dict(params)
    return payload


def _candidate_actions(
    record: dict[str, Any],
    *,
    action_vocab_size: int,
    limit: int,
) -> list[dict[str, Any]]:
    phase = str(record.get("phase") or "OTHER").upper()
    rows: list[dict[str, Any]] = []

    if phase == "SELECTING_HAND":
        legal_ids = [int(aid) for aid in (record.get("legal_action_ids") or []) if _safe_int(aid, -1) >= 0]
        expert_id = _safe_int(record.get("expert_action_id"), -1)
        ordered = ([expert_id] if expert_id >= 0 else []) + [aid for aid in legal_ids if aid != expert_id]
        hand_size = max(1, _safe_int(record.get("hand_size"), 1))
        for aid in ordered[: max(1, limit)]:
            action = _hand_action_dict(hand_size, aid)
            token = action_token_from_parts(
                phase=phase,
                action_type=str(action.get("action_type") or "OTHER"),
                action_payload=action,
                numeric_action=int(aid),
            )
            rows.append(
                {
                    "policy_action_id": int(aid),
                    "world_model_action_id": stable_hash_int(token, action_vocab_size),
                    "action": action,
                    "action_token": token,
                }
            )
    else:
        legal_ids = [int(aid) for aid in (record.get("shop_legal_action_ids") or []) if _safe_int(aid, -1) >= 0]
        expert_id = _safe_int(record.get("shop_expert_action_id"), -1)
        ordered = ([expert_id] if expert_id >= 0 else []) + [aid for aid in legal_ids if aid != expert_id]
        for aid in ordered[: max(1, limit)]:
            action = _shop_action_dict(aid)
            token = action_token_from_parts(
                phase=phase,
                action_type=str(action.get("action_type") or "OTHER"),
                action_payload=action,
                numeric_action=int(aid),
            )
            rows.append(
                {
                    "policy_action_id": int(aid),
                    "world_model_action_id": stable_hash_int(token, action_vocab_size),
                    "action": action,
                    "action_token": token,
                }
            )
    return rows


def _slice_labels(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    phase = str(record.get("phase") or "")
    action_type = str(action.get("action_type") or "")
    state = {
        "state": {
            "state": phase,
            "round": {
                "hands_left": _safe_int(((record.get("features") or {}).get("context_values") or {}).get("hands_left"), 0),
                "discards_left": _safe_int(((record.get("features") or {}).get("context_values") or {}).get("discards_left"), 0),
            },
            "money": _safe_int(((record.get("features") or {}).get("context_values") or {}).get("money"), 0),
            "ante_num": _safe_int(((record.get("features") or {}).get("context_values") or {}).get("ante_num"), 0),
            "round_num": _safe_int(((record.get("features") or {}).get("context_values") or {}).get("round_num"), 0),
            "jokers": [0] * _safe_int(((record.get("features") or {}).get("context_values") or {}).get("jokers_count"), 0),
        },
        "phase": phase,
        "action_type": action_type,
    }
    return compute_slice_labels(state)


def _rollout_step(
    model: Any,
    *,
    obs_vector: list[float],
    action_id: int,
    horizon: int,
    device: Any,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    obs_t = torch.tensor([list(obs_vector)], dtype=torch.float32, device=device)
    action_tensor = torch.tensor([int(action_id)], dtype=torch.long, device=device)

    with torch.no_grad():
        z_curr = model.encode(obs_t)
        rows: list[dict[str, Any]] = []
        for step_idx in range(1, max(1, int(horizon)) + 1):
            joint = model.joint(z_curr, action_tensor)
            z_next = model.transition(joint)
            reward_pred = model.reward_head(joint).squeeze(-1)
            score_pred = model.score_head(joint).squeeze(-1)
            resource_pred = model.resource_head(joint)
            uncertainty_pred = model.uncertainty_head(joint).squeeze(-1)
            rows.append(
                {
                    "imagined_step_idx": int(step_idx),
                    "predicted_latent": list(z_next.detach().cpu().tolist()[0]),
                    "predicted_reward": float(reward_pred.detach().cpu().item()),
                    "predicted_score_delta": float(score_pred.detach().cpu().item()),
                    "predicted_resource_delta": list(resource_pred.detach().cpu().tolist()[0]),
                    "uncertainty_score": float(abs(uncertainty_pred.detach().cpu().item())),
                    "predicted_done": False,
                }
            )
            z_curr = z_next
    return rows


def _stats_markdown(stats: dict[str, Any]) -> list[str]:
    lines = [
        f"# P46 Imagination Rollouts ({stats.get('run_id')})",
        "",
        f"- world_model_checkpoint: `{stats.get('world_model_checkpoint')}`",
        f"- root_file_count: {int(stats.get('root_file_count') or 0)}",
        f"- total_roots: {int(stats.get('total_roots') or 0)}",
        f"- total_imagined_samples: {int(stats.get('total_imagined_samples') or 0)}",
        f"- accepted_samples: {int(stats.get('accepted_samples') or 0)}",
        f"- acceptance_rate: {float(stats.get('acceptance_rate') or 0.0):.4f}",
        f"- mean_uncertainty: {float(stats.get('mean_uncertainty') or 0.0):.6f}",
        "",
        "## Source Distribution",
    ]
    for row in stats.get("source_distribution") if isinstance(stats.get("source_distribution"), list) else []:
        if not isinstance(row, dict):
            continue
        lines.append(
            "- {label}: count={count} ratio={ratio:.3f}".format(
                label=row.get("label"),
                count=int(row.get("count") or 0),
                ratio=float(row.get("ratio") or 0.0),
            )
        )
    lines.extend(["", "## Slice Distribution"])
    slice_distribution = stats.get("slice_distribution") if isinstance(stats.get("slice_distribution"), dict) else {}
    for key, rows in sorted(slice_distribution.items()):
        lines.append(f"- {key}:")
        if isinstance(rows, list):
            for row in rows[:6]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  - {label}: count={count} ratio={ratio:.3f}".format(
                        label=row.get("label"),
                        count=int(row.get("count") or 0),
                        ratio=float(row.get("ratio") or 0.0),
                    )
                )
    warnings = stats.get("warnings") if isinstance(stats.get("warnings"), list) else []
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {str(warning)}")
    return lines


def _distribution(counter: Counter[str]) -> list[dict[str, Any]]:
    total = max(1, sum(counter.values()))
    return [
        {"label": label, "count": int(count), "ratio": float(count) / float(total)}
        for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def run_imagination_rollout(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    seeds_override: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = to_abs_path(repo_root, config_path)
    cfg = _read_yaml_or_json(cfg_path)
    imag_cfg = cfg.get("imagination") if isinstance(cfg.get("imagination"), dict) else {}
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}

    checkpoint_path = _resolve_checkpoint(repo_root, cfg)
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError("world model checkpoint unavailable for imagination rollout")
    horizon = max(1, _safe_int(imag_cfg.get("horizon"), 1))
    if quick:
        horizon = min(horizon, 2)
    num_imagined_per_root = max(1, _safe_int(imag_cfg.get("num_imagined_per_root"), 1))
    max_samples = max(1, _safe_int(imag_cfg.get("max_samples"), 128 if quick else 512))
    max_roots = max(1, _safe_int(imag_cfg.get("max_roots"), max_samples))
    uncertainty_threshold = max(0.0, _safe_float(imag_cfg.get("uncertainty_threshold"), 0.75))
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())

    out_root = (
        (
            repo_root
            / str(
                imag_cfg.get("output_artifacts_root")
                or output_cfg.get("imagination_artifacts_root")
                or output_cfg.get("artifacts_root")
                or "docs/artifacts/p46/imagination_rollouts"
            )
        ).resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    run_dir = out_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = run_dir / "imagined_rollouts.jsonl"
    if rollouts_path.exists():
        rollouts_path.unlink()
    runtime_profile_payload = load_runtime_profile(config=cfg, component="p46_imagination_rollout").to_dict()
    runtime_resolved = (
        runtime_profile_payload.get("resolved_profile", {}).get("resolved")
        if isinstance(runtime_profile_payload.get("resolved_profile"), dict)
        else {}
    )
    learner_device = str((runtime_resolved or {}).get("learner_device") or "cpu")
    rollout_device = str((runtime_resolved or {}).get("rollout_device") or "cpu")
    runtime_profile_json = run_dir / "runtime_profile.json"
    progress_unified_path = run_dir / "progress.unified.jsonl"
    write_json(runtime_profile_json, runtime_profile_payload)
    model, model_cfg, _payload = load_world_model_from_checkpoint(checkpoint_path, device=learner_device)

    root_files = _resolve_root_files(repo_root, cfg, quick=quick)
    seeds = [str(seed).strip() for seed in (seeds_override or imag_cfg.get("seeds") or []) if str(seed).strip()]
    root_seeds: set[str] = set()
    if not seeds:
        seeds = ["AAAAAAA"]

    warnings: list[str] = []
    source_counter: Counter[str] = Counter()
    root_source_counter: Counter[str] = Counter()
    slice_counters: dict[str, Counter[str]] = defaultdict(Counter)
    uncertainty_values: list[float] = []
    reward_values: list[float] = []
    imagined_rows: list[dict[str, Any]] = []
    total_roots = 0
    accepted_samples = 0

    for root_path, root_source_type in root_files:
        records = _iter_bc_records(root_path, max_rows=max_roots if quick else max_roots * 2)
        for record in records:
            if total_roots >= max_roots or len(imagined_rows) >= max_samples:
                break
            root_seed = str(record.get("seed") or "")
            if root_seed:
                root_seeds.add(root_seed)
            total_roots += 1
            root_sample_id = make_root_sample_id(record, source_path=root_path)
            obs_vector = _vectorize_record(record, input_dim=int(model_cfg.input_dim))
            action_rows = _candidate_actions(record, action_vocab_size=int(model_cfg.action_vocab_size), limit=num_imagined_per_root)
            if not action_rows:
                warnings.append(f"no_candidate_actions:{root_path.name}:{root_sample_id}")
                continue

            for action_row in action_rows:
                slice_labels = _slice_labels(record, action_row.get("action") if isinstance(action_row.get("action"), dict) else {})
                for rollout_row in _rollout_step(
                    model,
                    obs_vector=obs_vector,
                    action_id=int(action_row.get("world_model_action_id") or 0),
                    horizon=horizon,
                    device=learner_device,
                ):
                    if len(imagined_rows) >= max_samples:
                        break
                    uncertainty_score = max(0.0, _safe_float(rollout_row.get("uncertainty_score"), 0.0))
                    gate_passed = uncertainty_score <= uncertainty_threshold
                    valid_for_training = bool(rollout_row.get("imagined_step_idx") == 1)
                    imagined = apply_imagination_metadata(
                        record,
                        world_model_checkpoint=checkpoint_path,
                        imagination_horizon=horizon,
                        uncertainty_score=uncertainty_score,
                        uncertainty_gate_passed=gate_passed,
                        root_sample_id=root_sample_id,
                        imagined_step_idx=int(rollout_row.get("imagined_step_idx") or 1),
                        teacher_seed=str(record.get("seed") or seeds[0]),
                        valid_for_training=valid_for_training,
                        predicted_reward=float(rollout_row.get("predicted_reward") or 0.0),
                        predicted_score_delta=float(rollout_row.get("predicted_score_delta") or 0.0),
                        predicted_resource_delta=list(rollout_row.get("predicted_resource_delta") or []),
                        predicted_done=bool(rollout_row.get("predicted_done")),
                        predicted_latent=list(rollout_row.get("predicted_latent") or []),
                        source_path=str(root_path),
                        source_run_id=str(root_path.parent.name),
                    )
                    imagined["root_source_type"] = str(root_source_type)
                    imagined["action_token"] = str(action_row.get("action_token") or "")
                    imagined["policy_action_id"] = int(action_row.get("policy_action_id") or 0)
                    imagined["world_model_action_id"] = int(action_row.get("world_model_action_id") or 0)
                    imagined["slice_labels"] = dict(slice_labels)
                    imagined["generation_method"] = "world_model_imagination"
                    imagined["root_source_path"] = str(root_path)
                    imagined["training_eligible"] = bool(valid_for_training and gate_passed)
                    imagined["imagined_record_id"] = "{root}:{action}:{step}".format(
                        root=root_sample_id,
                        action=int(action_row.get("policy_action_id") or 0),
                        step=int(rollout_row.get("imagined_step_idx") or 1),
                    )

                    if not dry_run:
                        _append_jsonl(rollouts_path, imagined)
                    imagined_rows.append(imagined)
                    source_counter[IMAGINED_SOURCE_TYPE] += 1
                    root_source_counter[str(root_source_type)] += 1
                    for key, value in slice_labels.items():
                        slice_counters[str(key)][str(value)] += 1
                    uncertainty_values.append(uncertainty_score)
                    reward_values.append(float(rollout_row.get("predicted_reward") or 0.0))
                    if gate_passed:
                        accepted_samples += 1
                if len(imagined_rows) >= max_samples:
                    break
        if total_roots >= max_roots or len(imagined_rows) >= max_samples:
            break

    seeds_payload = build_seeds_payload(sorted(set(seeds) | root_seeds), seed_policy_version="p46.imagination_rollout")
    seeds_payload["metadata"] = {
        "requested_seeds": list(seeds),
        "root_seed_count": len(root_seeds),
        "root_file_count": len(root_files),
    }
    write_json(run_dir / "seeds_used.json", seeds_payload)

    total_samples = len(imagined_rows)
    stats = {
        "schema": "p46_imagination_stats_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": "ok" if total_samples > 0 else "stub",
        "world_model_checkpoint": checkpoint_path,
        "runtime_profile": runtime_profile_payload,
        "learner_device": learner_device,
        "rollout_device": rollout_device,
        "gpu_mem_mb": get_gpu_mem_mb(learner_device),
        "root_file_count": len(root_files),
        "root_files": [str(path) for path, _source_type in root_files],
        "total_roots": int(total_roots),
        "total_imagined_samples": int(total_samples),
        "accepted_samples": int(accepted_samples),
        "acceptance_rate": float(accepted_samples) / max(1, total_samples),
        "mean_uncertainty": (sum(uncertainty_values) / max(1, len(uncertainty_values))),
        "mean_reward_pred": (sum(reward_values) / max(1, len(reward_values))),
        "uncertainty_threshold": float(uncertainty_threshold),
        "source_distribution": _distribution(root_source_counter),
        "slice_distribution": {
            key: _distribution(counter)
            for key, counter in sorted(slice_counters.items())
        },
        "warnings": warnings,
    }
    manifest = {
        "schema": "p46_imagination_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": str(stats.get("status") or "stub"),
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "world_model_checkpoint": checkpoint_path,
        "imagined_rollouts_jsonl": str(rollouts_path) if not dry_run else "",
        "imagined_stats_json": str(run_dir / "imagined_stats.json"),
        "imagined_stats_md": str(run_dir / "imagined_stats.md"),
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "accepted_sample_count": int(accepted_samples),
        "total_sample_count": int(total_samples),
        "acceptance_rate": float(stats.get("acceptance_rate") or 0.0),
        "average_uncertainty": float(stats.get("mean_uncertainty") or 0.0),
        "uncertainty_threshold": float(uncertainty_threshold),
        "horizon": int(horizon),
        "max_training_horizon": 1,
        "source_type": IMAGINED_SOURCE_TYPE,
        "generation_method": "world_model_imagination",
        "runtime_profile_json": str(runtime_profile_json),
        "progress_unified_jsonl": str(progress_unified_path),
    }
    write_json(run_dir / "imagined_manifest.json", manifest)
    write_json(run_dir / "imagined_stats.json", stats)
    write_markdown(run_dir / "imagined_stats.md", _stats_markdown(stats))
    write_json(run_dir / "schema_preview.json", imagined_rows[:5])
    write_markdown(run_dir / "schema_preview.md", imagination_schema_markdown())
    append_progress_event(
        progress_unified_path,
        build_progress_event(
            run_id=chosen_run_id,
            component="p46_imagination",
            phase="rollout",
            status=str(stats.get("status") or "stub"),
            step=int(total_samples),
            epoch_or_iter=int(total_roots),
            metrics={
                "accepted_samples": int(accepted_samples),
                "total_imagined_samples": int(total_samples),
                "acceptance_rate": float(stats.get("acceptance_rate") or 0.0),
                "mean_uncertainty": float(stats.get("mean_uncertainty") or 0.0),
                "mean_reward_pred": float(stats.get("mean_reward_pred") or 0.0),
            },
            device_profile=runtime_profile_payload,
            learner_device=learner_device,
            rollout_device=rollout_device,
            throughput=float(total_samples) / max(1.0, float(total_roots or 1)),
            gpu_mem_mb=get_gpu_mem_mb(learner_device),
            warning=(";".join(warnings[:3]) if warnings else ""),
        ),
    )

    return {
        "status": str(stats.get("status") or "stub"),
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "imagined_manifest_json": str(run_dir / "imagined_manifest.json"),
        "imagined_stats_json": str(run_dir / "imagined_stats.json"),
        "imagined_stats_md": str(run_dir / "imagined_stats.md"),
        "imagined_rollouts_jsonl": str(rollouts_path) if not dry_run else "",
        "seeds_used_json": str(run_dir / "seeds_used.json"),
        "accepted_samples": int(accepted_samples),
        "total_imagined_samples": int(total_samples),
        "acceptance_rate": float(stats.get("acceptance_rate") or 0.0),
        "runtime_profile_json": str(runtime_profile_json),
        "progress_unified_jsonl": str(progress_unified_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate short uncertainty-gated imagined rollouts from a P45 world model.")
    parser.add_argument("--config", default="configs/experiments/p46_imagination_smoke.yaml")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [token.strip() for token in str(args.seeds or "").split(",") if token.strip()]
    summary = run_imagination_rollout(
        config_path=args.config,
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        seeds_override=(seeds or None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
