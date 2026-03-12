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

from trainer.closed_loop.failure_buckets import KNOWN_FAILURE_BUCKETS, classify_failure_bucket, scarce_failure_buckets
from trainer.closed_loop.replay_manifest import now_iso, now_stamp, read_json, write_json, write_markdown


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yml", ".yaml"}:
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


def _normalized_float_mapping(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        token = str(key).strip()
        if not token:
            continue
        out[token] = _safe_float(value, 0.0)
    return out


def _normalized_int_mapping(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        token = str(key).strip()
        if not token:
            continue
        out[token] = _safe_int(value, 0)
    return out


def _normalized_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    token = str(raw or "").strip()
    return [token] if token else []


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    qq = min(1.0, max(0.0, float(q)))
    pos = (len(ordered) - 1) * qq
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _read_jsonl_with_lineno(path: Path, *, max_rows: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for idx, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            obj["_source_line"] = idx
            out.append(obj)
            if max_rows > 0 and len(out) >= max_rows:
                break
    return out


def _latest_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    dirs = sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        return None
    return dirs[-1]


def _resolve_indexed_path_list(
    raw_paths: list[str],
    *,
    repo_root: Path,
) -> list[Path]:
    resolved: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        resolved.append(path)
    return resolved


def _resolve_candidate_decision_path(
    *,
    repo_root: Path,
    explicit_path: str,
    promotion_decision_path: Path | None,
    p39_root: Path,
) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path
    if promotion_decision_path is not None and promotion_decision_path.exists():
        payload = read_json(promotion_decision_path)
        if isinstance(payload, dict):
            inner = payload.get("champion_rules_payload")
            if isinstance(inner, dict):
                path_text = str(inner.get("candidate_decision_json") or "").strip()
                if path_text:
                    path = Path(path_text)
                    if not path.is_absolute():
                        path = (repo_root / path).resolve()
                    return path
    latest_eval = _latest_dir(p39_root)
    if latest_eval and latest_eval.name.startswith("champion_eval_"):
        path = latest_eval / "candidate_decision.json"
        if path.exists():
            return path
    return None


def _resolve_slice_breakdown_path(
    *,
    repo_root: Path,
    explicit_path: str,
    promotion_decision_path: Path | None,
) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path
    if promotion_decision_path is not None and promotion_decision_path.exists():
        payload = read_json(promotion_decision_path)
        if isinstance(payload, dict):
            path_text = str(payload.get("slice_decision_breakdown_json") or "").strip()
            if not path_text:
                inner = payload.get("champion_rules_payload")
                if isinstance(inner, dict):
                    path_text = str(inner.get("slice_decision_breakdown_json") or "").strip()
            if path_text:
                path = Path(path_text)
                if not path.is_absolute():
                    path = (repo_root / path).resolve()
                return path
    return None


def _resolve_failure_sources(
    *,
    repo_root: Path,
    p39_root: Path,
    input_cfg: dict[str, Any],
    arena_run_dir_override: str | Path | None,
) -> list[dict[str, Any]]:
    if arena_run_dir_override:
        arena_run_dirs = [str(arena_run_dir_override)]
    else:
        arena_run_dirs = _normalized_string_list(input_cfg.get("arena_run_dirs"))
        if not arena_run_dirs:
            arena_run_dir = str(input_cfg.get("arena_run_dir") or "").strip()
            if arena_run_dir:
                arena_run_dirs = [arena_run_dir]
            else:
                latest = _latest_dir(p39_root / "arena_runs")
                if latest is not None:
                    arena_run_dirs = [str(latest)]

    candidate_decision_paths = _normalized_string_list(input_cfg.get("candidate_decision_jsons"))
    if not candidate_decision_paths:
        single_decision = str(input_cfg.get("candidate_decision_json") or "").strip()
        if single_decision:
            candidate_decision_paths = [single_decision]

    triage_report_paths = _normalized_string_list(input_cfg.get("triage_report_jsons"))
    if not triage_report_paths:
        single_triage = str(input_cfg.get("triage_report_json") or "").strip()
        if single_triage:
            triage_report_paths = [single_triage]

    slice_breakdown_paths = _normalized_string_list(input_cfg.get("slice_breakdown_jsons"))
    if not slice_breakdown_paths:
        single_slice = str(input_cfg.get("slice_breakdown_json") or "").strip()
        if single_slice:
            slice_breakdown_paths = [single_slice]

    promotion_decision_paths = _normalized_string_list(input_cfg.get("promotion_decision_jsons"))
    if not promotion_decision_paths:
        single_promotion = str(input_cfg.get("promotion_decision_json") or "").strip()
        if single_promotion:
            promotion_decision_paths = [single_promotion]

    resolved_arena = _resolve_indexed_path_list(arena_run_dirs, repo_root=repo_root)
    resolved_candidate = _resolve_indexed_path_list(candidate_decision_paths, repo_root=repo_root)
    resolved_triage = _resolve_indexed_path_list(triage_report_paths, repo_root=repo_root)
    resolved_slice = _resolve_indexed_path_list(slice_breakdown_paths, repo_root=repo_root)
    resolved_promotion = _resolve_indexed_path_list(promotion_decision_paths, repo_root=repo_root)

    sources: list[dict[str, Any]] = []
    for idx, arena_run_dir in enumerate(resolved_arena):
        promotion_path = (
            resolved_promotion[idx]
            if idx < len(resolved_promotion)
            else (resolved_promotion[0] if len(resolved_promotion) == 1 else None)
        )
        candidate_decision_path = _resolve_candidate_decision_path(
            repo_root=repo_root,
            explicit_path=str(resolved_candidate[idx]) if idx < len(resolved_candidate) else (
                str(resolved_candidate[0]) if len(resolved_candidate) == 1 else ""
            ),
            promotion_decision_path=promotion_path,
            p39_root=p39_root,
        )
        triage_report_path = (
            resolved_triage[idx]
            if idx < len(resolved_triage)
            else (resolved_triage[0] if len(resolved_triage) == 1 else None)
        )
        slice_breakdown_path = _resolve_slice_breakdown_path(
            repo_root=repo_root,
            explicit_path=str(resolved_slice[idx]) if idx < len(resolved_slice) else (
                str(resolved_slice[0]) if len(resolved_slice) == 1 else ""
            ),
            promotion_decision_path=promotion_path,
        )
        sources.append(
            {
                "source_index": idx,
                "source_run_id": str(arena_run_dir.name),
                "source_type": "arena_failure_mining",
                "arena_run_dir": arena_run_dir,
                "episode_records_path": arena_run_dir / "episode_records.jsonl",
                "summary_path": arena_run_dir / "summary_table.json",
                "bucket_path": arena_run_dir / "bucket_metrics.json",
                "candidate_decision_path": candidate_decision_path,
                "triage_report_path": triage_report_path,
                "slice_breakdown_path": slice_breakdown_path,
                "promotion_decision_path": promotion_path,
            }
        )
    return sources


def _pick_policy_row(summary_rows: list[dict[str, Any]], policy_id: str | None) -> dict[str, Any] | None:
    if not policy_id:
        return None
    token = str(policy_id).strip().lower()
    for row in summary_rows:
        if str(row.get("policy_id") or "").strip().lower() == token:
            return row
    return None


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _top_bucket_labels(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    pairs = [
        (str(label).strip(), _safe_int(count))
        for label, count in raw.items()
        if str(label).strip() and _safe_int(count) > 0
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _count in pairs]


def _infer_slice_tags(row: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    bucket_counts = row.get("bucket_counts")
    if isinstance(bucket_counts, dict):
        for category, payload in bucket_counts.items():
            category_token = str(category).strip()
            if not category_token or not isinstance(payload, dict):
                continue
            for label in _top_bucket_labels(payload)[:3]:
                tags.add(f"{category_token}:{label}")
    for key in ("phase", "action_type", "resource_pressure", "stake", "deck"):
        token = str(row.get(key) or "").strip()
        if token:
            tags.add(f"{key}:{token}")
    return sorted(tags)


def _infer_risk_tags(row: dict[str, Any], *, high_risk_round_threshold: int, low_threshold: float) -> list[str]:
    tags: set[str] = set()
    bucket_counts = row.get("bucket_counts")
    if isinstance(bucket_counts, dict):
        risk_payload = bucket_counts.get("risk")
        if isinstance(risk_payload, dict):
            for label in _top_bucket_labels(risk_payload):
                tags.add(str(label))
    rounds_survived = _safe_int(row.get("rounds_survived"), 0)
    score = _safe_float(row.get("total_score"), 0.0)
    if rounds_survived <= high_risk_round_threshold:
        tags.add("early_collapse")
    if score <= low_threshold:
        tags.add("low_score_tail")
    return sorted(tags)


def _load_slice_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    payload = raw.get("degraded_slices_topk")
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    payload = raw.get("rows")
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _slice_tag_from_row(row: dict[str, Any]) -> str:
    slice_key = str(row.get("slice_key") or "").strip()
    slice_label = str(row.get("slice_label") or "").strip()
    if not slice_key or not slice_label:
        return ""
    return f"{slice_key}:{slice_label}"


def _build_slice_priority_weights(*slice_payloads: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    for payload in slice_payloads:
        for row in _load_slice_rows(payload):
            tag = _slice_tag_from_row(row)
            if not tag:
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
            champion_count = _safe_int(row.get("champion_count"), 0)
            candidate_count = _safe_int(row.get("candidate_count"), 0)
            score_delta = _safe_float(metrics.get("mean_total_score_delta"), 0.0)
            weight = 1.0
            if bool(signals.get("degraded_significant")):
                weight += 0.5
            if champion_count > candidate_count:
                weight += min(0.75, 0.1 * float(champion_count - candidate_count))
            if candidate_count == 0 and champion_count > 0:
                weight += 0.5
            if score_delta < 0.0:
                weight += min(1.25, abs(score_delta) / 200.0)
            weights[tag] = max(weights.get(tag, 0.0), round(weight, 4))
    return weights


def _row_selection_matches_policy(
    *,
    payload: dict[str, Any],
    selection_by_type: Counter[str],
    selection_by_seed: Counter[str],
    selection_by_bucket: Counter[str],
    selection_by_source: Counter[str],
    selection_by_source_type: Counter[str],
    selection_by_source_variant: Counter[str],
    selection_by_overlap_key: Counter[str],
    max_failures_per_type: int,
    max_failures_per_seed: int,
    max_failures_per_bucket: dict[str, int],
    max_failures_per_source: dict[str, int],
    max_failures_per_source_variant: dict[str, int],
    max_failures_per_overlap_key: int,
) -> bool:
    seed = str(payload.get("seed") or "")
    bucket = str(payload.get("failure_bucket") or "")
    source_run_id = str(payload.get("source_run_id") or "")
    source_variant = str(payload.get("source_variant") or "")
    overlap_key = _selection_overlap_key(payload)
    failure_types = payload.get("failure_types") if isinstance(payload.get("failure_types"), list) else []
    if max_failures_per_seed > 0 and selection_by_seed[seed] >= max_failures_per_seed:
        return False
    if max_failures_per_type > 0 and any(selection_by_type[str(token)] >= max_failures_per_type for token in failure_types):
        return False
    bucket_cap = _safe_int(max_failures_per_bucket.get(bucket), 0)
    if bucket_cap > 0 and selection_by_bucket[bucket] >= bucket_cap:
        return False
    source_cap = _safe_int(max_failures_per_source.get(source_run_id), 0)
    if source_cap > 0 and selection_by_source[source_run_id] >= source_cap:
        return False
    variant_cap = _safe_int(max_failures_per_source_variant.get(source_variant), 0)
    if source_variant and variant_cap > 0 and selection_by_source_variant[source_variant] >= variant_cap:
        return False
    if overlap_key and max_failures_per_overlap_key > 0 and selection_by_overlap_key[overlap_key] >= max_failures_per_overlap_key:
        return False
    return True


def _bucket_label_count(row: dict[str, Any], category: str, label: str) -> int:
    counts = row.get("bucket_counts") if isinstance(row.get("bucket_counts"), dict) else {}
    payload = counts.get(category) if isinstance(counts.get(category), dict) else {}
    return _safe_int(payload.get(label), 0)


def _refine_failure_bucket_for_slice_pressure(
    *,
    row: dict[str, Any],
    failure_types: set[str],
    failure_bucket: str,
    bucket_reason: str,
    failure_bucket_candidates: list[str],
    failure_bucket_signals: list[str],
) -> tuple[str, str, list[str], list[str]]:
    if failure_bucket != "discard_mismanagement" or "triage_degraded_slice" not in failure_types:
        return failure_bucket, bucket_reason, failure_bucket_candidates, failure_bucket_signals

    play_count = _bucket_label_count(row, "slice_action_type", "play")
    discard_count = _bucket_label_count(row, "slice_action_type", "discard")
    shop_count = _bucket_label_count(row, "slice_action_type", "shop")
    low_pressure = _bucket_label_count(row, "slice_resource_pressure", "low")
    medium_pressure = _bucket_label_count(row, "slice_resource_pressure", "medium")
    high_pressure = _bucket_label_count(row, "slice_resource_pressure", "high")

    if shop_count > 0 and shop_count >= max(1, discard_count):
        new_bucket = "shop_or_economy_misallocation"
        new_reason = "triage_slice_shop_or_economy_gap"
    elif play_count > 0 and play_count >= discard_count and high_pressure > 0:
        new_bucket = "risk_overcommit"
        new_reason = "triage_slice_play_under_resource_tight"
    elif play_count > 0 and play_count >= discard_count and (low_pressure > 0 or medium_pressure > 0):
        new_bucket = "risk_undercommit"
        new_reason = "triage_slice_play_under_resource_slack"
    else:
        return failure_bucket, bucket_reason, failure_bucket_candidates, failure_bucket_signals

    refined_candidates = [new_bucket] + [token for token in failure_bucket_candidates if token != new_bucket]
    refined_signals = [new_reason] + [token for token in failure_bucket_signals if token != new_reason]
    return new_bucket, new_reason, refined_candidates, refined_signals


def _bucket_from_slice_tag(tag: str) -> str:
    token = str(tag or "").strip().lower()
    if not token:
        return ""
    if token.endswith(":unknown") or token.endswith(":none") or token.endswith(":absent") or token.endswith(":false"):
        return ""
    if token.startswith("slice_action_type:shop"):
        return "shop_or_economy_misallocation"
    if token.startswith("slice_action_type:discard"):
        return "discard_mismanagement"
    if token.startswith("slice_action_type:play"):
        return "risk_undercommit"
    if token.startswith("slice_resource_pressure:high") or token.startswith("slice_resource_pressure:medium"):
        return "resource_pressure_misplay"
    if token.startswith("slice_stage:early"):
        return "early_collapse"
    if token.startswith("slice_position_sensitive:true") or token.startswith("slice_position_sensitive:yes"):
        return "position_sensitive_misplay"
    if token.startswith("slice_stateful_joker_present:true") or token.startswith("slice_stateful_joker_present:yes"):
        return "stateful_joker_misplay"
    return ""


def _canonicalize_source_variant_token(tag: str) -> str:
    token = str(tag or "").strip().lower()
    if token.startswith("slice_position_sensitive:yes"):
        return token.replace(":yes", ":true", 1)
    if token.startswith("slice_stateful_joker_present:yes"):
        return token.replace(":yes", ":true", 1)
    return token


def _source_variant_from_slice_tag(tag: str) -> str:
    token = _canonicalize_source_variant_token(tag)
    if not token:
        return ""
    if token.endswith(":unknown") or token.endswith(":none") or token.endswith(":absent") or token.endswith(":false"):
        return ""
    for prefix in (
        "slice_position_sensitive:true",
        "slice_stateful_joker_present:true",
        "slice_action_type:shop",
        "slice_action_type:discard",
        "slice_resource_pressure:high",
        "slice_resource_pressure:medium",
        "slice_action_type:play",
        "slice_stage:early",
    ):
        if token.startswith(prefix):
            return token
    return ""


def _infer_source_variant(*, slice_tags: list[str], failure_bucket: str) -> str:
    actionable_tokens = [
        _source_variant_from_slice_tag(tag)
        for tag in slice_tags
        if _source_variant_from_slice_tag(tag)
    ]
    bucket_token = str(failure_bucket or "").strip().lower()
    bucket_source_override = {
        "resource_pressure_misplay": "bucket:resource_pressure_misplay",
        "shop_or_economy_misallocation": "bucket:shop_or_economy_misallocation",
        "position_sensitive_misplay": "bucket:position_sensitive_misplay",
        "stateful_joker_misplay": "bucket:stateful_joker_misplay",
    }
    bucket_override_requires_explicit = {
        "position_sensitive_misplay": {"slice_position_sensitive:true"},
        "stateful_joker_misplay": {"slice_stateful_joker_present:true"},
    }
    if actionable_tokens:
        if bucket_token in bucket_source_override:
            required_tokens = bucket_override_requires_explicit.get(bucket_token, set())
            if not required_tokens or not any(token in required_tokens for token in actionable_tokens):
                return bucket_source_override[bucket_token]
        priority = {
            "slice_position_sensitive:true": 0,
            "slice_stateful_joker_present:true": 1,
            "slice_action_type:shop": 2,
            "slice_action_type:discard": 3,
            "slice_resource_pressure:high": 4,
            "slice_resource_pressure:medium": 5,
            "slice_action_type:play": 6,
            "slice_stage:early": 7,
        }
        actionable_tokens.sort(key=lambda token: (priority.get(token, 99), token))
        return actionable_tokens[0]
    if bucket_token:
        return f"bucket:{bucket_token}"
    return ""


def _preferred_source_variant(*, primary_slice_tag: str, slice_tags: list[str], failure_bucket: str) -> str:
    bucket_or_fallback = _infer_source_variant(slice_tags=slice_tags, failure_bucket=failure_bucket)
    if bucket_or_fallback.startswith("bucket:"):
        return bucket_or_fallback
    return _source_variant_from_slice_tag(primary_slice_tag) or bucket_or_fallback


def _compound_actionable_slice_tags(slice_tags: list[str], *, primary_tag: str) -> list[str]:
    primary = str(primary_tag or "").strip().lower()
    results: list[str] = []
    seen: set[str] = set()
    for tag in slice_tags:
        token = str(tag or "").strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered == primary:
            continue
        if lowered in seen:
            continue
        if not _bucket_from_slice_tag(token):
            continue
        seen.add(lowered)
        results.append(token)
    return results


def _selection_overlap_key(payload: dict[str, Any]) -> str:
    source_run_id = str(payload.get("source_run_id") or "").strip()
    policy_id = str(payload.get("policy_id") or "").strip()
    seed = str(payload.get("seed") or "").strip()
    episode_index = _safe_int(payload.get("episode_index"), -1)
    if source_run_id or policy_id or seed or episode_index >= 0:
        return f"{source_run_id}|{policy_id}|{seed}|{episode_index}"
    return str(payload.get("episode_id") or "").strip()


def _decrement_counter(counter: Counter[str], token: str) -> None:
    key = str(token or "").strip()
    if not key:
        return
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


def _can_swap_overlap_selected_row(
    *,
    candidate_payload: dict[str, Any],
    existing_payload: dict[str, Any],
    selection_by_type: Counter[str],
    selection_by_seed: Counter[str],
    selection_by_bucket: Counter[str],
    selection_by_source: Counter[str],
    selection_by_source_type: Counter[str],
    selection_by_source_variant: Counter[str],
    selection_by_overlap_key: Counter[str],
    max_failures_per_type: int,
    max_failures_per_seed: int,
    max_failures_per_bucket: dict[str, int],
    max_failures_per_source: dict[str, int],
    max_failures_per_source_variant: dict[str, int],
    max_failures_per_overlap_key: int,
    min_failures_per_bucket: dict[str, int],
    min_failures_per_source_type: dict[str, int],
    min_failures_per_source_variant: dict[str, int],
) -> bool:
    candidate_overlap = _selection_overlap_key(candidate_payload)
    existing_overlap = _selection_overlap_key(existing_payload)
    if not candidate_overlap or candidate_overlap != existing_overlap:
        return False

    candidate_variant = str(candidate_payload.get("source_variant") or "").strip()
    existing_variant = str(existing_payload.get("source_variant") or "").strip()
    if not candidate_variant or candidate_variant == existing_variant:
        return False

    temp_by_type = Counter(selection_by_type)
    temp_by_seed = Counter(selection_by_seed)
    temp_by_bucket = Counter(selection_by_bucket)
    temp_by_source = Counter(selection_by_source)
    temp_by_source_type = Counter(selection_by_source_type)
    temp_by_source_variant = Counter(selection_by_source_variant)
    temp_by_overlap_key = Counter(selection_by_overlap_key)

    existing_seed = str(existing_payload.get("seed") or "").strip()
    existing_bucket = str(existing_payload.get("failure_bucket") or "").strip()
    existing_source = str(existing_payload.get("source_run_id") or "").strip()
    existing_source_type = str(existing_payload.get("source_type") or "").strip()

    _decrement_counter(temp_by_seed, existing_seed)
    _decrement_counter(temp_by_bucket, existing_bucket)
    _decrement_counter(temp_by_source, existing_source)
    _decrement_counter(temp_by_source_type, existing_source_type)
    _decrement_counter(temp_by_source_variant, existing_variant)
    _decrement_counter(temp_by_overlap_key, existing_overlap)
    for token in existing_payload.get("failure_types") if isinstance(existing_payload.get("failure_types"), list) else []:
        _decrement_counter(temp_by_type, str(token))

    if existing_bucket and temp_by_bucket[existing_bucket] < _safe_int(min_failures_per_bucket.get(existing_bucket), 0):
        return False
    if existing_source_type and temp_by_source_type[existing_source_type] < _safe_int(min_failures_per_source_type.get(existing_source_type), 0):
        return False
    if existing_variant and temp_by_source_variant[existing_variant] < _safe_int(min_failures_per_source_variant.get(existing_variant), 0):
        return False

    return _row_selection_matches_policy(
        payload=candidate_payload,
        selection_by_type=temp_by_type,
        selection_by_seed=temp_by_seed,
        selection_by_bucket=temp_by_bucket,
        selection_by_source=temp_by_source,
        selection_by_source_type=temp_by_source_type,
        selection_by_source_variant=temp_by_source_variant,
        selection_by_overlap_key=temp_by_overlap_key,
        max_failures_per_type=max_failures_per_type,
        max_failures_per_seed=max_failures_per_seed,
        max_failures_per_bucket=max_failures_per_bucket,
        max_failures_per_source=max_failures_per_source,
        max_failures_per_source_variant=max_failures_per_source_variant,
        max_failures_per_overlap_key=max_failures_per_overlap_key,
    )


def _pair_counter_from_groups(groups: list[list[dict[str, Any]]], field_name: str) -> dict[str, int]:
    pairs: Counter[str] = Counter()
    for rows in groups:
        tokens = sorted(
            {
                str(row.get(field_name) or "").strip()
                for row in rows
                if str(row.get(field_name) or "").strip()
            }
        )
        if len(tokens) < 2:
            continue
        for idx, left in enumerate(tokens):
            for right in tokens[idx + 1 :]:
                pairs[f"{left}__{right}"] += 1
    return dict(sorted(pairs.items()))


def _distribution_from_rows(rows: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        token = str(row.get(field_name) or "").strip()
        if token:
            counter[token] += 1
    return dict(sorted(counter.items()))


def _build_overlap_report(
    *,
    run_id: str,
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_selection_overlap_key(row)].append(row)
        overlap_groups = [group for group in groups.values() if len(group) > 1]
        example_groups: list[dict[str, Any]] = []
        for group in overlap_groups[:12]:
            example_groups.append(
                {
                    "overlap_key": _selection_overlap_key(group[0]),
                    "row_count": len(group),
                    "source_types": sorted(
                        {
                            str(row.get("source_type") or "").strip()
                            for row in group
                            if str(row.get("source_type") or "").strip()
                        }
                    ),
                    "source_variants": sorted(
                        {
                            str(row.get("source_variant") or "").strip()
                            for row in group
                            if str(row.get("source_variant") or "").strip()
                        }
                    ),
                    "failure_buckets": sorted(
                        {
                            str(row.get("failure_bucket") or "").strip()
                            for row in group
                            if str(row.get("failure_bucket") or "").strip()
                        }
                    ),
                    "episode_ids": sorted(
                        {
                            str(row.get("episode_id") or "").strip()
                            for row in group
                            if str(row.get("episode_id") or "").strip()
                        }
                    )[:6],
                }
            )
        return {
            "row_count": len(rows),
            "unique_underlying_episodes": len(groups),
            "overlap_group_count": len(overlap_groups),
            "overlap_row_count": int(sum(len(group) for group in overlap_groups)),
            "source_type_counts": _distribution_from_rows(rows, "source_type"),
            "source_variant_counts": _distribution_from_rows(rows, "source_variant"),
            "failure_bucket_counts": _distribution_from_rows(rows, "failure_bucket"),
            "source_type_overlap_pairs": _pair_counter_from_groups(overlap_groups, "source_type"),
            "source_variant_overlap_pairs": _pair_counter_from_groups(overlap_groups, "source_variant"),
            "failure_bucket_overlap_pairs": _pair_counter_from_groups(overlap_groups, "failure_bucket"),
            "overlap_examples": example_groups,
        }

    return {
        "schema": "p40_failure_source_overlap_v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "candidate_pool": _summarize(candidate_rows),
        "selected_pool": _summarize(selected_rows),
    }


def _build_overlap_markdown(report: dict[str, Any]) -> list[str]:
    candidate = report.get("candidate_pool") if isinstance(report.get("candidate_pool"), dict) else {}
    selected = report.get("selected_pool") if isinstance(report.get("selected_pool"), dict) else {}
    lines = [
        f"# Failure Source Overlap ({str(report.get('run_id') or '')})",
        "",
        "## Candidate Pool",
        f"- rows: `{_safe_int(candidate.get('row_count'), 0)}`",
        f"- unique underlying episodes: `{_safe_int(candidate.get('unique_underlying_episodes'), 0)}`",
        f"- overlap groups: `{_safe_int(candidate.get('overlap_group_count'), 0)}`",
        f"- overlap rows: `{_safe_int(candidate.get('overlap_row_count'), 0)}`",
        "",
        "## Selected Pool",
        f"- rows: `{_safe_int(selected.get('row_count'), 0)}`",
        f"- unique underlying episodes: `{_safe_int(selected.get('unique_underlying_episodes'), 0)}`",
        f"- overlap groups: `{_safe_int(selected.get('overlap_group_count'), 0)}`",
        f"- overlap rows: `{_safe_int(selected.get('overlap_row_count'), 0)}`",
        "",
        "## Selected Source Variants",
    ]
    for key, value in dict(selected.get("source_variant_counts") or {}).items():
        lines.append(f"- `{key}`: `{_safe_int(value, 0)}`")
    lines.append("")
    lines.append("## Selected Overlap Examples")
    for item in list(selected.get("overlap_examples") or [])[:8]:
        if not isinstance(item, dict):
            continue
        lines.append(
            "- `{key}` rows=`{rows}` variants=`{variants}` buckets=`{buckets}`".format(
                key=str(item.get("overlap_key") or ""),
                rows=_safe_int(item.get("row_count"), 0),
                variants=",".join(item.get("source_variants") or []),
                buckets=",".join(item.get("failure_buckets") or []),
            )
        )
    return lines


def _replay_weight_for(*, failure_types: set[str], score: float, low_threshold: float, risk_tags: list[str]) -> float:
    weight = 1.0
    if "champion_regression_segment" in failure_types:
        weight += 1.0
    if "high_risk_bucket_failure" in failure_types:
        weight += 0.75
    if {"invalid_action", "timeout", "execution_error"} & failure_types:
        weight += 0.5
    if "low_score_quantile" in failure_types:
        denom = max(1.0, abs(low_threshold) if low_threshold != 0.0 else 1.0)
        weight += min(1.0, max(0.0, (low_threshold - score) / denom))
    weight += 0.1 * float(len(risk_tags))
    return round(weight, 4)


def _selection_reason(*, failure_types: set[str], failure_bucket: str, risk_tags: list[str]) -> str:
    parts = [failure_bucket]
    parts.extend(sorted(failure_types))
    if risk_tags:
        parts.extend(f"risk:{tag}" for tag in risk_tags[:3])
    return ",".join(parts[:8])


def _build_markdown(
    *,
    run_id: str,
    status: str,
    arena_run_dir: Path | None,
    selected_total: int,
    counters_by_type: dict[str, int],
    counters_by_bucket: dict[str, int],
    counters_by_source: dict[str, int],
    counters_by_slice: dict[str, int],
    counters_by_risk: dict[str, int],
    counters_by_policy: dict[str, int],
    counters_by_seed: dict[str, int],
    replay_weight_summary: dict[str, float],
    scarce_buckets: list[str],
    warnings: list[str],
) -> list[str]:
    lines = [
        f"# P40 Failure Mining ({run_id})",
        "",
        f"- status: `{status}`",
        f"- arena_run_dir: `{str(arena_run_dir) if arena_run_dir else ''}`",
        f"- selected_failures: `{selected_total}`",
        "",
        "## Failure Type Distribution",
    ]
    if counters_by_type:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_type.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Failure Bucket Distribution"])
    if counters_by_bucket:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_bucket.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Source Distribution"])
    if counters_by_source:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_source.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Scarce Buckets"])
    if scarce_buckets:
        lines.extend([f"- {bucket}" for bucket in scarce_buckets])
    else:
        lines.append("- none")
    lines.extend(["", "## Slice Tag Coverage"])
    if counters_by_slice:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_slice.items(), key=lambda kv: (-kv[1], kv[0]))[:16]])
    else:
        lines.append("- none")
    lines.extend(["", "## Risk Tag Coverage"])
    if counters_by_risk:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_risk.items(), key=lambda kv: (-kv[1], kv[0]))[:16]])
    else:
        lines.append("- none")
    lines.extend(["", "## Policy Distribution"])
    if counters_by_policy:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_policy.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Seed Distribution"])
    if counters_by_seed:
        lines.extend([f"- {k}: {v}" for k, v in sorted(counters_by_seed.items(), key=lambda kv: (-kv[1], kv[0]))])
    else:
        lines.append("- none")
    lines.extend(["", "## Replay Weight Summary"])
    if replay_weight_summary:
        lines.extend([f"- {k}: {v}" for k, v in replay_weight_summary.items()])
    else:
        lines.append("- none")
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {w}" for w in warnings])
    return lines


def run_failure_mining(
    *,
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    run_id: str = "",
    quick: bool = False,
    dry_run: bool = False,
    arena_run_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    cfg: dict[str, Any] = {}
    cfg_path: Path | None = None
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        cfg = _read_yaml_or_json(cfg_path)

    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    artifacts_root = str(output_cfg.get("artifacts_root") or "docs/artifacts/p40/failure_mining")
    chosen_run_id = str(run_id or output_cfg.get("run_id") or now_stamp())
    if out_dir:
        run_dir = Path(out_dir)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
    else:
        run_dir = (repo_root / artifacts_root).resolve() / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_cfg = cfg.get("input") if isinstance(cfg.get("input"), dict) else {}
    p39_root_raw = str(input_cfg.get("p39_root") or "docs/artifacts/p39")
    p39_root = Path(p39_root_raw)
    if not p39_root.is_absolute():
        p39_root = (repo_root / p39_root).resolve()

    source_specs = _resolve_failure_sources(
        repo_root=repo_root,
        p39_root=p39_root,
        input_cfg=input_cfg,
        arena_run_dir_override=arena_run_dir_override,
    )

    criteria_cfg = cfg.get("criteria") if isinstance(cfg.get("criteria"), dict) else {}
    bottom_q = float(criteria_cfg.get("bottom_quantile") or 0.2)
    score_reg_threshold = float(criteria_cfg.get("champion_score_regression_ratio") or 0.05)
    high_risk_round_threshold = int(criteria_cfg.get("high_risk_round_threshold") or 2)
    max_failures = int(criteria_cfg.get("max_failures") or 1200)
    max_failures_per_type = int(criteria_cfg.get("max_failures_per_type") or 0)
    max_failures_per_seed = int(criteria_cfg.get("max_failures_per_seed") or 0)
    max_failures_per_bucket = _normalized_int_mapping(criteria_cfg.get("max_failures_per_bucket"))
    min_failures_per_bucket = _normalized_int_mapping(criteria_cfg.get("min_failures_per_bucket"))
    max_failures_per_source = _normalized_int_mapping(criteria_cfg.get("max_failures_per_source"))
    min_failures_per_source_type = _normalized_int_mapping(criteria_cfg.get("min_failures_per_source_type"))
    max_failures_per_source_variant = _normalized_int_mapping(criteria_cfg.get("max_failures_per_source_variant"))
    min_failures_per_source_variant = _normalized_int_mapping(criteria_cfg.get("min_failures_per_source_variant"))
    max_failures_per_overlap_key = int(criteria_cfg.get("max_failures_per_overlap_key") or 0)
    allow_overlap_variant_swaps = bool(criteria_cfg.get("allow_overlap_variant_swaps", False))
    overlap_swap_source_variants_allowlist = set(
        _normalized_string_list(criteria_cfg.get("overlap_swap_source_variants_allowlist"))
    )
    bucket_priority_weights = _normalized_float_mapping(criteria_cfg.get("bucket_priority_weights"))
    source_priority_weights = _normalized_float_mapping(criteria_cfg.get("source_priority_weights"))
    policy_priority_weights = _normalized_float_mapping(criteria_cfg.get("policy_priority_weights"))
    max_slice_gap_failures_per_source = int(criteria_cfg.get("max_slice_gap_failures_per_source") or 0)
    slice_gap_priority_weight = float(criteria_cfg.get("slice_gap_priority_weight") or 1.25)
    max_candidate_slice_failures_per_source = int(criteria_cfg.get("max_candidate_slice_failures_per_source") or 0)
    candidate_slice_priority_weight = float(criteria_cfg.get("candidate_slice_priority_weight") or 1.15)
    max_compound_slice_failures_per_source = int(criteria_cfg.get("max_compound_slice_failures_per_source") or 0)
    compound_slice_priority_weight = float(criteria_cfg.get("compound_slice_priority_weight") or 1.10)
    skip_unknown_slice_gap_tags = bool(criteria_cfg.get("skip_unknown_slice_gap_tags", True))

    quick_cfg = cfg.get("quick") if isinstance(cfg.get("quick"), dict) else {}
    max_episode_scan = int(quick_cfg.get("max_episode_scan") or 240) if quick else 0

    warnings: list[str] = []

    if not source_specs:
        manifest = {
            "schema": "p40_failure_pack_manifest_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": "stub",
            "reason": "failure mining sources unavailable",
            "paths": {},
            "failures": [],
            "warnings": warnings,
            "replay_jsonl_path": str(run_dir / "hard_failure_replay.jsonl"),
        }
        stats = {
            "schema": "p40_failure_pack_stats_v1",
            "generated_at": now_iso(),
            "run_id": chosen_run_id,
            "status": "stub",
            "selected_failures": 0,
            "by_type": {},
            "by_policy": {},
            "by_seed": {},
        }
        write_json(run_dir / "failure_pack_manifest.json", manifest)
        write_json(run_dir / "failure_pack_stats.json", stats)
        write_markdown(
            run_dir / "failure_pack_stats.md",
            _build_markdown(
                run_id=chosen_run_id,
                status="stub",
                arena_run_dir=None,
                selected_total=0,
                counters_by_type={},
                counters_by_bucket={},
                counters_by_source={},
                counters_by_slice={},
                counters_by_risk={},
                counters_by_policy={},
                counters_by_seed={},
                replay_weight_summary={},
                scarce_buckets=list(KNOWN_FAILURE_BUCKETS),
                warnings=warnings,
            ),
        )
        return {
            "status": "stub",
            "run_id": chosen_run_id,
            "run_dir": str(run_dir),
            "failure_pack_manifest": str(run_dir / "failure_pack_manifest.json"),
            "failure_pack_stats": str(run_dir / "failure_pack_stats.json"),
            "selected_failures": 0,
        }

    primary_arena_run_dir = Path(source_specs[0]["arena_run_dir"]) if source_specs else None
    candidate_policy = str(criteria_cfg.get("candidate_policy") or "").strip()
    champion_policy = str(criteria_cfg.get("champion_policy") or "").strip()
    candidate_mean_values: list[float] = []
    champion_mean_values: list[float] = []
    champion_regression_detected = False
    aggregate_scan_rows = 0
    aggregate_working_rows = 0
    source_summaries: list[dict[str, Any]] = []

    selected: list[dict[str, Any]] = []
    by_type: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_source_type: Counter[str] = Counter()
    by_source_variant: Counter[str] = Counter()
    by_slice: Counter[str] = Counter()
    by_risk: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    by_bucket_reason: Counter[str] = Counter()
    bucket_fail_counter: Counter[str] = Counter()
    selection_by_type: Counter[str] = Counter()
    selection_by_seed: Counter[str] = Counter()
    selection_by_bucket: Counter[str] = Counter()
    selection_by_source: Counter[str] = Counter()
    selection_by_source_type: Counter[str] = Counter()
    selection_by_source_variant: Counter[str] = Counter()
    selection_by_overlap_key: Counter[str] = Counter()
    selected_by_overlap_payload: dict[str, dict[str, Any]] = {}
    swap_events: list[dict[str, Any]] = []
    candidate_rows_for_selection: list[dict[str, Any]] = []

    for spec in source_specs:
        arena_run_dir = Path(spec["arena_run_dir"])
        if not arena_run_dir.exists():
            warnings.append(f"arena run directory unavailable: {arena_run_dir}")
            continue
        episode_records_path = Path(spec["episode_records_path"])
        summary_path = Path(spec["summary_path"])
        bucket_path = Path(spec["bucket_path"])
        candidate_decision_path = spec.get("candidate_decision_path")
        triage_report_path = spec.get("triage_report_path")
        slice_breakdown_path = spec.get("slice_breakdown_path")
        promotion_decision_path = spec.get("promotion_decision_path")

        rows = _read_jsonl_with_lineno(episode_records_path, max_rows=max_episode_scan)
        summary_rows = _load_summary_rows(summary_path)
        candidate_decision = (
            read_json(candidate_decision_path)
            if isinstance(candidate_decision_path, Path) and candidate_decision_path.exists()
            else None
        )
        promotion_decision = (
            read_json(promotion_decision_path)
            if isinstance(promotion_decision_path, Path) and promotion_decision_path.exists()
            else None
        )
        triage_report = (
            read_json(triage_report_path)
            if isinstance(triage_report_path, Path) and triage_report_path.exists()
            else None
        )
        slice_breakdown = (
            read_json(slice_breakdown_path)
            if isinstance(slice_breakdown_path, Path) and slice_breakdown_path.exists()
            else None
        )

        if not rows:
            warnings.append(f"episode records missing or empty: {episode_records_path}")
            continue

        source_candidate_policy = candidate_policy
        source_champion_policy = champion_policy
        if not source_candidate_policy and isinstance(candidate_decision, dict):
            source_candidate_policy = str(candidate_decision.get("candidate_policy_id") or "").strip()
        if not source_candidate_policy and isinstance(promotion_decision, dict):
            source_candidate_policy = str(promotion_decision.get("candidate_policy") or "").strip()
        if not source_champion_policy and isinstance(candidate_decision, dict):
            source_champion_policy = str(candidate_decision.get("champion_policy_id") or "").strip()
        if not source_champion_policy and isinstance(promotion_decision, dict):
            source_champion_policy = str(promotion_decision.get("champion_policy") or "").strip()
        if not source_candidate_policy and summary_rows:
            source_candidate_policy = str(summary_rows[0].get("policy_id") or "").strip()

        candidate_policy = candidate_policy or source_candidate_policy
        champion_policy = champion_policy or source_champion_policy

        candidate_row = _pick_policy_row(summary_rows, source_candidate_policy) if source_candidate_policy else None
        champion_row = _pick_policy_row(summary_rows, source_champion_policy) if source_champion_policy else None
        candidate_rows = (
            [r for r in rows if str(r.get("policy_id") or "").strip() == source_candidate_policy]
            if source_candidate_policy
            else list(rows)
        )
        working_rows = candidate_rows if candidate_rows else list(rows)
        if source_candidate_policy and not candidate_rows:
            warnings.append(f"candidate policy not present in episode records: {source_candidate_policy} @ {arena_run_dir}")

        scores = [_safe_float(r.get("total_score")) for r in working_rows]
        low_threshold = _quantile(scores, bottom_q) if scores else 0.0
        source_candidate_mean = _safe_float((candidate_row or {}).get("mean_total_score"), 0.0)
        source_champion_mean = _safe_float((champion_row or {}).get("mean_total_score"), 0.0)
        if candidate_row is not None:
            candidate_mean_values.append(source_candidate_mean)
        if champion_row is not None:
            champion_mean_values.append(source_champion_mean)
        source_regression = False
        if candidate_row is not None and champion_row is not None and source_champion_mean > 0.0:
            source_regression = source_candidate_mean < source_champion_mean * (1.0 - score_reg_threshold)
        elif isinstance(candidate_decision, dict):
            decision_token = str(candidate_decision.get("decision") or "").lower()
            if decision_token in {"hold", "reject"}:
                source_regression = True
        elif isinstance(promotion_decision, dict):
            decision_token = str(promotion_decision.get("recommendation") or "").lower()
            if decision_token in {"hold", "reject", "observe"}:
                source_regression = True
        champion_regression_detected = champion_regression_detected or source_regression
        slice_priority_weights = _build_slice_priority_weights(triage_report, slice_breakdown)
        aggregate_scan_rows += len(rows)
        aggregate_working_rows += len(working_rows)
        source_run_id = str(spec.get("source_run_id") or arena_run_dir.name)
        source_priority = source_priority_weights.get(source_run_id, source_priority_weights.get(str(spec.get("source_type") or ""), 1.0))
        source_summaries.append(
            {
                "source_run_id": source_run_id,
                "arena_run_dir": str(arena_run_dir),
                "episode_records": str(episode_records_path),
                "summary_table": str(summary_path),
                "candidate_decision_json": str(candidate_decision_path) if isinstance(candidate_decision_path, Path) else "",
                "triage_report_json": str(triage_report_path) if isinstance(triage_report_path, Path) else "",
                "slice_breakdown_json": str(slice_breakdown_path) if isinstance(slice_breakdown_path, Path) else "",
                "promotion_decision_json": str(promotion_decision_path) if isinstance(promotion_decision_path, Path) else "",
                "candidate_policy": source_candidate_policy,
                "champion_policy": source_champion_policy,
                "candidate_mean_total_score": source_candidate_mean,
                "champion_mean_total_score": source_champion_mean,
                "selected_scan_rows": len(rows),
                "working_rows": len(working_rows),
                "low_score_threshold": low_threshold,
                "champion_regression_detected": source_regression,
                "slice_priority_weights": slice_priority_weights,
            }
        )

        for row in working_rows:
            failure_types: set[str] = set()
            status = str(row.get("status") or "unknown").lower()
            error = str(row.get("error") or "").strip().lower()
            score = _safe_float(row.get("total_score"), 0.0)
            rounds_survived = _safe_int(row.get("rounds_survived"), 0)
            invalid_rate = _safe_float(row.get("invalid_action_rate"), 0.0)
            timeout_rate = _safe_float(row.get("timeout_rate"), 0.0)

            if status != "ok":
                failure_types.add("episode_failure_status")
            if invalid_rate > 0.0:
                failure_types.add("invalid_action")
            if timeout_rate > 0.0:
                failure_types.add("timeout")
            if "exception" in error or "failed" in error or "timeout" in error:
                failure_types.add("execution_error")
            if score <= low_threshold:
                failure_types.add("low_score_quantile")

            risk_counts: dict[str, int] = {}
            if isinstance(row.get("bucket_counts"), dict):
                raw_risk = (row.get("bucket_counts") or {}).get("risk")
                if isinstance(raw_risk, dict):
                    risk_counts = {str(k): _safe_int(v) for k, v in raw_risk.items()}
            if risk_counts.get("resource_tight", 0) > 0 and (rounds_survived <= high_risk_round_threshold or score <= low_threshold):
                failure_types.add("high_risk_bucket_failure")

            if source_regression and score <= source_candidate_mean:
                failure_types.add("champion_regression_segment")

            if not failure_types:
                continue

            slice_tags = _infer_slice_tags(row)
            triage_slice_hits = [tag for tag in slice_tags if tag in slice_priority_weights]
            if triage_slice_hits:
                failure_types.add("triage_degraded_slice")
            risk_tags = _infer_risk_tags(
                row,
                high_risk_round_threshold=high_risk_round_threshold,
                low_threshold=low_threshold,
            )
            bucket_payload = classify_failure_bucket(
                row=row,
                failure_types=failure_types,
                high_risk_round_threshold=high_risk_round_threshold,
                low_score_threshold=low_threshold,
            )
            failure_bucket = str(bucket_payload.get("failure_bucket") or "low_score_survival")
            bucket_reason = str(bucket_payload.get("bucket_reason") or "")
            failure_bucket_candidates = list(bucket_payload.get("failure_bucket_candidates") or [])
            failure_bucket_signals = list(bucket_payload.get("failure_bucket_signals") or [])
            failure_bucket, bucket_reason, failure_bucket_candidates, failure_bucket_signals = _refine_failure_bucket_for_slice_pressure(
                row=row,
                failure_types=failure_types,
                failure_bucket=failure_bucket,
                bucket_reason=bucket_reason,
                failure_bucket_candidates=failure_bucket_candidates,
                failure_bucket_signals=failure_bucket_signals,
            )
            bucket_priority = bucket_priority_weights.get(failure_bucket, 1.0)
            policy_priority = policy_priority_weights.get(source_candidate_policy, 1.0)
            triage_priority = max((slice_priority_weights.get(tag, 1.0) for tag in triage_slice_hits), default=1.0)
            replay_weight = _replay_weight_for(
                failure_types=failure_types,
                score=score,
                low_threshold=low_threshold,
                risk_tags=risk_tags,
            )
            replay_weight = round(replay_weight * max(0.1, source_priority) * max(0.1, bucket_priority) * max(0.1, policy_priority) * max(0.1, triage_priority), 4)

            if failure_bucket == "high_risk_collapse":
                for tag in risk_tags:
                    bucket_fail_counter[str(tag)] += 1

            episode_id = "{run}|{policy}|{seed}|{ep}".format(
                run=source_run_id,
                policy=str(row.get("policy_id") or ""),
                seed=str(row.get("seed") or ""),
                ep=_safe_int(row.get("episode_index"), 0),
            )
            payload = {
                "episode_id": episode_id,
                "policy_id": str(row.get("policy_id") or ""),
                "seed": str(row.get("seed") or ""),
                "episode_index": _safe_int(row.get("episode_index"), 0),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or ""),
                "total_score": score,
                "rounds_survived": rounds_survived,
                "invalid_action_rate": invalid_rate,
                "timeout_rate": timeout_rate,
                "failure_types": sorted(failure_types),
                "failure_bucket": failure_bucket,
                "bucket_reason": bucket_reason,
                "failure_bucket_candidates": failure_bucket_candidates,
                "failure_bucket_signals": failure_bucket_signals,
                "slice_tags": slice_tags,
                "risk_tags": risk_tags,
                "source_type": str(spec.get("source_type") or "arena_failure_mining"),
                "source_variant": _infer_source_variant(slice_tags=slice_tags, failure_bucket=failure_bucket),
                "selection_reason": _selection_reason(
                    failure_types=failure_types,
                    failure_bucket=failure_bucket,
                    risk_tags=risk_tags,
                ),
                "replay_weight": replay_weight,
                "source": {
                    "arena_run_dir": str(arena_run_dir),
                    "episode_records": str(episode_records_path),
                    "summary_table": str(summary_path),
                    "bucket_metrics": str(bucket_path),
                    "triage_report": str(triage_report_path) if isinstance(triage_report_path, Path) else "",
                    "slice_breakdown": str(slice_breakdown_path) if isinstance(slice_breakdown_path, Path) else "",
                    "candidate_decision": str(candidate_decision_path) if isinstance(candidate_decision_path, Path) else "",
                    "promotion_decision": str(promotion_decision_path) if isinstance(promotion_decision_path, Path) else "",
                    "line": _safe_int(row.get("_source_line"), 0),
                },
                "source_run_id": source_run_id,
                "source_campaign_id": str(cfg.get("campaign_id") or ""),
                "source_checkpoint_refs": {
                    "candidate_policy": source_candidate_policy,
                    "champion_policy": source_champion_policy,
                },
                "source_priority": source_priority,
                "bucket_priority": bucket_priority,
                "triage_slice_hits": triage_slice_hits,
                "triage_priority": triage_priority,
                "raw_episode": row,
            }
            candidate_rows_for_selection.append(payload)

        if (
            max_slice_gap_failures_per_source > 0
            or max_candidate_slice_failures_per_source > 0
            or max_compound_slice_failures_per_source > 0
        ) and (source_champion_policy or source_candidate_policy):
            gap_rows = sorted(
                [
                    row
                    for row in _load_slice_rows(triage_report) + _load_slice_rows(slice_breakdown)
                    if _safe_int(row.get("champion_count"), 0) > _safe_int(row.get("candidate_count"), 0)
                    and _safe_int(row.get("champion_count"), 0) > 0
                ],
                key=lambda item: (
                    -(_safe_int(item.get("champion_count"), 0) - _safe_int(item.get("candidate_count"), 0)),
                    _safe_float(((item.get("metrics") or {}) if isinstance(item.get("metrics"), dict) else {}).get("mean_total_score_delta"), 0.0),
                    str(item.get("slice_key") or ""),
                    str(item.get("slice_label") or ""),
                ),
            )
            added_compound_gap_rows = 0
            seen_compound_gap_tags: set[str] = set()

            def _append_compound_gap_rows(
                *,
                picked: dict[str, Any],
                source_run_id: str,
                replay_weight: float,
                source_priority: float,
                slice_tags: list[str],
                risk_tags: list[str],
                primary_slice_tag: str,
            ) -> None:
                nonlocal added_compound_gap_rows
                if max_compound_slice_failures_per_source <= 0:
                    return
                for compound_slice_tag in _compound_actionable_slice_tags(slice_tags, primary_tag=primary_slice_tag):
                    if added_compound_gap_rows >= max_compound_slice_failures_per_source:
                        break
                    normalized_compound_tag = str(compound_slice_tag).strip().lower()
                    if normalized_compound_tag in seen_compound_gap_tags:
                        continue
                    compound_bucket = _bucket_from_slice_tag(compound_slice_tag)
                    if not compound_bucket:
                        continue
                    compound_bucket_priority = bucket_priority_weights.get(compound_bucket, 1.0)
                    compound_replay_weight = round(
                        replay_weight
                        * max(0.1, compound_slice_priority_weight)
                        * max(0.1, compound_bucket_priority)
                        * 0.86,
                        4,
                    )
                    compound_episode_id = "{run}|compound_gap|{policy}|{seed}|{ep}|{tag}".format(
                        run=source_run_id,
                        policy=str(picked.get("policy_id") or ""),
                        seed=str(picked.get("seed") or ""),
                        ep=_safe_int(picked.get("episode_index"), 0),
                        tag=normalized_compound_tag.replace(":", "_"),
                    )
                    candidate_rows_for_selection.append(
                        {
                            "episode_id": compound_episode_id,
                            "policy_id": str(picked.get("policy_id") or ""),
                            "seed": str(picked.get("seed") or ""),
                            "episode_index": _safe_int(picked.get("episode_index"), 0),
                            "status": str(picked.get("status") or ""),
                            "error": str(picked.get("error") or ""),
                            "total_score": _safe_float(picked.get("total_score"), 0.0),
                            "rounds_survived": _safe_int(picked.get("rounds_survived"), 0),
                            "invalid_action_rate": _safe_float(picked.get("invalid_action_rate"), 0.0),
                            "timeout_rate": _safe_float(picked.get("timeout_rate"), 0.0),
                            "failure_types": ["compound_slice_failure_seed", "triage_degraded_slice"],
                            "failure_bucket": compound_bucket,
                            "bucket_reason": "compound_slice_failure_seed",
                            "failure_bucket_candidates": [compound_bucket],
                            "failure_bucket_signals": [f"compound_slice_gap:{compound_slice_tag}"],
                            "slice_tags": slice_tags,
                            "risk_tags": risk_tags,
                            "source_type": "arena_compound_slice_seed",
                            "source_variant": _preferred_source_variant(
                                primary_slice_tag=compound_slice_tag,
                                slice_tags=slice_tags,
                                failure_bucket=compound_bucket,
                            ),
                            "selection_reason": f"{compound_bucket},compound_slice_failure_seed,{compound_slice_tag}",
                            "replay_weight": compound_replay_weight,
                            "source": {
                                "arena_run_dir": str(arena_run_dir),
                                "episode_records": str(episode_records_path),
                                "summary_table": str(summary_path),
                                "triage_report": str(triage_report_path) if isinstance(triage_report_path, Path) else "",
                                "slice_breakdown": str(slice_breakdown_path) if isinstance(slice_breakdown_path, Path) else "",
                                "line": _safe_int(picked.get("_source_line"), 0),
                            },
                            "source_run_id": source_run_id,
                            "source_campaign_id": str(cfg.get("campaign_id") or ""),
                            "source_checkpoint_refs": {
                                "candidate_policy": source_candidate_policy,
                                "champion_policy": source_champion_policy,
                            },
                            "source_priority": source_priority,
                            "bucket_priority": compound_bucket_priority,
                            "triage_slice_hits": [compound_slice_tag],
                            "triage_priority": compound_slice_priority_weight,
                            "raw_episode": picked,
                        }
                    )
                    seen_compound_gap_tags.add(normalized_compound_tag)
                    added_compound_gap_rows += 1

            if max_slice_gap_failures_per_source > 0 and source_champion_policy:
                added_gap_rows = 0
                seen_gap_tags: set[str] = set()
                for gap_row in gap_rows:
                    if added_gap_rows >= max_slice_gap_failures_per_source:
                        break
                    slice_tag = _slice_tag_from_row(gap_row)
                    if not slice_tag or slice_tag in seen_gap_tags:
                        continue
                    bucket = _bucket_from_slice_tag(slice_tag)
                    if skip_unknown_slice_gap_tags and not bucket:
                        continue
                    if not bucket:
                        bucket = "low_score_survival"
                    bucket_priority = bucket_priority_weights.get(bucket, 1.0)
                    champion_gap = max(
                        1,
                        _safe_int(gap_row.get("champion_count"), 0) - _safe_int(gap_row.get("candidate_count"), 0),
                    )
                    score_delta = _safe_float(
                        ((gap_row.get("metrics") or {}) if isinstance(gap_row.get("metrics"), dict) else {}).get("mean_total_score_delta"),
                        0.0,
                    )
                    matching_rows = [
                        row
                        for row in rows
                        if str(row.get("policy_id") or "").strip() == source_champion_policy
                        and slice_tag in _infer_slice_tags(row)
                    ]
                    if not matching_rows:
                        continue
                    matching_rows.sort(
                        key=lambda row: (
                            -_safe_float(row.get("total_score"), 0.0),
                            -_safe_int(row.get("rounds_survived"), 0),
                            str(row.get("seed") or ""),
                        )
                    )
                    picked = matching_rows[0]
                    slice_tags = _infer_slice_tags(picked)
                    risk_tags = _infer_risk_tags(
                        picked,
                        high_risk_round_threshold=high_risk_round_threshold,
                        low_threshold=low_threshold,
                    )
                    replay_weight = round(
                        (1.5 + min(1.5, 0.2 * float(champion_gap)) + min(1.0, abs(score_delta) / 200.0))
                        * max(0.1, source_priority)
                        * max(0.1, bucket_priority)
                        * max(0.1, slice_gap_priority_weight),
                        4,
                    )
                    episode_id = "{run}|gap|{policy}|{seed}|{ep}".format(
                        run=source_run_id,
                        policy=str(picked.get("policy_id") or ""),
                        seed=str(picked.get("seed") or ""),
                        ep=_safe_int(picked.get("episode_index"), 0),
                    )
                    candidate_rows_for_selection.append(
                        {
                            "episode_id": episode_id,
                            "policy_id": str(picked.get("policy_id") or ""),
                            "seed": str(picked.get("seed") or ""),
                            "episode_index": _safe_int(picked.get("episode_index"), 0),
                            "status": str(picked.get("status") or ""),
                            "error": str(picked.get("error") or ""),
                            "total_score": _safe_float(picked.get("total_score"), 0.0),
                            "rounds_survived": _safe_int(picked.get("rounds_survived"), 0),
                            "invalid_action_rate": _safe_float(picked.get("invalid_action_rate"), 0.0),
                            "timeout_rate": _safe_float(picked.get("timeout_rate"), 0.0),
                            "failure_types": ["slice_coverage_gap_seed", "triage_degraded_slice"],
                            "failure_bucket": bucket,
                            "bucket_reason": "slice_coverage_gap_seed",
                            "failure_bucket_candidates": [bucket],
                            "failure_bucket_signals": [f"slice_gap:{slice_tag}"],
                            "slice_tags": slice_tags,
                            "risk_tags": risk_tags,
                            "source_type": "arena_slice_gap_seed",
                            "source_variant": _preferred_source_variant(
                                primary_slice_tag=slice_tag,
                                slice_tags=slice_tags,
                                failure_bucket=bucket,
                            ),
                            "selection_reason": f"{bucket},slice_coverage_gap_seed,{slice_tag}",
                            "replay_weight": replay_weight,
                            "source": {
                                "arena_run_dir": str(arena_run_dir),
                                "episode_records": str(episode_records_path),
                                "summary_table": str(summary_path),
                                "triage_report": str(triage_report_path) if isinstance(triage_report_path, Path) else "",
                                "slice_breakdown": str(slice_breakdown_path) if isinstance(slice_breakdown_path, Path) else "",
                                "line": _safe_int(picked.get("_source_line"), 0),
                            },
                            "source_run_id": source_run_id,
                            "source_campaign_id": str(cfg.get("campaign_id") or ""),
                            "source_checkpoint_refs": {
                                "candidate_policy": source_candidate_policy,
                                "champion_policy": source_champion_policy,
                            },
                            "source_priority": source_priority,
                            "bucket_priority": bucket_priority,
                            "triage_slice_hits": [slice_tag],
                            "triage_priority": slice_gap_priority_weight,
                            "raw_episode": picked,
                        }
                    )
                    seen_gap_tags.add(slice_tag)
                    added_gap_rows += 1
                    if added_compound_gap_rows < max_compound_slice_failures_per_source:
                        _append_compound_gap_rows(
                            picked=picked,
                            source_run_id=source_run_id,
                            replay_weight=replay_weight,
                            source_priority=source_priority,
                            slice_tags=slice_tags,
                            risk_tags=risk_tags,
                            primary_slice_tag=slice_tag,
                        )

            if (
                max_compound_slice_failures_per_source > 0
                and source_champion_policy
                and max_slice_gap_failures_per_source <= 0
            ):
                for gap_row in gap_rows:
                    if added_compound_gap_rows >= max_compound_slice_failures_per_source:
                        break
                    slice_tag = _slice_tag_from_row(gap_row)
                    if not slice_tag:
                        continue
                    bucket = _bucket_from_slice_tag(slice_tag)
                    if skip_unknown_slice_gap_tags and not bucket:
                        continue
                    if not bucket:
                        bucket = "low_score_survival"
                    bucket_priority = bucket_priority_weights.get(bucket, 1.0)
                    champion_gap = max(
                        1,
                        _safe_int(gap_row.get("champion_count"), 0) - _safe_int(gap_row.get("candidate_count"), 0),
                    )
                    score_delta = _safe_float(
                        ((gap_row.get("metrics") or {}) if isinstance(gap_row.get("metrics"), dict) else {}).get("mean_total_score_delta"),
                        0.0,
                    )
                    matching_rows = [
                        row
                        for row in rows
                        if str(row.get("policy_id") or "").strip() == source_champion_policy
                        and slice_tag in _infer_slice_tags(row)
                    ]
                    if not matching_rows:
                        continue
                    matching_rows.sort(
                        key=lambda row: (
                            -_safe_float(row.get("total_score"), 0.0),
                            -_safe_int(row.get("rounds_survived"), 0),
                            str(row.get("seed") or ""),
                        )
                    )
                    picked = matching_rows[0]
                    slice_tags = _infer_slice_tags(picked)
                    risk_tags = _infer_risk_tags(
                        picked,
                        high_risk_round_threshold=high_risk_round_threshold,
                        low_threshold=low_threshold,
                    )
                    replay_weight = round(
                        (1.5 + min(1.5, 0.2 * float(champion_gap)) + min(1.0, abs(score_delta) / 200.0))
                        * max(0.1, source_priority)
                        * max(0.1, bucket_priority)
                        * max(0.1, slice_gap_priority_weight),
                        4,
                    )
                    _append_compound_gap_rows(
                        picked=picked,
                        source_run_id=source_run_id,
                        replay_weight=replay_weight,
                        source_priority=source_priority,
                        slice_tags=slice_tags,
                        risk_tags=risk_tags,
                        primary_slice_tag=slice_tag,
                    )

            if max_candidate_slice_failures_per_source > 0 and source_candidate_policy:
                added_candidate_gap_rows = 0
                seen_candidate_gap_tags: set[str] = set()
                for gap_row in gap_rows:
                    if added_candidate_gap_rows >= max_candidate_slice_failures_per_source:
                        break
                    slice_tag = _slice_tag_from_row(gap_row)
                    if not slice_tag or slice_tag in seen_candidate_gap_tags:
                        continue
                    bucket = _bucket_from_slice_tag(slice_tag)
                    if skip_unknown_slice_gap_tags and not bucket:
                        continue
                    if not bucket:
                        bucket = "low_score_survival"
                    bucket_priority = bucket_priority_weights.get(bucket, 1.0)
                    champion_gap = max(
                        1,
                        _safe_int(gap_row.get("champion_count"), 0) - _safe_int(gap_row.get("candidate_count"), 0),
                    )
                    score_delta = _safe_float(
                        ((gap_row.get("metrics") or {}) if isinstance(gap_row.get("metrics"), dict) else {}).get("mean_total_score_delta"),
                        0.0,
                    )
                    matching_rows = [
                        row
                        for row in rows
                        if str(row.get("policy_id") or "").strip() == source_candidate_policy
                        and slice_tag in _infer_slice_tags(row)
                    ]
                    if not matching_rows:
                        continue
                    matching_rows.sort(
                        key=lambda row: (
                            _safe_float(row.get("total_score"), 0.0),
                            _safe_int(row.get("rounds_survived"), 0),
                            -_safe_float(row.get("invalid_action_rate"), 0.0),
                            str(row.get("seed") or ""),
                        )
                    )
                    picked = matching_rows[0]
                    slice_tags = _infer_slice_tags(picked)
                    risk_tags = _infer_risk_tags(
                        picked,
                        high_risk_round_threshold=high_risk_round_threshold,
                        low_threshold=low_threshold,
                    )
                    replay_weight = round(
                        (1.2 + min(1.2, 0.18 * float(champion_gap)) + min(0.9, abs(score_delta) / 220.0))
                        * max(0.1, source_priority)
                        * max(0.1, bucket_priority)
                        * max(0.1, candidate_slice_priority_weight),
                        4,
                    )
                    episode_id = "{run}|candidate_gap|{policy}|{seed}|{ep}".format(
                        run=source_run_id,
                        policy=str(picked.get("policy_id") or ""),
                        seed=str(picked.get("seed") or ""),
                        ep=_safe_int(picked.get("episode_index"), 0),
                    )
                    candidate_rows_for_selection.append(
                        {
                            "episode_id": episode_id,
                            "policy_id": str(picked.get("policy_id") or ""),
                            "seed": str(picked.get("seed") or ""),
                            "episode_index": _safe_int(picked.get("episode_index"), 0),
                            "status": str(picked.get("status") or ""),
                            "error": str(picked.get("error") or ""),
                            "total_score": _safe_float(picked.get("total_score"), 0.0),
                            "rounds_survived": _safe_int(picked.get("rounds_survived"), 0),
                            "invalid_action_rate": _safe_float(picked.get("invalid_action_rate"), 0.0),
                            "timeout_rate": _safe_float(picked.get("timeout_rate"), 0.0),
                            "failure_types": ["candidate_slice_failure_seed", "triage_degraded_slice"],
                            "failure_bucket": bucket,
                            "bucket_reason": "candidate_slice_failure_seed",
                            "failure_bucket_candidates": [bucket],
                            "failure_bucket_signals": [f"candidate_slice_gap:{slice_tag}"],
                            "slice_tags": slice_tags,
                            "risk_tags": risk_tags,
                            "source_type": "arena_candidate_slice_seed",
                            "source_variant": _preferred_source_variant(
                                primary_slice_tag=slice_tag,
                                slice_tags=slice_tags,
                                failure_bucket=bucket,
                            ),
                            "selection_reason": f"{bucket},candidate_slice_failure_seed,{slice_tag}",
                            "replay_weight": replay_weight,
                            "source": {
                                "arena_run_dir": str(arena_run_dir),
                                "episode_records": str(episode_records_path),
                                "summary_table": str(summary_path),
                                "triage_report": str(triage_report_path) if isinstance(triage_report_path, Path) else "",
                                "slice_breakdown": str(slice_breakdown_path) if isinstance(slice_breakdown_path, Path) else "",
                                "line": _safe_int(picked.get("_source_line"), 0),
                            },
                            "source_run_id": source_run_id,
                            "source_campaign_id": str(cfg.get("campaign_id") or ""),
                            "source_checkpoint_refs": {
                                "candidate_policy": source_candidate_policy,
                                "champion_policy": source_champion_policy,
                            },
                            "source_priority": source_priority,
                            "bucket_priority": bucket_priority,
                            "triage_slice_hits": [slice_tag],
                            "triage_priority": candidate_slice_priority_weight,
                            "raw_episode": picked,
                        }
                    )
                    seen_candidate_gap_tags.add(slice_tag)
                    added_candidate_gap_rows += 1

    candidate_rows_for_selection.sort(
        key=lambda item: (
            -_safe_float(item.get("replay_weight"), 0.0),
            _safe_float(item.get("total_score"), 0.0),
            str(item.get("source_run_id") or ""),
            str(item.get("episode_id") or ""),
        )
    )

    selected_ids: set[str] = set()

    def _record_selected(payload: dict[str, Any]) -> None:
        selected.append(payload)
        selected_ids.add(str(payload.get("episode_id") or ""))
        selection_by_seed[str(payload.get("seed") or "")] += 1
        selection_by_bucket[str(payload.get("failure_bucket") or "")] += 1
        selection_by_source[str(payload.get("source_run_id") or "")] += 1
        selection_by_source_type[str(payload.get("source_type") or "")] += 1
        if str(payload.get("source_variant") or "").strip():
            selection_by_source_variant[str(payload.get("source_variant") or "").strip()] += 1
        overlap_key = _selection_overlap_key(payload)
        if overlap_key:
            selection_by_overlap_key[overlap_key] += 1
            selected_by_overlap_payload[overlap_key] = payload
        for token in payload.get("failure_types") if isinstance(payload.get("failure_types"), list) else []:
            by_type[str(token)] += 1
            selection_by_type[str(token)] += 1
        by_bucket[str(payload.get("failure_bucket") or "unknown")] += 1
        by_source[str(payload.get("source_run_id") or "unknown")] += 1
        by_source_type[str(payload.get("source_type") or "unknown")] += 1
        if str(payload.get("source_variant") or "").strip():
            by_source_variant[str(payload.get("source_variant") or "").strip()] += 1
        by_bucket_reason[str(payload.get("bucket_reason") or "unknown")] += 1
        for tag in payload.get("slice_tags") if isinstance(payload.get("slice_tags"), list) else []:
            by_slice[str(tag)] += 1
        for tag in payload.get("risk_tags") if isinstance(payload.get("risk_tags"), list) else []:
            by_risk[str(tag)] += 1
        by_policy[str(payload.get("policy_id") or "")] += 1
        by_seed[str(payload.get("seed") or "")] += 1

    def _remove_selected(payload: dict[str, Any]) -> None:
        episode_id = str(payload.get("episode_id") or "")
        if episode_id:
            selected_ids.discard(episode_id)
            for index, row in enumerate(selected):
                if str(row.get("episode_id") or "") == episode_id:
                    selected.pop(index)
                    break
        _decrement_counter(selection_by_seed, str(payload.get("seed") or ""))
        _decrement_counter(selection_by_bucket, str(payload.get("failure_bucket") or ""))
        _decrement_counter(selection_by_source, str(payload.get("source_run_id") or ""))
        _decrement_counter(selection_by_source_type, str(payload.get("source_type") or ""))
        _decrement_counter(selection_by_source_variant, str(payload.get("source_variant") or ""))
        overlap_key = _selection_overlap_key(payload)
        if overlap_key:
            _decrement_counter(selection_by_overlap_key, overlap_key)
            if selection_by_overlap_key.get(overlap_key, 0) <= 0:
                selected_by_overlap_payload.pop(overlap_key, None)
        for token in payload.get("failure_types") if isinstance(payload.get("failure_types"), list) else []:
            _decrement_counter(selection_by_type, str(token))
            _decrement_counter(by_type, str(token))
        _decrement_counter(by_bucket, str(payload.get("failure_bucket") or "unknown"))
        _decrement_counter(by_source, str(payload.get("source_run_id") or "unknown"))
        _decrement_counter(by_source_type, str(payload.get("source_type") or "unknown"))
        if str(payload.get("source_variant") or "").strip():
            _decrement_counter(by_source_variant, str(payload.get("source_variant") or "").strip())
        _decrement_counter(by_bucket_reason, str(payload.get("bucket_reason") or "unknown"))
        for tag in payload.get("slice_tags") if isinstance(payload.get("slice_tags"), list) else []:
            _decrement_counter(by_slice, str(tag))
        for tag in payload.get("risk_tags") if isinstance(payload.get("risk_tags"), list) else []:
            _decrement_counter(by_risk, str(tag))
        _decrement_counter(by_policy, str(payload.get("policy_id") or ""))
        _decrement_counter(by_seed, str(payload.get("seed") or ""))

    if min_failures_per_bucket:
        for payload in candidate_rows_for_selection:
            bucket = str(payload.get("failure_bucket") or "")
            target = _safe_int(min_failures_per_bucket.get(bucket), 0)
            if target <= 0 or selection_by_bucket[bucket] >= target:
                continue
            if _row_selection_matches_policy(
                payload=payload,
                selection_by_type=selection_by_type,
                selection_by_seed=selection_by_seed,
                selection_by_bucket=selection_by_bucket,
                selection_by_source=selection_by_source,
                selection_by_source_type=selection_by_source_type,
                selection_by_source_variant=selection_by_source_variant,
                selection_by_overlap_key=selection_by_overlap_key,
                max_failures_per_type=max_failures_per_type,
                max_failures_per_seed=max_failures_per_seed,
                max_failures_per_bucket=max_failures_per_bucket,
                max_failures_per_source=max_failures_per_source,
                max_failures_per_source_variant=max_failures_per_source_variant,
                max_failures_per_overlap_key=max_failures_per_overlap_key,
            ):
                _record_selected(payload)
            else:
                swapped = False
                overlap_key = _selection_overlap_key(payload)
                existing_payload = selected_by_overlap_payload.get(overlap_key)
                source_variant = str(payload.get("source_variant") or "")
                if (
                    allow_overlap_variant_swaps
                    and existing_payload is not None
                    and (not overlap_swap_source_variants_allowlist or not source_variant or source_variant in overlap_swap_source_variants_allowlist)
                    and _can_swap_overlap_selected_row(
                        candidate_payload=payload,
                        existing_payload=existing_payload,
                        selection_by_type=selection_by_type,
                        selection_by_seed=selection_by_seed,
                        selection_by_bucket=selection_by_bucket,
                        selection_by_source=selection_by_source,
                        selection_by_source_type=selection_by_source_type,
                        selection_by_source_variant=selection_by_source_variant,
                        selection_by_overlap_key=selection_by_overlap_key,
                        max_failures_per_type=max_failures_per_type,
                        max_failures_per_seed=max_failures_per_seed,
                        max_failures_per_bucket=max_failures_per_bucket,
                        max_failures_per_source=max_failures_per_source,
                        max_failures_per_source_variant=max_failures_per_source_variant,
                        max_failures_per_overlap_key=max_failures_per_overlap_key,
                        min_failures_per_bucket=min_failures_per_bucket,
                        min_failures_per_source_type=min_failures_per_source_type,
                        min_failures_per_source_variant=min_failures_per_source_variant,
                    )
                ):
                    removed_variant = str(existing_payload.get("source_variant") or "")
                    _remove_selected(existing_payload)
                    _record_selected(payload)
                    swap_events.append(
                        {
                            "overlap_key": overlap_key,
                            "removed_episode_id": str(existing_payload.get("episode_id") or ""),
                            "removed_source_variant": removed_variant,
                            "added_episode_id": str(payload.get("episode_id") or ""),
                            "added_source_variant": source_variant,
                            "reason": f"bucket_minimum:{bucket}",
                        }
                    )
                    swapped = True
                if not swapped:
                    continue
            if len(selected) >= max(1, max_failures):
                warnings.append(f"failure cap reached while satisfying bucket minimums: {max_failures}")
                break

    if min_failures_per_source_type and len(selected) < max(1, max_failures):
        for payload in candidate_rows_for_selection:
            source_type = str(payload.get("source_type") or "")
            target = _safe_int(min_failures_per_source_type.get(source_type), 0)
            if target <= 0 or selection_by_source_type[source_type] >= target:
                continue
            if _row_selection_matches_policy(
                payload=payload,
                selection_by_type=selection_by_type,
                selection_by_seed=selection_by_seed,
                selection_by_bucket=selection_by_bucket,
                selection_by_source=selection_by_source,
                selection_by_source_type=selection_by_source_type,
                selection_by_source_variant=selection_by_source_variant,
                selection_by_overlap_key=selection_by_overlap_key,
                max_failures_per_type=max_failures_per_type,
                max_failures_per_seed=max_failures_per_seed,
                max_failures_per_bucket=max_failures_per_bucket,
                max_failures_per_source=max_failures_per_source,
                max_failures_per_source_variant=max_failures_per_source_variant,
                max_failures_per_overlap_key=max_failures_per_overlap_key,
            ):
                _record_selected(payload)
            else:
                swapped = False
                overlap_key = _selection_overlap_key(payload)
                existing_payload = selected_by_overlap_payload.get(overlap_key)
                if (
                    allow_overlap_variant_swaps
                    and existing_payload is not None
                    and _can_swap_overlap_selected_row(
                        candidate_payload=payload,
                        existing_payload=existing_payload,
                        selection_by_type=selection_by_type,
                        selection_by_seed=selection_by_seed,
                        selection_by_bucket=selection_by_bucket,
                        selection_by_source=selection_by_source,
                        selection_by_source_type=selection_by_source_type,
                        selection_by_source_variant=selection_by_source_variant,
                        selection_by_overlap_key=selection_by_overlap_key,
                        max_failures_per_type=max_failures_per_type,
                        max_failures_per_seed=max_failures_per_seed,
                        max_failures_per_bucket=max_failures_per_bucket,
                        max_failures_per_source=max_failures_per_source,
                        max_failures_per_source_variant=max_failures_per_source_variant,
                        max_failures_per_overlap_key=max_failures_per_overlap_key,
                        min_failures_per_bucket=min_failures_per_bucket,
                        min_failures_per_source_type=min_failures_per_source_type,
                        min_failures_per_source_variant=min_failures_per_source_variant,
                    )
                ):
                    removed_source_type = str(existing_payload.get("source_type") or "")
                    removed_variant = str(existing_payload.get("source_variant") or "")
                    _remove_selected(existing_payload)
                    _record_selected(payload)
                    swap_events.append(
                        {
                            "overlap_key": overlap_key,
                            "removed_episode_id": str(existing_payload.get("episode_id") or ""),
                            "removed_source_type": removed_source_type,
                            "removed_source_variant": removed_variant,
                            "added_episode_id": str(payload.get("episode_id") or ""),
                            "added_source_type": source_type,
                            "added_source_variant": str(payload.get("source_variant") or ""),
                            "reason": f"source_type_minimum:{source_type}",
                        }
                    )
                    swapped = True
                if not swapped:
                    continue
            if len(selected) >= max(1, max_failures):
                warnings.append(f"failure cap reached while satisfying source-type minimums: {max_failures}")
                break

    if min_failures_per_source_variant and len(selected) < max(1, max_failures):
        for payload in candidate_rows_for_selection:
            source_variant = str(payload.get("source_variant") or "")
            target = _safe_int(min_failures_per_source_variant.get(source_variant), 0)
            if target <= 0 or selection_by_source_variant[source_variant] >= target:
                continue
            if _row_selection_matches_policy(
                payload=payload,
                selection_by_type=selection_by_type,
                selection_by_seed=selection_by_seed,
                selection_by_bucket=selection_by_bucket,
                selection_by_source=selection_by_source,
                selection_by_source_type=selection_by_source_type,
                selection_by_source_variant=selection_by_source_variant,
                selection_by_overlap_key=selection_by_overlap_key,
                max_failures_per_type=max_failures_per_type,
                max_failures_per_seed=max_failures_per_seed,
                max_failures_per_bucket=max_failures_per_bucket,
                max_failures_per_source=max_failures_per_source,
                max_failures_per_source_variant=max_failures_per_source_variant,
                max_failures_per_overlap_key=max_failures_per_overlap_key,
            ):
                _record_selected(payload)
            else:
                swapped = False
                overlap_key = _selection_overlap_key(payload)
                existing_payload = selected_by_overlap_payload.get(overlap_key)
                if (
                    allow_overlap_variant_swaps
                    and existing_payload is not None
                    and (not overlap_swap_source_variants_allowlist or source_variant in overlap_swap_source_variants_allowlist)
                    and _can_swap_overlap_selected_row(
                        candidate_payload=payload,
                        existing_payload=existing_payload,
                        selection_by_type=selection_by_type,
                        selection_by_seed=selection_by_seed,
                        selection_by_bucket=selection_by_bucket,
                        selection_by_source=selection_by_source,
                        selection_by_source_type=selection_by_source_type,
                        selection_by_source_variant=selection_by_source_variant,
                        selection_by_overlap_key=selection_by_overlap_key,
                        max_failures_per_type=max_failures_per_type,
                        max_failures_per_seed=max_failures_per_seed,
                        max_failures_per_bucket=max_failures_per_bucket,
                        max_failures_per_source=max_failures_per_source,
                        max_failures_per_source_variant=max_failures_per_source_variant,
                        max_failures_per_overlap_key=max_failures_per_overlap_key,
                        min_failures_per_bucket=min_failures_per_bucket,
                        min_failures_per_source_type=min_failures_per_source_type,
                        min_failures_per_source_variant=min_failures_per_source_variant,
                    )
                ):
                    removed_variant = str(existing_payload.get("source_variant") or "")
                    _remove_selected(existing_payload)
                    _record_selected(payload)
                    swap_events.append(
                        {
                            "overlap_key": overlap_key,
                            "removed_episode_id": str(existing_payload.get("episode_id") or ""),
                            "removed_source_variant": removed_variant,
                            "added_episode_id": str(payload.get("episode_id") or ""),
                            "added_source_variant": source_variant,
                            "reason": f"source_variant_minimum:{source_variant}",
                        }
                    )
                    swapped = True
                if not swapped:
                    continue
            if len(selected) >= max(1, max_failures):
                warnings.append(f"failure cap reached while satisfying source-variant minimums: {max_failures}")
                break

    for payload in candidate_rows_for_selection:
        if str(payload.get("episode_id") or "") in selected_ids:
            continue
        if not _row_selection_matches_policy(
            payload=payload,
            selection_by_type=selection_by_type,
            selection_by_seed=selection_by_seed,
            selection_by_bucket=selection_by_bucket,
            selection_by_source=selection_by_source,
            selection_by_source_type=selection_by_source_type,
            selection_by_source_variant=selection_by_source_variant,
            selection_by_overlap_key=selection_by_overlap_key,
            max_failures_per_type=max_failures_per_type,
            max_failures_per_seed=max_failures_per_seed,
            max_failures_per_bucket=max_failures_per_bucket,
            max_failures_per_source=max_failures_per_source,
            max_failures_per_source_variant=max_failures_per_source_variant,
            max_failures_per_overlap_key=max_failures_per_overlap_key,
        ):
            continue
        _record_selected(payload)
        if len(selected) >= max(1, max_failures):
            warnings.append(f"failure cap reached: {max_failures}")
            break

    overlap_report = _build_overlap_report(
        run_id=chosen_run_id,
        candidate_rows=candidate_rows_for_selection,
        selected_rows=selected,
    )
    overlap_report_json = run_dir / "overlap_report.json"
    overlap_report_md = run_dir / "overlap_report.md"
    write_json(overlap_report_json, overlap_report)
    write_markdown(overlap_report_md, _build_overlap_markdown(overlap_report))

    candidate_mean = round(sum(candidate_mean_values) / max(1, len(candidate_mean_values)), 4) if candidate_mean_values else 0.0
    champion_mean = round(sum(champion_mean_values) / max(1, len(champion_mean_values)), 4) if champion_mean_values else 0.0
    low_threshold = round(sum(_safe_float(item.get("low_score_threshold"), 0.0) for item in source_summaries) / max(1, len(source_summaries)), 4)

    replay_jsonl_path = run_dir / "hard_failure_replay.jsonl"
    if not dry_run:
        with replay_jsonl_path.open("w", encoding="utf-8", newline="\n") as fp:
            for item in selected:
                # Keep replay pack concise but still reusable for downstream mixers.
                replay_row = {
                    "schema": "p40_hard_failure_row_v1",
                    "episode_id": item["episode_id"],
                    "policy_id": item["policy_id"],
                    "seed": item["seed"],
                    "episode_index": item["episode_index"],
                    "failure_types": item["failure_types"],
                    "failure_bucket": item["failure_bucket"],
                    "bucket_reason": item["bucket_reason"],
                    "failure_bucket_candidates": list(item.get("failure_bucket_candidates") or []),
                    "failure_bucket_signals": list(item.get("failure_bucket_signals") or []),
                    "slice_tags": item["slice_tags"],
                    "risk_tags": item["risk_tags"],
                    "source_type": item["source_type"],
                    "source_variant": str(item.get("source_variant") or ""),
                    "selection_reason": item["selection_reason"],
                    "replay_weight": item["replay_weight"],
                    "selected_for_training": True,
                    "status": item["status"],
                    "error": item["error"],
                    "total_score": item["total_score"],
                    "rounds_survived": item["rounds_survived"],
                    "invalid_action_rate": item["invalid_action_rate"],
                    "timeout_rate": item["timeout_rate"],
                    "source": item["source"],
                    "episode_record": item["raw_episode"],
                }
                fp.write(json.dumps(replay_row, ensure_ascii=False) + "\n")

    status = "ok" if selected else "stub"
    manifest = {
        "schema": "p40_failure_pack_manifest_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "config_path": str(cfg_path) if cfg_path else "",
        "arena_run_dir": str(primary_arena_run_dir) if primary_arena_run_dir else "",
        "source_count": len(source_summaries),
        "sources": source_summaries,
        "criteria": {
            "bottom_quantile": bottom_q,
            "champion_score_regression_ratio": score_reg_threshold,
            "high_risk_round_threshold": high_risk_round_threshold,
            "max_failures": max_failures,
            "max_failures_per_type": max_failures_per_type,
            "max_failures_per_seed": max_failures_per_seed,
            "max_failures_per_bucket": max_failures_per_bucket,
            "min_failures_per_bucket": min_failures_per_bucket,
            "max_failures_per_source": max_failures_per_source,
            "min_failures_per_source_type": min_failures_per_source_type,
            "max_failures_per_overlap_key": max_failures_per_overlap_key,
            "allow_overlap_variant_swaps": allow_overlap_variant_swaps,
            "overlap_swap_source_variants_allowlist": sorted(overlap_swap_source_variants_allowlist),
            "max_compound_slice_failures_per_source": max_compound_slice_failures_per_source,
            "compound_slice_priority_weight": compound_slice_priority_weight,
            "bucket_priority_weights": bucket_priority_weights,
            "source_priority_weights": source_priority_weights,
            "policy_priority_weights": policy_priority_weights,
        },
        "inputs": {
            "p39_root": str(p39_root),
            "source_run_ids": [str(item.get("source_run_id") or "") for item in source_summaries],
            "episode_records": [str(item.get("episode_records") or "") for item in source_summaries],
            "summary_table": [str(item.get("summary_table") or "") for item in source_summaries],
            "candidate_decision": [str(item.get("candidate_decision_json") or "") for item in source_summaries],
            "triage_report_json": [str(item.get("triage_report_json") or "") for item in source_summaries],
            "slice_breakdown_json": [str(item.get("slice_breakdown_json") or "") for item in source_summaries],
        },
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
        "candidate_mean_total_score": candidate_mean,
        "champion_mean_total_score": champion_mean,
        "champion_regression_detected": champion_regression_detected,
        "low_score_threshold": low_threshold,
        "selected_count": len(selected),
        "overlap_report_json": str(overlap_report_json),
        "overlap_report_md": str(overlap_report_md),
        "swap_events": swap_events,
        "failures": [
            {
                "episode_id": item["episode_id"],
                "policy_id": item["policy_id"],
                "seed": item["seed"],
                "episode_index": item["episode_index"],
                "failure_types": item["failure_types"],
                "failure_bucket": item["failure_bucket"],
                "bucket_reason": item["bucket_reason"],
                "failure_bucket_candidates": list(item.get("failure_bucket_candidates") or []),
                "failure_bucket_signals": list(item.get("failure_bucket_signals") or []),
                "slice_tags": item["slice_tags"],
                "risk_tags": item["risk_tags"],
                "source_type": item["source_type"],
                "source_variant": str(item.get("source_variant") or ""),
                "selection_reason": item["selection_reason"],
                "replay_weight": item["replay_weight"],
                "total_score": item["total_score"],
                "status": item["status"],
                "error": item["error"],
                "source": item["source"],
                "source_run_id": item["source_run_id"],
                "source_campaign_id": item["source_campaign_id"],
                "source_checkpoint_refs": item["source_checkpoint_refs"],
                "source_priority": item["source_priority"],
                "bucket_priority": item["bucket_priority"],
                "triage_slice_hits": list(item.get("triage_slice_hits") or []),
                "triage_priority": item["triage_priority"],
            }
            for item in selected
        ],
        "replay_jsonl_path": str(replay_jsonl_path),
        "warnings": warnings,
    }
    stats = {
        "schema": "p40_failure_pack_stats_v1",
        "generated_at": now_iso(),
        "run_id": chosen_run_id,
        "status": status,
        "selected_failures": len(selected),
        "by_type": dict(sorted(by_type.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_source_type": dict(sorted(by_source_type.items())),
        "by_source_variant": dict(sorted(by_source_variant.items())),
        "candidate_pool_source_type_counts": dict(
            sorted(
                Counter(
                    str(item.get("source_type") or "").strip()
                    for item in candidate_rows_for_selection
                    if str(item.get("source_type") or "").strip()
                ).items()
            )
        ),
        "candidate_pool_source_variant_counts": dict(
            sorted(
                Counter(
                    str(item.get("source_variant") or "").strip()
                    for item in candidate_rows_for_selection
                    if str(item.get("source_variant") or "").strip()
                ).items()
            )
        ),
        "candidate_pool_bucket_counts": dict(
            sorted(
                Counter(
                    str(item.get("failure_bucket") or "").strip()
                    for item in candidate_rows_for_selection
                    if str(item.get("failure_bucket") or "").strip()
                ).items()
            )
        ),
        "known_failure_buckets": list(KNOWN_FAILURE_BUCKETS),
        "scarce_buckets": scarce_failure_buckets(by_bucket),
        "by_slice_tag": dict(sorted(by_slice.items())),
        "by_risk_tag": dict(sorted(by_risk.items())),
        "by_policy": dict(sorted(by_policy.items())),
        "by_seed": dict(sorted(by_seed.items())),
        "by_bucket_reason": dict(sorted(by_bucket_reason.items())),
        "risk_bucket_failures": dict(sorted(bucket_fail_counter.items())),
        "replay_weight": {
            "mean": round(sum(_safe_float(item.get("replay_weight"), 0.0) for item in selected) / max(1, len(selected)), 4),
            "max": round(max((_safe_float(item.get("replay_weight"), 0.0) for item in selected), default=0.0), 4),
            "min": round(min((_safe_float(item.get("replay_weight"), 0.0) for item in selected), default=0.0), 4),
        },
        "scan_rows": aggregate_scan_rows,
        "working_rows": aggregate_working_rows,
        "candidate_policy": candidate_policy,
        "champion_policy": champion_policy,
        "overlap_report_json": str(overlap_report_json),
        "overlap_report_md": str(overlap_report_md),
        "swap_event_count": len(swap_events),
        "swap_events": swap_events,
    }
    md_lines = _build_markdown(
        run_id=chosen_run_id,
        status=status,
        arena_run_dir=primary_arena_run_dir,
        selected_total=len(selected),
        counters_by_type=dict(by_type),
        counters_by_bucket=dict(by_bucket),
        counters_by_source=dict(by_source),
        counters_by_slice=dict(by_slice),
        counters_by_risk=dict(by_risk),
        counters_by_policy=dict(by_policy),
        counters_by_seed=dict(by_seed),
        replay_weight_summary=stats.get("replay_weight") if isinstance(stats.get("replay_weight"), dict) else {},
        scarce_buckets=list(stats.get("scarce_buckets") or []),
        warnings=warnings,
    )

    write_json(run_dir / "failure_pack_manifest.json", manifest)
    write_json(run_dir / "failure_pack_stats.json", stats)
    write_json(run_dir / "failure_bucket_stats.json", stats)
    write_markdown(run_dir / "failure_pack_stats.md", md_lines)
    write_markdown(run_dir / "failure_bucket_stats.md", md_lines)

    return {
        "status": status,
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "failure_pack_manifest": str(run_dir / "failure_pack_manifest.json"),
        "failure_pack_stats": str(run_dir / "failure_pack_stats.json"),
        "failure_bucket_stats": str(run_dir / "failure_bucket_stats.json"),
        "replay_jsonl_path": str(replay_jsonl_path),
        "selected_failures": len(selected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P40 failure mining from P39 arena artifacts.")
    parser.add_argument("--config", default="", help="Optional YAML/JSON config path")
    parser.add_argument("--out-dir", default="", help="Optional explicit output directory")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id")
    parser.add_argument("--arena-run-dir", default="", help="Optional explicit arena run directory")
    parser.add_argument("--quick", action="store_true", help="Use small scan budget")
    parser.add_argument("--dry-run", action="store_true", help="Plan-only mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_failure_mining(
        config_path=(args.config if str(args.config).strip() else None),
        out_dir=(args.out_dir if str(args.out_dir).strip() else None),
        run_id=str(args.run_id or ""),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
        arena_run_dir_override=(args.arena_run_dir if str(args.arena_run_dir).strip() else None),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if str(summary.get("status")) in {"ok", "stub"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
