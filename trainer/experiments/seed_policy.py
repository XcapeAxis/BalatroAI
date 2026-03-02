from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency in some local venvs
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REQUIRED_SEED_SETS = (
    "contract_regression",
    "perf_gate_100",
    "milestone_500",
    "milestone_1000",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_seed_policy(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("seed policy config must be a mapping")
    return payload


def _deterministic_letter_seed(source: str, *, width: int, alphabet: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    num = int(digest, 16)
    out: list[str] = []
    base = len(alphabet)
    for _ in range(width):
        out.append(alphabet[num % base])
        num //= base
    return "".join(out)


def _hash_seeds(seeds: list[str]) -> str:
    blob = json.dumps(seeds, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _get_generation_rule(policy: dict[str, Any]) -> dict[str, Any]:
    rule = policy.get("generation_rule")
    if not isinstance(rule, dict):
        raise ValueError("generation_rule must be a mapping")
    alphabet = str(rule.get("alphabet") or DEFAULT_ALPHABET).strip()
    if not alphabet:
        raise ValueError("generation_rule.alphabet must not be empty")
    width = int(rule.get("width") or 7)
    if width <= 0:
        raise ValueError("generation_rule.width must be > 0")
    rule = dict(rule)
    rule["alphabet"] = alphabet
    rule["width"] = width
    return rule


def _materialize_generated_set(
    *,
    set_name: str,
    generated_cfg: dict[str, Any],
    policy_version: str,
    rule: dict[str, Any],
) -> list[str]:
    count = int(generated_cfg.get("count") or 0)
    if count <= 0:
        raise ValueError(f"seed_sets.{set_name}.generated.count must be > 0")
    salt = str(generated_cfg.get("salt") or set_name)
    template = str(
        generated_cfg.get("template")
        or "{policy_version}|{set_name}|{salt}|{index}|{nonce}"
    )

    seeds: list[str] = []
    used: set[str] = set()
    idx = 0
    nonce = 0
    while len(seeds) < count:
        source = template.format(
            policy_version=policy_version,
            set_name=set_name,
            salt=salt,
            index=idx,
            nonce=nonce,
        )
        candidate = _deterministic_letter_seed(
            source,
            width=int(rule["width"]),
            alphabet=str(rule["alphabet"]),
        )
        idx += 1
        if candidate in used:
            nonce += 1
            continue
        used.add(candidate)
        seeds.append(candidate)
    return seeds


def _resolve_seed_set(
    policy: dict[str, Any],
    set_name: str,
) -> tuple[list[str], dict[str, Any]]:
    sets = policy.get("seed_sets")
    if not isinstance(sets, dict):
        raise ValueError("seed_sets must be a mapping")
    item = sets.get(set_name)
    if item is None:
        raise ValueError(f"unknown seed set: {set_name}")
    if not isinstance(item, dict):
        raise ValueError(f"seed_sets.{set_name} must be a mapping")

    version = str(policy.get("seed_policy_version") or "p23.v1")
    rule = _get_generation_rule(policy)

    fixed = item.get("fixed")
    if fixed is not None:
        if not isinstance(fixed, list) or not fixed:
            raise ValueError(f"seed_sets.{set_name}.fixed must be a non-empty list")
        seeds = [str(s) for s in fixed]
        return seeds, {"source": "fixed", "count": len(seeds)}

    generated = item.get("generated")
    if not isinstance(generated, dict):
        raise ValueError(f"seed_sets.{set_name} must define fixed or generated")
    seeds = _materialize_generated_set(
        set_name=set_name,
        generated_cfg=generated,
        policy_version=version,
        rule=rule,
    )
    return seeds, {
        "source": "generated",
        "count": len(seeds),
        "generated": generated,
    }


def validate_seed_policy(policy: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    version = str(policy.get("seed_policy_version") or "").strip()
    if not version:
        issues.append("seed_policy_version is required")

    disallow_single = policy.get("disallow_single_seed_default")
    if not isinstance(disallow_single, bool):
        issues.append("disallow_single_seed_default must be boolean")

    try:
        _get_generation_rule(policy)
    except Exception as exc:  # pragma: no cover
        issues.append(str(exc))

    sets = policy.get("seed_sets")
    if not isinstance(sets, dict):
        issues.append("seed_sets must be a mapping")
        sets = {}

    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for set_name in REQUIRED_SEED_SETS:
        try:
            seeds, _meta = _resolve_seed_set(policy, set_name)
            counts[set_name] = len(seeds)
            hashes[set_name] = _hash_seeds(seeds)
            if len(seeds) != len(set(seeds)):
                issues.append(f"{set_name} contains duplicated seeds")
            if set_name == "perf_gate_100" and len(seeds) != 100:
                issues.append("perf_gate_100 must contain 100 seeds")
            if set_name == "milestone_500" and len(seeds) != 500:
                issues.append("milestone_500 must contain 500 seeds")
            if set_name == "milestone_1000" and len(seeds) != 1000:
                issues.append("milestone_1000 must contain 1000 seeds")
            if set_name == "contract_regression" and len(seeds) < 8:
                issues.append("contract_regression should contain at least 8 seeds")
        except Exception as exc:
            issues.append(str(exc))

    nightly = policy.get("nightly_extra_random")
    if not isinstance(nightly, dict):
        issues.append("nightly_extra_random must be a mapping")
    else:
        if int(nightly.get("default_count") or 0) <= 0:
            issues.append("nightly_extra_random.default_count must be > 0")
        from_set = str(nightly.get("from_set") or "")
        if not from_set:
            issues.append("nightly_extra_random.from_set must be set")

    return {
        "schema": "p23_seed_policy_validation_v1",
        "validated_at": now_iso(),
        "seed_policy_version": version,
        "ok": len(issues) == 0,
        "issues": issues,
        "set_counts": counts,
        "set_hashes": hashes,
    }


def materialize_seed_set(
    policy: dict[str, Any],
    set_name: str,
    *,
    explicit_single_seed_override: bool = False,
) -> dict[str, Any]:
    seeds, meta = _resolve_seed_set(policy, set_name)
    disallow_single = bool(policy.get("disallow_single_seed_default"))
    if disallow_single and len(seeds) == 1 and not explicit_single_seed_override:
        raise ValueError(
            f"implicit single-seed default is disallowed by policy for set {set_name}"
        )
    return {
        "schema": "p23_seeds_used_v1",
        "generated_at": now_iso(),
        "seed_policy_version": str(policy.get("seed_policy_version") or "p23.v1"),
        "seed_set_name": set_name,
        "seed_hash": _hash_seeds(seeds),
        "seed_count": len(seeds),
        "seeds": seeds,
        "metadata": meta,
    }


def materialize_nightly_seed_set(
    policy: dict[str, Any],
    *,
    git_commit: str,
    date_bucket: str,
    run_id: str,
    extra_count_override: int | None = None,
    explicit_single_seed_override: bool = False,
) -> dict[str, Any]:
    nightly = policy.get("nightly_extra_random")
    if not isinstance(nightly, dict):
        raise ValueError("nightly_extra_random must be configured")

    base_set = str(nightly.get("from_set") or "perf_gate_100")
    base_payload = materialize_seed_set(
        policy,
        base_set,
        explicit_single_seed_override=explicit_single_seed_override,
    )
    base_seeds = list(base_payload["seeds"])
    existing: set[str] = set(base_seeds)

    count = int(extra_count_override if extra_count_override is not None else (nightly.get("default_count") or 0))
    if count <= 0:
        raise ValueError("nightly extra seed count must be > 0")
    salt = str(nightly.get("salt") or "nightly")

    rule = _get_generation_rule(policy)
    template = str(
        nightly.get("template")
        or "{policy_version}|nightly|{salt}|{git_commit}|{date_bucket}|{run_id}|{index}|{nonce}"
    )
    extras: list[str] = []
    idx = 0
    nonce = 0
    while len(extras) < count:
        source = template.format(
            policy_version=str(policy.get("seed_policy_version") or "p23.v1"),
            salt=salt,
            git_commit=git_commit or "unknown",
            date_bucket=date_bucket,
            run_id=run_id,
            index=idx,
            nonce=nonce,
        )
        candidate = _deterministic_letter_seed(
            source,
            width=int(rule["width"]),
            alphabet=str(rule["alphabet"]),
        )
        idx += 1
        if candidate in existing:
            nonce += 1
            continue
        existing.add(candidate)
        extras.append(candidate)

    seeds = base_seeds + extras
    return {
        "schema": "p23_seeds_used_v1",
        "generated_at": now_iso(),
        "seed_policy_version": str(policy.get("seed_policy_version") or "p23.v1"),
        "seed_set_name": f"{base_set}+nightly_extra_random",
        "seed_hash": _hash_seeds(seeds),
        "seed_count": len(seeds),
        "seeds": seeds,
        "metadata": {
            "source": "nightly_materialized",
            "base_set": base_set,
            "base_seed_count": len(base_seeds),
            "nightly_extra_count": len(extras),
            "nightly_extra_random": {
                "git_commit": git_commit or "unknown",
                "date_bucket": date_bucket,
                "run_id": run_id,
                "salt": salt,
            },
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
