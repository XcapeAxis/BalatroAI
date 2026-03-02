"""Generate P29 weakness-targeted dataset with bucket/source composition summaries."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer import action_space, action_space_shop
from trainer.dataset import validate_record
from trainer.features_shop import SHOP_CONTEXT_DIM

from .targeted_sampling import (
    build_bucket_weights,
    build_teacher_weights,
    derive_ante_weights,
    derive_phase_plan,
    derive_stake_weights,
    weighted_pick,
)

try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rand_vec(rng: random.Random, n: int, *, lo: float = 0.0, hi: float = 1.0) -> list[float]:
    return [round(rng.uniform(lo, hi), 6) for _ in range(n)]


def _build_features(rng: random.Random, hand_size: int) -> dict[str, Any]:
    max_hand = action_space.MAX_HAND
    ranks = [rng.randint(1, 13) for _ in range(hand_size)] + [0] * (max_hand - hand_size)
    suits = [rng.randint(1, 4) for _ in range(hand_size)] + [0] * (max_hand - hand_size)
    chip_hint = _rand_vec(rng, hand_size, lo=0.0, hi=2.0) + [0.0] * (max_hand - hand_size)
    pad = [1.0] * hand_size + [0.0] * (max_hand - hand_size)
    return {
        "card_rank_ids": ranks,
        "card_suit_ids": suits,
        "card_chip_hint": chip_hint,
        "card_has_enhancement": [float(rng.randint(0, 1)) for _ in range(max_hand)],
        "card_has_edition": [float(rng.randint(0, 1)) for _ in range(max_hand)],
        "card_has_seal": [float(rng.randint(0, 1)) for _ in range(max_hand)],
        "hand_pad_mask": pad,
        "context": _rand_vec(rng, 12, lo=-0.5, hi=1.0),
    }


def _sample_legal_ids(rng: random.Random, max_actions: int, *, k_min: int, k_max: int) -> list[int]:
    k = rng.randint(k_min, min(k_max, max_actions))
    picks = rng.sample(list(range(max_actions)), k=k)
    return sorted(int(x) for x in picks)


def _ante_value(ante_tier: str, rng: random.Random) -> int:
    if ante_tier == "high":
        return rng.randint(6, 8)
    if ante_tier == "mid":
        return rng.randint(3, 5)
    return rng.randint(1, 2)


def generate_dataset(
    *,
    cfg: dict[str, Any],
    weakness_report: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    seed = int(cfg.get("seed") or 29)
    rng = random.Random(seed)

    target = cfg.get("target_size") if isinstance(cfg.get("target_size"), dict) else {}
    hand_target = int(target.get("hand_records") or 5200)
    shop_target = int(target.get("shop_records") or 1600)

    bucket_weights = build_bucket_weights(weakness_report, top_n=int(cfg.get("weakness_top_n") or 10))
    teacher_weights = build_teacher_weights(cfg)
    phase_plan = derive_phase_plan(cfg)
    stake_weights = derive_stake_weights(cfg)
    ante_weights = derive_ante_weights(cfg)

    records: list[dict[str, Any]] = []
    source_comp: dict[str, int] = {}
    bucket_cov: dict[str, int] = {}
    phase_counts = {"SELECTING_HAND": 0, "SHOP": 0}
    invalid_rows: list[dict[str, Any]] = []

    base_url = str(cfg.get("base_url") or "sim://p29-targeted")
    instance_prefix = str(cfg.get("instance_prefix") or "p29")

    def push_record(rec: dict[str, Any], source: str, bucket: str) -> None:
        try:
            validate_record(rec)
        except Exception as exc:
            if len(invalid_rows) < 20:
                invalid_rows.append({"error": str(exc), "phase": rec.get("phase"), "bucket_id": bucket})
            return
        records.append(rec)
        source_comp[source] = source_comp.get(source, 0) + 1
        bucket_cov[bucket] = bucket_cov.get(bucket, 0) + 1
        phase = str(rec.get("phase") or "")
        if phase in phase_counts:
            phase_counts[phase] += 1

    # Generate hand records.
    for idx in range(hand_target):
        bucket = weighted_pick(bucket_weights, rng)
        source = weighted_pick(teacher_weights, rng)
        stake = weighted_pick(stake_weights, rng)
        ante_tier = weighted_pick(ante_weights, rng)
        hand_size = rng.randint(4, action_space.MAX_HAND)
        features = _build_features(rng, hand_size)
        legal_ids = _sample_legal_ids(rng, action_space.max_actions(), k_min=4, k_max=12)
        expert = int(rng.choice(legal_ids))

        rec = {
            "schema": "record_v1",
            "timestamp": now_iso(),
            "episode_id": f"{instance_prefix}-h-{idx // 8}",
            "step_id": idx,
            "instance_id": f"{instance_prefix}-hand-{idx}",
            "base_url": base_url,
            "phase": "SELECTING_HAND",
            "done": False,
            "stake": stake,
            "ante_num": _ante_value(ante_tier, rng),
            "hand_size": hand_size,
            "legal_action_ids": legal_ids,
            "expert_action_id": expert,
            "macro_action": "play_hand",
            "reward": round(rng.uniform(-0.15, 0.95), 6),
            "reward_info": {
                "bucket_id": bucket,
                "teacher_source": source,
                "ante_tier": ante_tier,
                "replay_mode": "failure_state_replay" if source == "failure_replay" else "direct",
            },
            "features": features,
            "shop_legal_action_ids": None,
            "shop_expert_action_id": None,
            "shop_features": None,
        }
        push_record(rec, source, bucket)

    # Generate shop records.
    for idx in range(shop_target):
        bucket = weighted_pick(bucket_weights, rng)
        source = weighted_pick(teacher_weights, rng)
        stake = weighted_pick(stake_weights, rng)
        ante_tier = weighted_pick(ante_weights, rng)
        hand_size = rng.randint(0, action_space.MAX_HAND)
        features = _build_features(rng, max(1, hand_size if hand_size > 0 else 1))
        legal_ids = _sample_legal_ids(rng, action_space.max_actions(), k_min=1, k_max=4)
        shop_legal = _sample_legal_ids(rng, action_space_shop.max_actions(), k_min=2, k_max=10)
        shop_expert = int(rng.choice(shop_legal))
        shop_ctx = _rand_vec(rng, SHOP_CONTEXT_DIM, lo=-0.5, hi=1.0)

        rec = {
            "schema": "record_v1",
            "timestamp": now_iso(),
            "episode_id": f"{instance_prefix}-s-{idx // 6}",
            "step_id": idx,
            "instance_id": f"{instance_prefix}-shop-{idx}",
            "base_url": base_url,
            "phase": "SHOP",
            "done": False,
            "stake": stake,
            "ante_num": _ante_value(ante_tier, rng),
            "hand_size": hand_size,
            "legal_action_ids": legal_ids,
            "expert_action_id": None,
            "macro_action": "shop_update",
            "reward": round(rng.uniform(-0.20, 0.80), 6),
            "reward_info": {
                "bucket_id": bucket,
                "teacher_source": source,
                "ante_tier": ante_tier,
                "phase_balanced": bool(phase_plan.get("SHOP", 0.25) > 0),
            },
            "features": features,
            "shop_legal_action_ids": shop_legal,
            "shop_expert_action_id": shop_expert,
            "shop_features": {"shop_context": shop_ctx},
        }
        push_record(rec, source, bucket)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "schema": "p29_targeted_dataset_summary_v1",
        "generated_at": now_iso(),
        "dataset_path": str(out_path.resolve()),
        "total_records": len(records),
        "hand_records": int(phase_counts.get("SELECTING_HAND") or 0),
        "shop_records": int(phase_counts.get("SHOP") or 0),
        "source_composition": source_comp,
        "bucket_coverage": bucket_cov,
        "invalid_rows": {
            "count": len(invalid_rows),
            "examples": invalid_rows,
        },
        "sampling_snapshot": {
            "seed": seed,
            "bucket_weights": [{"bucket_id": x.key, "weight": x.weight} for x in bucket_weights],
            "teacher_weights": [{"source": x.key, "weight": x.weight} for x in teacher_weights],
            "phase_balance": phase_plan,
            "stake_mix": [{"stake": x.key, "weight": x.weight} for x in stake_weights],
            "ante_tier_mix": [{"tier": x.key, "weight": x.weight} for x in ante_weights],
        },
        "top_buckets_used": sorted(bucket_cov.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# P29 Targeted Dataset Summary",
        "",
        f"- dataset_path: `{summary.get('dataset_path')}`",
        f"- total_records: `{summary.get('total_records')}`",
        f"- hand_records: `{summary.get('hand_records')}`",
        f"- shop_records: `{summary.get('shop_records')}`",
        f"- invalid_rows: `{(summary.get('invalid_rows') or {}).get('count')}`",
        "",
        "## Top Buckets",
    ]
    for item in summary.get("top_buckets_used") or []:
        bucket, count = item
        lines.append(f"- {bucket}: {count}")
    lines += ["", "## Source Composition"]
    src = summary.get("source_composition") if isinstance(summary.get("source_composition"), dict) else {}
    for key in sorted(src.keys()):
        lines.append(f"- {key}: {src[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate weakness-targeted dataset for P29")
    p.add_argument("--config", required=True)
    p.add_argument("--weakness-report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--artifacts-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    report_path = Path(args.weakness_report).resolve()
    out_path = Path(args.out).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_mapping(cfg_path)
    weakness_report = read_json(report_path)
    summary = generate_dataset(cfg=cfg, weakness_report=weakness_report, out_path=out_path)
    summary["config_path"] = str(cfg_path)
    summary["weakness_report_path"] = str(report_path)

    summary_json = artifacts_dir / "p29_targeted_v1_summary.json"
    summary_md = artifacts_dir / "p29_targeted_v1_summary.md"
    write_json(summary_json, summary)
    write_summary_md(summary_md, summary)

    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset": str(out_path),
                "hand_records": summary.get("hand_records"),
                "shop_records": summary.get("shop_records"),
                "invalid_rows": (summary.get("invalid_rows") or {}).get("count"),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

