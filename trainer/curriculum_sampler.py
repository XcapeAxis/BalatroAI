from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cfg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            raise RuntimeError(f"failed to parse config: {path}") from exc


def _load_failure_counts(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts = payload.get("counts") if isinstance(payload, dict) else {}
    if not isinstance(counts, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in counts.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _normalize(weights: dict[str, float], floor: float) -> dict[str, float]:
    clipped = {k: max(float(v), floor) for k, v in weights.items()}
    s = sum(clipped.values())
    if s <= 0:
        n = max(1, len(clipped))
        return {k: 1.0 / n for k in clipped}
    return {k: v / s for k, v in clipped.items()}


def _stage_for_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    schedule = cfg.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return {"name": "default", "mode": mode, "stake_weight_overrides": {}}
    for row in schedule:
        if isinstance(row, dict) and str(row.get("mode")) == mode:
            return row
    return schedule[0] if isinstance(schedule[0], dict) else {"name": "default", "mode": mode, "stake_weight_overrides": {}}


def build_plan(cfg: dict[str, Any], failure_counts: dict[str, float], mode: str) -> dict[str, Any]:
    stakes = [str(x) for x in (cfg.get("stakes") or [])]
    base = cfg.get("sampling_weights") if isinstance(cfg.get("sampling_weights"), dict) else {}
    weights: dict[str, float] = {s: float(base.get(s, 1.0)) for s in stakes}

    stage = _stage_for_mode(cfg, mode)
    overrides = stage.get("stake_weight_overrides") if isinstance(stage, dict) else {}
    if isinstance(overrides, dict):
        for k, v in overrides.items():
            try:
                weights[str(k)] = float(v)
            except Exception:
                continue

    prio = cfg.get("failure_bucket_priority") if isinstance(cfg.get("failure_bucket_priority"), dict) else {}
    if failure_counts and prio:
        total_fail = sum(max(0.0, float(v)) for v in failure_counts.values())
        severity = 0.0
        if total_fail > 0:
            for bucket, cnt in failure_counts.items():
                severity += (float(cnt) / total_fail) * float(prio.get(bucket, 1.0))
        # Increase harder stakes when severity is high.
        if "gold" in weights:
            weights["gold"] = float(weights["gold"]) * (1.0 + 0.5 * severity)
        if "orange" in weights:
            weights["orange"] = float(weights["orange"]) * (1.0 + 0.25 * severity)

    floor = float(cfg.get("min_sampling_floor") or 0.05)
    probs = _normalize(weights, floor=floor)
    return {
        "schema": "p18_curriculum_plan_v1",
        "generated_at": _now_iso(),
        "mode": mode,
        "stage": stage,
        "stakes": stakes,
        "sampling_probabilities": probs,
        "ante_buckets": cfg.get("ante_buckets") or [],
        "boss_buckets": cfg.get("boss_buckets") or [],
        "failure_counts": failure_counts,
        "promotion_rule": cfg.get("promotion_rule") or {},
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build P18 curriculum sampling plan.")
    p.add_argument("--config", default="trainer/config/p18_curriculum.yaml")
    p.add_argument("--failure-input", default="")
    p.add_argument("--mode", choices=["smoke", "full", "stage0_easy", "stage1_mixed", "stage2_gold_failure"], default="smoke")
    p.add_argument("--out", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    cfg = _load_cfg(Path(args.config))
    failure_counts = _load_failure_counts(Path(args.failure_input)) if args.failure_input else {}
    mode = "stage0_easy" if args.mode == "smoke" else ("stage2_gold_failure" if args.mode == "full" else args.mode)
    plan = build_plan(cfg, failure_counts, mode=mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out), "mode": mode}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
