from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.closed_loop.replay_manifest import write_json, write_markdown
from trainer.hybrid.controller_registry import build_controller_registry, discover_policy_model_path, discover_world_model_checkpoint
from trainer.hybrid.routing_features import run_routing_feature_smoke


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _budget_rank(level: str) -> int:
    token = str(level or "low").lower()
    if token == "high":
        return 2
    if token == "medium":
        return 1
    return 0


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


class RuleBasedHybridRouter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.config = {
            "policy_margin_high": _safe_float(cfg.get("policy_margin_high"), 0.20),
            "policy_margin_low": _safe_float(cfg.get("policy_margin_low"), 0.08),
            "policy_entropy_high": _safe_float(cfg.get("policy_entropy_high"), 1.25),
            "wm_uncertainty_prefer_max": _safe_float(cfg.get("wm_uncertainty_prefer_max"), 0.80),
            "wm_uncertainty_disable": _safe_float(cfg.get("wm_uncertainty_disable"), 1.20),
            "search_budget_min_level": str(cfg.get("search_budget_min_level") or "medium"),
            "high_risk_search_bonus": _safe_float(cfg.get("high_risk_search_bonus"), 0.75),
            "late_stage_search_bonus": _safe_float(cfg.get("late_stage_search_bonus"), 0.50),
            "wm_reward_bonus_scale": _safe_float(cfg.get("wm_reward_bonus_scale"), 0.20),
        }

    def describe(self) -> dict[str, Any]:
        return {"schema": "p48_router_config_v1", "config": dict(self.config)}

    def route(
        self,
        *,
        features: dict[str, Any],
        available_controllers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        budget_rank = _budget_rank(str(features.get("budget_level") or "low"))
        required_budget = _budget_rank(str(self.config.get("search_budget_min_level") or "medium"))
        policy_available = bool(features.get("policy_available"))
        heuristic_available = bool(features.get("heuristic_available"))
        search_available = bool(features.get("search_available")) and budget_rank >= required_budget
        wm_available = bool(features.get("wm_available")) and bool(features.get("policy_available"))

        margin = _safe_float(features.get("policy_margin"), 0.0)
        entropy = _safe_float(features.get("policy_entropy"), 0.0)
        wm_uncertainty = _safe_float(features.get("wm_uncertainty"), 1.0)
        wm_predicted_return = _safe_float(features.get("wm_predicted_return"), 0.0)
        stage = str(features.get("slice_stage") or "unknown")
        resource_pressure = str(features.get("slice_resource_pressure") or "unknown")
        phase = str(features.get("phase") or "UNKNOWN")

        high_confidence = margin >= _safe_float(self.config.get("policy_margin_high"), 0.20) and entropy <= _safe_float(
            self.config.get("policy_entropy_high"), 1.25
        )
        low_confidence = margin <= _safe_float(self.config.get("policy_margin_low"), 0.08) or entropy >= _safe_float(
            self.config.get("policy_entropy_high"), 1.25
        )
        high_risk = resource_pressure == "high" or stage == "late"
        wm_disabled = not wm_available or wm_uncertainty >= _safe_float(self.config.get("wm_uncertainty_disable"), 1.20)
        wm_preferred = not wm_disabled and wm_uncertainty <= _safe_float(self.config.get("wm_uncertainty_prefer_max"), 0.80)

        controller_scores: dict[str, float] = {
            "policy_baseline": (-999.0 if not policy_available else 0.55 + (margin * 1.8) - (entropy * 0.10)),
            "policy_plus_wm_rerank": (
                -999.0
                if wm_disabled
                else 0.80 + (margin * 1.2) + (_safe_float(self.config.get("wm_reward_bonus_scale"), 0.20) * wm_predicted_return) - (wm_uncertainty * 0.60)
            ),
            "search_baseline": (
                -999.0
                if not search_available
                else 0.35
                + (0.70 if low_confidence else 0.0)
                + (_safe_float(self.config.get("high_risk_search_bonus"), 0.75) if high_risk else 0.0)
                + (_safe_float(self.config.get("late_stage_search_bonus"), 0.50) if stage == "late" else 0.0)
                + (0.15 if phase == "SELECTING_HAND" else -0.10)
            ),
            "heuristic_baseline": (0.25 if heuristic_available else -999.0) + (0.10 if not policy_available else 0.0),
        }

        reasons: list[str] = []
        if wm_disabled and bool(features.get("wm_available")):
            reasons.append("wm_disabled_due_to_high_uncertainty")
        if low_confidence:
            reasons.append("policy_confidence_low")
        if high_confidence:
            reasons.append("policy_confidence_high")
        if high_risk:
            reasons.append("high_risk_slice")
        if not search_available and bool(features.get("search_available")):
            reasons.append("search_budget_too_low")

        selected = "heuristic_baseline"
        routing_reason = "fallback_to_heuristic"
        if wm_preferred and high_confidence and controller_scores["policy_plus_wm_rerank"] > -900.0:
            selected = "policy_plus_wm_rerank"
            routing_reason = "policy_confident_and_wm_reliable"
        elif low_confidence and controller_scores["search_baseline"] > -900.0:
            selected = "search_baseline"
            routing_reason = "policy_ambiguous_search_budget_available"
        elif high_risk and controller_scores["search_baseline"] > controller_scores["policy_baseline"]:
            selected = "search_baseline"
            routing_reason = "high_risk_slice_prefers_search"
        elif controller_scores["policy_baseline"] > -900.0:
            selected = "policy_baseline"
            routing_reason = "policy_baseline_sufficient"
        elif heuristic_available:
            selected = "heuristic_baseline"
            routing_reason = "policy_unavailable_use_heuristic"

        if selected not in available_controllers:
            for candidate in ("policy_baseline", "heuristic_baseline", "search_baseline"):
                if candidate in available_controllers and controller_scores.get(candidate, -999.0) > -900.0:
                    selected = candidate
                    routing_reason = f"selected_controller_unavailable_fallback_to_{candidate}"
                    break

        rejected: list[dict[str, Any]] = []
        for controller_id, score in controller_scores.items():
            if controller_id == selected:
                continue
            if controller_id not in available_controllers:
                rejected.append({"controller_id": controller_id, "reason": "not_registered", "score": float(score)})
            elif score <= -900.0:
                rejected.append({"controller_id": controller_id, "reason": "gated_out", "score": float(score)})
            else:
                rejected.append({"controller_id": controller_id, "reason": "lower_score", "score": float(score)})

        return {
            "schema": "p48_router_decision_v1",
            "generated_at": _now_iso(),
            "selected_controller": selected,
            "routing_reason": routing_reason,
            "routing_score_breakdown": {key: float(value) for key, value in controller_scores.items()},
            "rejected_controllers": rejected,
            "key_feature_values": {
                "policy_margin": float(margin),
                "policy_entropy": float(entropy),
                "wm_uncertainty": float(wm_uncertainty),
                "wm_predicted_return": float(wm_predicted_return),
                "slice_stage": stage,
                "slice_resource_pressure": resource_pressure,
                "budget_level": str(features.get("budget_level") or "low"),
                "phase": phase,
            },
            "explainability": {
                "high_confidence": bool(high_confidence),
                "low_confidence": bool(low_confidence),
                "high_risk": bool(high_risk),
                "wm_disabled": bool(wm_disabled),
                "wm_preferred": bool(wm_preferred),
                "reasons_triggered": reasons,
            },
        }


def summarize_routing_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        selected_counter[str(row.get("selected_controller") or "unknown")] += 1
        reason_counter[str(row.get("routing_reason") or "unknown")] += 1
    return {
        "schema": "p48_routing_summary_v1",
        "generated_at": _now_iso(),
        "decision_count": int(sum(selected_counter.values())),
        "controller_selection_distribution": [
            {"controller_id": key, "count": int(value), "ratio": float(value) / max(1, sum(selected_counter.values()))}
            for key, value in sorted(selected_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "top_routing_reasons": [
            {"routing_reason": key, "count": int(value), "ratio": float(value) / max(1, sum(reason_counter.values()))}
            for key, value in sorted(reason_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ],
    }


def _summary_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# P48 Router Smoke",
        "",
        f"- decision_count: {int(summary.get('decision_count') or 0)}",
        "",
        "## Controller Selection Distribution",
    ]
    rows = summary.get("controller_selection_distribution") if isinstance(summary.get("controller_selection_distribution"), list) else []
    if rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {controller}: count={count} ratio={ratio:.3f}".format(
                    controller=row.get("controller_id"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Typical Routing Reasons"])
    reasons = summary.get("top_routing_reasons") if isinstance(summary.get("top_routing_reasons"), list) else []
    if reasons:
        for row in reasons:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- {reason}: count={count} ratio={ratio:.3f}".format(
                    reason=row.get("routing_reason"),
                    count=int(row.get("count") or 0),
                    ratio=float(row.get("ratio") or 0.0),
                )
            )
    else:
        lines.append("- none")
    return lines


def run_router_smoke(
    *,
    out_dir: str | Path | None = None,
    seed: str = "AAAAAAA",
    max_states: int = 6,
    model_path: str = "",
    world_model_checkpoint: str = "",
    router_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = (
        (repo_root / "docs/artifacts/p48").resolve()
        if out_dir is None
        else (Path(out_dir) if Path(out_dir).is_absolute() else (repo_root / out_dir).resolve())
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = _now_stamp()
    trace_dir = (output_root / "router_traces" / run_id).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "routing_trace.jsonl"

    feature_smoke = run_routing_feature_smoke(
        out_dir=output_root,
        seed=seed,
        max_states=max_states,
        model_path=model_path or discover_policy_model_path(repo_root),
        world_model_checkpoint=world_model_checkpoint or discover_world_model_checkpoint(repo_root),
    )
    rows = (feature_smoke.get("payload") or {}).get("rows") if isinstance(feature_smoke.get("payload"), dict) else []
    registry = build_controller_registry(
        repo_root=repo_root,
        world_model_checkpoint=world_model_checkpoint or discover_world_model_checkpoint(repo_root),
        model_path=model_path or discover_policy_model_path(repo_root),
    )
    controller_map = {
        str(row.get("controller_id") or ""): row
        for row in (registry.get("controllers") if isinstance(registry.get("controllers"), list) else [])
        if isinstance(row, dict) and str(row.get("controller_id") or "")
    }
    router = RuleBasedHybridRouter(router_config)

    decisions: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        decision = router.route(features=features, available_controllers=controller_map)
        decision["sample_id"] = str(row.get("sample_id") or "")
        decision["step_idx"] = int(row.get("step_idx") or 0)
        decision["phase"] = str(row.get("phase") or features.get("phase") or "UNKNOWN")
        decisions.append(decision)
        _append_jsonl(trace_path, decision)

    summary = summarize_routing_trace(decisions)
    summary["trace_path"] = str(trace_path)
    summary["feature_smoke_out"] = str(feature_smoke.get("out") or "")
    summary_path = trace_dir / "routing_summary.json"
    write_json(summary_path, summary)
    md_path = output_root / f"router_smoke_{run_id}.md"
    write_markdown(md_path, _summary_markdown(summary))
    return {
        "status": "ok",
        "trace_path": str(trace_path),
        "routing_summary_json": str(summary_path),
        "routing_summary_md": str(md_path),
        "payload": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P48 rule-based router smoke.")
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--max-states", type=int, default=6)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--world-model-checkpoint", default="")
    parser.add_argument("--out-dir", default="docs/artifacts/p48")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_router_smoke(
        out_dir=args.out_dir,
        seed=str(args.seed or "AAAAAAA"),
        max_states=max(1, int(args.max_states)),
        model_path=str(args.model_path or ""),
        world_model_checkpoint=str(args.world_model_checkpoint or ""),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
