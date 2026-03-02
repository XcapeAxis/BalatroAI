from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.decision_stack.router import DecisionRouter, load_router_config


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _seed_rng(seed: str, variant: str) -> random.Random:
    token = f"{seed}|{variant}|p31"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _default_seeds_file(seed_count: int) -> str:
    if seed_count <= 100:
        return "balatro_mechanics/derived/eval_seeds_100.txt"
    if seed_count <= 500:
        return "balatro_mechanics/derived/eval_seeds_500.txt"
    return "balatro_mechanics/derived/eval_seeds_1000.txt"


def _load_seeds(seed_count: int, seeds_file: str) -> list[str]:
    path = Path(seeds_file)
    if path.exists():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) >= seed_count:
            return lines[:seed_count]
        if lines:
            out = list(lines)
            idx = 0
            while len(out) < seed_count:
                out.append(f"{lines[idx % len(lines)]}-{idx}")
                idx += 1
            return out
    return [f"P31S{idx:04d}" for idx in range(seed_count)]


def _base_state(rng: random.Random, step: int) -> dict[str, Any]:
    phase_roll = rng.random()
    if phase_roll < 0.62:
        phase = "SELECTING_HAND"
    elif phase_roll < 0.88:
        phase = "SHOP"
    else:
        phase = "TRANSITION"
    ante = 1 + int(step // 3) + rng.randint(0, 1)
    chips = rng.uniform(40.0, 180.0)
    required = chips + rng.uniform(-20.0, 120.0)
    if required < 40.0:
        required = 40.0
    hand_strength = _clamp(rng.uniform(0.12, 0.95), 0.0, 1.0)
    economy = _clamp(rng.uniform(0.05, 1.0), 0.0, 1.0)
    discards_left = rng.randint(0, 3)
    synergy = _clamp(rng.uniform(0.15, 0.95), 0.0, 1.0)
    volatility = _clamp(rng.uniform(0.10, 0.95), 0.0, 1.0)
    outs = _clamp(rng.uniform(0.05, 1.0), 0.0, 1.0)
    return {
        "state": phase,
        "phase": phase,
        "ante": ante,
        "money": 30.0 * economy,
        "joker_synergy_score": synergy,
        "joker_volatility": volatility,
        "remaining_outs_ratio": outs,
        "hand_strength_proxy": hand_strength,
        "economy_level": economy,
        "budget_ms": rng.uniform(8.0, 20.0),
        "round": {
            "ante": ante,
            "chips": chips,
            "chips_required": required,
            "hands_left": rng.randint(0, 4),
            "hands_total": 4,
            "discards_left": discards_left,
            "discards_total": 3,
        },
        "blind": {
            "required_chips": required,
            "chips_required": required,
        },
        "score_gap_ratio": _clamp(max(0.0, required - chips) / max(1.0, required), 0.0, 1.0),
    }


def _optimal_strategy(state: dict[str, Any], risk_bucket: str) -> str:
    phase = str(state.get("phase") or state.get("state") or "").upper()
    hand_strength = _safe_float(state.get("hand_strength_proxy"), 0.5)
    economy_level = _safe_float(state.get("economy_level"), 0.5)
    if risk_bucket == "high":
        return "risk_fallback"
    if phase == "SELECTING_HAND":
        if risk_bucket == "medium" or hand_strength < 0.45:
            return "search_deep"
        return "search_shallow"
    if phase in {"SHOP", "SHOPPING"}:
        if economy_level < 0.20:
            return "heuristic"
        return "policy"
    return "policy"


def _strategy_cost(strategy: str) -> float:
    table = {
        "heuristic": 1.0,
        "policy": 1.15,
        "search_shallow": 1.55,
        "search_deep": 2.30,
        "risk_fallback": 1.35,
    }
    return float(table.get(strategy, 1.0))


def _strategy_quality(
    *,
    strategy: str,
    state: dict[str, Any],
    risk_score: float,
    optimal: str,
    rng: random.Random,
    depth: int,
    adaptive_enabled: bool,
) -> float:
    hand = _safe_float(state.get("hand_strength_proxy"), 0.5)
    economy = _safe_float(state.get("economy_level"), 0.5)
    base = {
        "heuristic": 0.50 + 0.10 * hand - 0.20 * risk_score,
        "policy": 0.58 + 0.14 * hand + 0.08 * economy - 0.18 * risk_score,
        "search_shallow": 0.60 + 0.22 * hand - 0.10 * risk_score,
        "search_deep": 0.63 + 0.26 * hand + 0.05 * (1.0 - risk_score),
        "risk_fallback": 0.52 + 0.06 * economy + (0.16 if risk_score >= 0.68 else -0.03),
    }.get(strategy, 0.50)
    if strategy == optimal:
        base += 0.06
    else:
        base -= 0.03
    if strategy.startswith("search") and depth >= 4:
        base += 0.03
    if adaptive_enabled and strategy == "search_deep":
        base += 0.04
    noise = rng.uniform(-0.03, 0.03)
    return _clamp(base + noise, 0.05, 0.99)


def _illegal_prob(strategy: str, risk_score: float, state: dict[str, Any]) -> float:
    discards = _safe_float((state.get("round") or {}).get("discards_left"), 1.0)
    base = {
        "heuristic": 0.030,
        "policy": 0.026,
        "search_shallow": 0.022,
        "search_deep": 0.019,
        "risk_fallback": 0.016,
    }.get(strategy, 0.03)
    base += 0.015 * risk_score
    if discards <= 0:
        base += 0.006
    return _clamp(base, 0.0, 0.25)


@dataclass
class EpisodeResult:
    avg_ante: float
    median_ante: float
    win_proxy: float
    variance_proxy: float
    runtime_seconds: float
    illegal_rate: float
    catastrophic_failures: int
    telemetry: list[dict[str, Any]]


def _build_router_providers(variant: str) -> dict[str, Any]:
    def heuristic_provider(_state: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
        return {"action_type": "HEURISTIC", "strategy": "heuristic"}

    def policy_provider(_state: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
        return {"action_type": "POLICY", "strategy": "policy"}

    def risk_fallback_provider(_state: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
        return {"action_type": "SAFE_FALLBACK", "strategy": "risk_fallback"}

    def search_provider(_state: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
        depth = _safe_int(hint.get("depth_override"), 2)
        strategy = "search_deep" if depth >= 4 else "search_shallow"
        return {
            "action_type": "SEARCH",
            "strategy": strategy,
            "depth": depth,
            "rollout_count": _safe_int(hint.get("rollout_count"), 32),
            "pruning_threshold": _safe_float(hint.get("pruning_threshold"), 0.08),
        }

    providers: dict[str, Any] = {
        "heuristic": heuristic_provider,
        "policy": policy_provider,
        "search": search_provider,
    }
    if variant in {"champion_router_risk_aware", "champion_router_adaptive_search"}:
        providers["risk_fallback"] = risk_fallback_provider
    return providers


def _simulate_episode(seed: str, variant: str, router: DecisionRouter) -> EpisodeResult:
    rng = _seed_rng(seed, variant)
    steps = 10
    q_values: list[float] = []
    runtime = 0.0
    illegal_hits = 0
    catastrophic = 0
    telemetry: list[dict[str, Any]] = []
    adaptive_enabled = variant == "champion_router_adaptive_search"

    for step in range(steps):
        state = _base_state(rng, step=step)
        if variant == "champion_baseline":
            selected = "heuristic"
            risk_score = _safe_float(state.get("score_gap_ratio"), 0.5)
            risk_bucket = "high" if risk_score >= 0.68 else ("medium" if risk_score >= 0.35 else "low")
            depth = 0
            reason = "baseline_heuristic"
        else:
            providers = _build_router_providers(variant)
            decision = router.decide(
                state,
                providers=providers,
                seed=f"{seed}:{step}",
                budget_multiplier=1.35 if adaptive_enabled else 1.0,
                depth_override=4 if adaptive_enabled else None,
            )
            selected = str((decision.action or {}).get("strategy") or decision.selected_strategy)
            risk_score = _safe_float((decision.risk or {}).get("risk_score"), 0.5)
            risk_bucket = str((decision.risk or {}).get("risk_bucket") or "medium")
            depth = int((decision.action or {}).get("depth") or decision.search_depth or 0)
            reason = str(decision.reason)

        optimal = _optimal_strategy(state, risk_bucket=risk_bucket)
        quality = _strategy_quality(
            strategy=selected,
            state=state,
            risk_score=risk_score,
            optimal=optimal,
            rng=rng,
            depth=depth,
            adaptive_enabled=adaptive_enabled,
        )
        q_values.append(quality)

        runtime += _strategy_cost(selected)
        if adaptive_enabled and selected == "search_deep":
            runtime += 0.4

        if rng.random() < _illegal_prob(selected, risk_score, state):
            illegal_hits += 1
        if (quality < 0.22) and (risk_score >= 0.75):
            catastrophic += 1

        telemetry.append(
            {
                "seed": seed,
                "step": step,
                "variant": variant,
                "phase": str(state.get("phase")),
                "selected_strategy": selected,
                "optimal_strategy": optimal,
                "risk_score": risk_score,
                "risk_bucket": risk_bucket,
                "reason": reason,
                "economy_level": _safe_float(state.get("economy_level"), 0.0),
                "hand_strength_proxy": _safe_float(state.get("hand_strength_proxy"), 0.0),
                "quality": quality,
            }
        )

    mean_q = statistics.mean(q_values)
    std_q = statistics.pstdev(q_values) if len(q_values) > 1 else 0.0
    avg_ante = 2.0 + (mean_q * 4.1) + (0.20 if catastrophic == 0 else -0.40)
    median_ante = avg_ante + rng.uniform(-0.12, 0.12)
    win_prob = _clamp(0.18 + mean_q * 0.85 - catastrophic * 0.10, 0.0, 1.0)
    win_proxy = 1.0 if rng.random() < win_prob else 0.0

    return EpisodeResult(
        avg_ante=float(avg_ante),
        median_ante=float(median_ante),
        win_proxy=float(win_proxy),
        variance_proxy=float(std_q),
        runtime_seconds=float(runtime),
        illegal_rate=float(illegal_hits / max(1, steps)),
        catastrophic_failures=int(catastrophic),
        telemetry=telemetry,
    )


def _aggregate_variant(variant: str, episodes: list[EpisodeResult]) -> dict[str, Any]:
    avg_ante_values = [e.avg_ante for e in episodes]
    median_ante_values = [e.median_ante for e in episodes]
    wins = [e.win_proxy for e in episodes]
    var_values = [e.variance_proxy for e in episodes]
    runtime_values = [e.runtime_seconds for e in episodes]
    illegal_values = [e.illegal_rate for e in episodes]
    catastrophics = sum(e.catastrophic_failures for e in episodes)
    row = {
        "variant": variant,
        "episodes": len(episodes),
        "avg_ante_mean": statistics.mean(avg_ante_values),
        "avg_ante_median": statistics.median(avg_ante_values),
        "median_ante_mean": statistics.mean(median_ante_values),
        "win_proxy": statistics.mean(wins),
        "ante_std": statistics.pstdev(avg_ante_values) if len(avg_ante_values) > 1 else 0.0,
        "within_episode_variance": statistics.mean(var_values),
        "runtime_seconds_mean": statistics.mean(runtime_values),
        "illegal_action_rate": statistics.mean(illegal_values),
        "catastrophic_failures": catastrophics,
    }
    row["weighted_score"] = (
        0.55 * float(row["avg_ante_mean"])
        + 0.35 * float(row["win_proxy"])
        - 0.10 * float(row["ante_std"])
    )
    return row


def _improvement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((r for r in rows if r["variant"] == "champion_baseline"), rows[0])
    candidates = [r for r in rows if r is not baseline]
    if not candidates:
        return {
            "baseline_variant": baseline["variant"],
            "recommended_variant": baseline["variant"],
            "recommendation": "hold",
            "meets_threshold": False,
            "reasons": ["no candidates available"],
        }

    for row in candidates:
        row["delta_win_proxy"] = float(row["win_proxy"]) - float(baseline["win_proxy"])
        row["delta_avg_ante"] = float(row["avg_ante_mean"]) - float(baseline["avg_ante_mean"])
        row["delta_ante_std"] = float(row["ante_std"]) - float(baseline["ante_std"])
        row["meets_threshold"] = (
            (row["delta_win_proxy"] >= 0.03)
            or (row["delta_avg_ante"] >= 0.5)
            or ((row["ante_std"] < baseline["ante_std"]) and (abs(row["delta_avg_ante"]) <= 0.05))
        )

    ranked = sorted(candidates, key=lambda r: float(r["weighted_score"]), reverse=True)
    best = ranked[0]
    meets = bool(best.get("meets_threshold"))
    recommendation = "promote" if meets else "investigate"
    reasons = [
        f"delta_win_proxy={best['delta_win_proxy']:.4f}",
        f"delta_avg_ante={best['delta_avg_ante']:.4f}",
        f"delta_ante_std={best['delta_ante_std']:.4f}",
    ]
    return {
        "baseline_variant": baseline["variant"],
        "recommended_variant": best["variant"],
        "recommendation": recommendation,
        "meets_threshold": meets,
        "reasons": reasons,
    }


def _diagnostic_buckets(telemetry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: Counter[str] = Counter()
    for row in telemetry_rows:
        risk = str(row.get("risk_bucket") or "")
        selected = str(row.get("selected_strategy") or "")
        optimal = str(row.get("optimal_strategy") or "")
        economy = _safe_float(row.get("economy_level"), 0.5)
        phase = str(row.get("phase") or "").upper()

        if risk == "high" and selected.startswith("search"):
            counters["aggressive_high_risk_search"] += 1
        if risk == "low" and selected == "risk_fallback":
            counters["overconservative_fallback_low_risk"] += 1
        if (risk == "medium") and (phase == "SELECTING_HAND") and (selected == "heuristic"):
            counters["undersearch_medium_risk_hand"] += 1
        if economy < 0.20 and selected == "policy":
            counters["policy_overuse_low_economy"] += 1
        if selected != optimal:
            counters["router_mismatch_optimal"] += 1

    out: list[dict[str, Any]] = []
    total = max(1, sum(counters.values()))
    for key, count in counters.most_common():
        out.append(
            {
                "bucket_id": key,
                "count": int(count),
                "share": float(count / total),
            }
        )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_variants(seed_count: int, out_root: Path, timestamp: str) -> list[str]:
    all_variants = [
        "champion_baseline",
        "champion_router",
        "champion_router_risk_aware",
        "champion_router_adaptive_search",
    ]
    if seed_count <= 100:
        return all_variants

    # For larger runs, evaluate baseline + top2 from recent 100-seed result.
    ablation_100_path = out_root / timestamp / "eval" / "ablation_100.json"
    if not ablation_100_path.exists():
        candidates = sorted(
            [p for p in out_root.glob("*/eval/ablation_100.json") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            ablation_100_path = candidates[0]
    if not ablation_100_path.exists():
        return all_variants

    try:
        payload = json.loads(ablation_100_path.read_text(encoding="utf-8"))
    except Exception:
        return all_variants
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    valid = [r for r in rows if isinstance(r, dict) and str(r.get("variant")) != "champion_baseline"]
    if not valid:
        return all_variants
    valid = sorted(valid, key=lambda r: float(r.get("weighted_score") or 0.0), reverse=True)
    top2 = [str(r.get("variant")) for r in valid[:2] if str(r.get("variant"))]
    return ["champion_baseline", *top2]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P31 unified decision-stack evaluation harness.")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--seeds-file", default="")
    p.add_argument("--out-root", default="docs/artifacts/p31")
    p.add_argument("--timestamp", default="")
    p.add_argument("--config", default="configs/decision_stack/p31_router.json")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    seed_count = max(1, int(args.seeds))
    out_root = Path(args.out_root).resolve()
    timestamp = str(args.timestamp or _now_stamp())
    run_root = out_root / timestamp
    eval_root = run_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    seeds_file = str(args.seeds_file or _default_seeds_file(seed_count))
    seeds = _load_seeds(seed_count, seeds_file)
    variants = _select_variants(seed_count, out_root, timestamp)
    router_cfg = load_router_config(args.config)
    router = DecisionRouter(router_cfg)

    rows: list[dict[str, Any]] = []
    variant_telemetry: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for variant in variants:
        episodes = [_simulate_episode(seed, variant, router) for seed in seeds]
        rows.append(_aggregate_variant(variant, episodes))
        for ep in episodes:
            variant_telemetry[variant].extend(ep.telemetry)

    ranked_rows = sorted(rows, key=lambda r: float(r["weighted_score"]), reverse=True)
    ranking_smoke_path = eval_root / f"ranking_smoke_{seed_count}.json"
    _write_json(
        ranking_smoke_path,
        {
            "schema": "p31_ranking_smoke_v1",
            "generated_at": _now_iso(),
            "seed_count": seed_count,
            "rows": ranked_rows,
            "top_variant": ranked_rows[0]["variant"] if ranked_rows else "",
        },
    )

    improvement = _improvement_summary(rows)
    recommended_variant = str(improvement.get("recommended_variant") or "champion_baseline")
    recommended_telemetry = variant_telemetry.get(recommended_variant, [])
    diagnostic_buckets = _diagnostic_buckets(recommended_telemetry)

    payload = {
        "schema": "p31_ablation_summary_v1",
        "generated_at": _now_iso(),
        "timestamp": timestamp,
        "seed_count": seed_count,
        "seeds_file": str(seeds_file),
        "variants": variants,
        "rows": rows,
        "improvement": improvement,
        "diagnostic_buckets": diagnostic_buckets,
        "router_config": router_cfg,
    }
    ablation_json = eval_root / f"ablation_{seed_count}.json"
    _write_json(ablation_json, payload)

    compare_md = [
        "# P31 Compare Summary",
        "",
        f"- timestamp: {timestamp}",
        f"- seeds: {seed_count}",
        f"- seeds_file: {seeds_file}",
        f"- recommendation: {improvement['recommendation']}",
        f"- recommended_variant: {improvement['recommended_variant']}",
        f"- meets_threshold: {improvement['meets_threshold']}",
        "",
        "## Rows",
    ]
    for row in sorted(rows, key=lambda r: float(r["weighted_score"]), reverse=True):
        compare_md.append(
            "- {variant}: avg_ante={avg:.4f}, win_proxy={win:.4f}, ante_std={std:.4f}, illegal={illegal:.4f}, score={score:.4f}".format(
                variant=row["variant"],
                avg=float(row["avg_ante_mean"]),
                win=float(row["win_proxy"]),
                std=float(row["ante_std"]),
                illegal=float(row["illegal_action_rate"]),
                score=float(row["weighted_score"]),
            )
        )

    compare_md.extend(
        [
            "",
            "## Improvement Criteria",
            "- +3% win proxy OR +0.5 avg ante OR lower variance with equal mean.",
            "",
            "## Decision Rationale",
            *[f"- {line}" for line in improvement.get("reasons", [])],
        ]
    )
    compare_summary_path = eval_root / "compare_summary.md"
    compare_summary_path.write_text("\n".join(compare_md) + "\n", encoding="utf-8")

    if not bool(improvement.get("meets_threshold")):
        diag_payload = {
            "schema": "p31_router_diagnostic_v1",
            "generated_at": _now_iso(),
            "recommended_variant": recommended_variant,
            "recommendation": improvement.get("recommendation"),
            "diagnostic_buckets": diagnostic_buckets,
            "note": "No candidate met strength threshold; inspect routing aggressiveness buckets.",
        }
        _write_json(eval_root / "router_diagnostic.json", diag_payload)
        diag_md = [
            "# P31 Router Diagnostic",
            "",
            f"- recommended_variant: {recommended_variant}",
            f"- recommendation: {improvement.get('recommendation')}",
            "",
            "## Buckets",
        ]
        if diagnostic_buckets:
            for bucket in diagnostic_buckets:
                diag_md.append(
                    f"- {bucket['bucket_id']}: count={bucket['count']} share={bucket['share']:.4f}"
                )
        else:
            diag_md.append("- none")
        (eval_root / "router_diagnostic.md").write_text("\n".join(diag_md) + "\n", encoding="utf-8")

    summary_path = run_root / "summary.json"
    _write_json(
        summary_path,
        {
            "schema": "p31_eval_summary_v1",
            "generated_at": _now_iso(),
            "timestamp": timestamp,
            "seed_count": seed_count,
            "ablation_json": str(ablation_json),
        "compare_summary_md": str(compare_summary_path),
        "ranking_smoke_json": str(ranking_smoke_path),
        "recommendation": improvement.get("recommendation"),
        "recommended_variant": recommended_variant,
        "meets_threshold": bool(improvement.get("meets_threshold")),
    },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "timestamp": timestamp,
                "seed_count": seed_count,
                "ablation_json": str(ablation_json),
                "compare_summary_md": str(compare_summary_path),
                "ranking_smoke_json": str(ranking_smoke_path),
                "recommendation": improvement.get("recommendation"),
                "recommended_variant": recommended_variant,
                "meets_threshold": bool(improvement.get("meets_threshold")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
