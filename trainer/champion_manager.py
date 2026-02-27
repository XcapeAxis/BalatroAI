from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Champion-challenger decision manager (P17/P18 compatible).")
    p.add_argument("--registry-root", default="docs/artifacts/p17/registry")
    p.add_argument("--baseline", required=True, help="Champion eval json (100 seeds).")
    p.add_argument("--candidate", required=True, help="Challenger eval json (100 or 500 seeds).")
    p.add_argument("--compare-summary", default="", help="Optional eval_compare summary json path.")
    p.add_argument("--rules", default="", help="Optional rules yaml/json for risk guard.")
    p.add_argument("--candidate-model", required=True)
    p.add_argument("--candidate-model-id", default="")
    p.add_argument("--dataset-id", default="")
    p.add_argument("--milestone-eval", default="", help="Optional 500/1000 seed eval json for candidate.")
    p.add_argument("--canary-summary", default="", help="Optional canary summary json for risk guard.")
    p.add_argument("--decision-out", required=True)
    p.add_argument("--median-threshold", type=float, default=0.5)
    p.add_argument("--avg-threshold", type=float, default=0.3)
    p.add_argument("--allow-promote", action="store_true", help="Actually update current_champion.json.")
    p.add_argument("--auto-rollback", action="store_true", help="Allow rollback decision when challenger fails safeguards.")
    p.add_argument("--git-commit", default="")
    # v4: release channel support
    p.add_argument("--release-channel", default="", choices=["", "dev", "canary", "stable"],
                   help="Target release channel for v4 staged promotion.")
    p.add_argument("--determinism-pass", default="", help="Determinism audit pass (true/false/skip).")
    p.add_argument("--package-verify-pass", default="", help="Package verify pass (true/false).")
    return p


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "win_rate": float(payload.get("win_rate") or 0.0),
        "avg_ante_reached": float(payload.get("avg_ante_reached") or 0.0),
        "median_ante": float(payload.get("median_ante") or 0.0),
    }


def _load_rules(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}


def _failure_rate(payload: dict[str, Any], key: str) -> float:
    br = payload.get("failure_breakdown") if isinstance(payload.get("failure_breakdown"), dict) else {}
    episodes = max(1.0, float(payload.get("episodes") or 0.0))
    try:
        cnt = float(br.get(key) or 0.0)
    except Exception:
        cnt = 0.0
    return (cnt / episodes) * 100.0


def _canary_status_pass(status: str) -> bool:
    token = str(status or "").strip().upper()
    return token in {"PASS", "SKIP", "SKIPPED"}


def main() -> int:
    args = _build_parser().parse_args()
    reg_root = Path(args.registry_root)
    reg_root.mkdir(parents=True, exist_ok=True)

    baseline = _read_json(Path(args.baseline))
    candidate = _read_json(Path(args.candidate))
    compare = _read_json(Path(args.compare_summary)) if args.compare_summary else {}
    if not baseline or not candidate:
        print("missing baseline/candidate eval json")
        return 2

    b = _extract_metrics(baseline)
    c = _extract_metrics(candidate)
    delta_avg = c["avg_ante_reached"] - b["avg_ante_reached"]
    delta_med = c["median_ante"] - b["median_ante"]
    delta_win = c["win_rate"] - b["win_rate"]

    ci_lb = None
    # compare summary from eval_compare has rows under candidates.
    if isinstance(compare.get("candidates"), list):
        for row in compare.get("candidates"):
            if isinstance(row, dict):
                ci = (row.get("delta") or {}).get("avg_ante_ci95")
                if isinstance(ci, dict):
                    ci_lb = ci.get("low")
                break

    rules = _load_rules(args.rules)
    median_thr = float(rules.get("median_ante_delta_threshold", args.median_threshold))
    avg_thr = float(rules.get("avg_ante_delta_threshold", args.avg_threshold))
    perf_pass = bool(
        (delta_med >= median_thr)
        or (delta_avg >= avg_thr)
        or (bool(rules.get("use_ci_lower_bound_guard", True)) and ci_lb is not None and float(ci_lb) > 0.0)
    )

    risk_thr = float(rules.get("critical_bucket_regression_max_pct_points", 1000.0))
    critical = [str(x) for x in (rules.get("critical_buckets") or [])]
    regressions: list[dict[str, Any]] = []
    for k in critical:
        b_pct = _failure_rate(baseline, k)
        c_pct = _failure_rate(candidate, k)
        delta_pct = c_pct - b_pct
        if delta_pct > risk_thr:
            regressions.append(
                {
                    "bucket": k,
                    "baseline_pct": b_pct,
                    "candidate_pct": c_pct,
                    "delta_pct_points": delta_pct,
                    "threshold_pct_points": risk_thr,
                }
            )
    risk_guard_pass = len(regressions) == 0

    milestone_payload = _read_json(Path(args.milestone_eval)) if args.milestone_eval else {}
    milestone_present = bool(milestone_payload)
    milestone_metrics = _extract_metrics(milestone_payload) if milestone_present else {}
    milestone_pass = False
    if milestone_present:
        milestone_pass = bool(
            (float(milestone_metrics.get("median_ante") or 0.0) >= float(b.get("median_ante") or 0.0))
            or (float(milestone_metrics.get("avg_ante_reached") or 0.0) >= float(b.get("avg_ante_reached") or 0.0))
        )

    canary_payload = _read_json(Path(args.canary_summary)) if args.canary_summary else {}
    canary_status = str(canary_payload.get("status") or "UNKNOWN").upper() if canary_payload else "MISSING"
    canary_pass = _canary_status_pass(canary_status)

    rollback_on_perf_streak = int(rules.get("rollback_on_perf_fail_streak", 2))
    rollback_canary_div_threshold = float(rules.get("rollback_on_canary_divergence_threshold", 0.45))
    probation_min_perf_passes = int(rules.get("probation_min_perf_passes", 1))

    canary_divergence_rate: float | None = None
    if canary_payload:
        canary_divergence_rate = canary_payload.get("top1_divergence_rate")
        if canary_divergence_rate is None and isinstance(canary_payload.get("divergence_summary"), str):
            div_path = Path(canary_payload["divergence_summary"])
            if div_path.exists():
                div_payload = _read_json(div_path)
                canary_divergence_rate = div_payload.get("top1_divergence_rate")
    canary_rollback_triggered = (
        canary_divergence_rate is not None
        and float(canary_divergence_rate) > rollback_canary_div_threshold
    )

    current_path = reg_root / "current_champion.json"
    current_payload = _read_json(current_path)
    current_model_id = str(current_payload.get("model_id") or "")
    current_model_path = str(current_payload.get("model_path") or "")
    previous_model_id = str(current_payload.get("previous_model_id") or "")
    previous_model_path = str(current_payload.get("previous_model_path") or "")

    candidate_model_path_norm = str(Path(args.candidate_model).resolve())
    current_model_path_norm = str(Path(current_model_path).resolve()) if current_model_path else ""
    is_champion_reeval = bool(current_model_id and candidate_model_path_norm == current_model_path_norm)

    streak_path = reg_root / "perf_fail_streak.json"
    streak_payload = _read_json(streak_path)
    streak_model_id = str(streak_payload.get("model_id") or "")
    streak_count = int(streak_payload.get("count") or 0)
    if streak_model_id != current_model_id:
        streak_count = 0

    if is_champion_reeval and not perf_pass:
        streak_count += 1
    else:
        streak_count = 0
    _write_json(
        streak_path,
        {"schema": "p19_perf_fail_streak_v1", "model_id": current_model_id or streak_model_id, "count": streak_count, "updated_at": _now_iso()},
    )

    perf_streak_rollback = (
        args.auto_rollback
        and current_model_id
        and is_champion_reeval
        and not perf_pass
        and streak_count >= rollback_on_perf_streak
    )

    if perf_pass and risk_guard_pass:
        if milestone_present and milestone_pass and canary_pass:
            final_decision = "promote" if args.allow_promote else "hold_for_promote"
            reason = "perf/risk passed and milestone+canary passed"
            new_status = "champion" if final_decision == "promote" else "probation"
        else:
            final_decision = "hold_for_more_data"
            reason = "perf/risk passed but awaiting milestone/canary confirmation"
            new_status = "probation"
    elif perf_pass and (not risk_guard_pass):
        final_decision = "hold_for_more_data"
        reason = "risk guard blocked promotion"
        new_status = "candidate"
    else:
        if perf_streak_rollback:
            final_decision = "rolled_back"
            reason = f"perf fail streak ({streak_count}) >= {rollback_on_perf_streak}; auto_rollback"
            new_status = "rolled_back"
        elif canary_rollback_triggered and args.auto_rollback and current_model_id:
            final_decision = "rolled_back"
            reason = f"canary divergence rate {canary_divergence_rate} > {rollback_canary_div_threshold}; auto_rollback"
            new_status = "rolled_back"
        elif args.auto_rollback and current_model_id:
            final_decision = "rolled_back"
            reason = "perf gate failed and auto_rollback enabled"
            new_status = "rolled_back"
        else:
            final_decision = "reject"
            reason = "perf gate did not pass thresholds"
            new_status = "rejected"

    model_id = args.candidate_model_id or f"p17_pv_challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    decision_payload = {
        "schema": "champion_decision_v3",
        "generated_at": _now_iso(),
        "decision": final_decision,
        "final_decision": final_decision,
        "status": new_status,
        "reason": reason,
        "perf_gate_pass": perf_pass,
        "risk_guard_pass": risk_guard_pass,
        "milestone_eval_present": milestone_present,
        "milestone_eval_pass": milestone_pass,
        "canary_status": canary_status,
        "canary_pass": canary_pass,
        "decision_reasons": [reason],
        "risk_regressions": regressions,
        "thresholds": {
            "median_ante_delta": median_thr,
            "avg_ante_delta": avg_thr,
            "ci_lower_gt_zero": True,
            "critical_bucket_regression_max_pct_points": risk_thr,
        },
        "metrics": {
            "baseline_100": b,
            "candidate_100": c,
            "candidate_milestone": milestone_metrics,
            "delta": {"avg_ante": delta_avg, "median_ante": delta_med, "win_rate": delta_win, "avg_ante_ci95_low": ci_lb},
        },
        "artifacts": {
            "baseline_eval": str(Path(args.baseline)),
            "candidate_eval": str(Path(args.candidate)),
            "compare_summary": str(Path(args.compare_summary)) if args.compare_summary else None,
            "candidate_model": str(Path(args.candidate_model)),
            "milestone_eval": str(Path(args.milestone_eval)) if args.milestone_eval else None,
            "canary_summary": str(Path(args.canary_summary)) if args.canary_summary else None,
        },
        "model_id": model_id,
        "dataset_id": args.dataset_id or None,
        "git_commit": args.git_commit or None,
        "parent_model_id": current_model_id or None,
        "rollback_parent_model_id": previous_model_id or None,
        "rollback_reason": reason if final_decision == "rolled_back" else None,
        "perf_fail_streak_count": streak_count,
        "perf_fail_streak_threshold": rollback_on_perf_streak,
        "canary_divergence_rate": canary_divergence_rate,
        "canary_rollback_threshold": rollback_canary_div_threshold,
        "canary_rollback_triggered": canary_rollback_triggered,
        "perf_streak_rollback": perf_streak_rollback,
    }
    _write_json(Path(args.decision_out), decision_payload)
    if final_decision == "rolled_back":
        decision_dir = Path(args.decision_out).parent
        rollback_report = {
            "schema": "p19_rollback_report_v1",
            "generated_at": _now_iso(),
            "reason": reason,
            "perf_gate_pass": perf_pass,
            "perf_fail_streak_count": streak_count,
            "perf_fail_streak_threshold": rollback_on_perf_streak,
            "canary_rollback_triggered": canary_rollback_triggered,
            "canary_divergence_rate": canary_divergence_rate,
            "previous_champion_restored": {"model_id": previous_model_id, "model_path": previous_model_path},
            "rolled_back_model": {"model_id": current_model_id, "model_path": current_model_path},
            "decision_artifact": str(Path(args.decision_out)),
        }
        _write_json(decision_dir / "rollback_report.json", rollback_report)
        md_lines = [
            "# P19 Rollback Report",
            "",
            f"- reason: {reason}",
            f"- perf_fail_streak: {streak_count} >= {rollback_on_perf_streak}",
            f"- canary_rollback_triggered: {canary_rollback_triggered}",
            f"- previous champion restored: {previous_model_id}",
        ]
        (decision_dir / "rollback_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    md_path = Path(args.decision_out).with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                "# Champion Decision",
                "",
                f"- decision: {decision_payload['decision']}",
                f"- status: {decision_payload['status']}",
                f"- reason: {decision_payload['reason']}",
                f"- perf_gate_pass: {decision_payload['perf_gate_pass']}",
                f"- risk_guard_pass: {decision_payload['risk_guard_pass']}",
                f"- milestone_eval_pass: {decision_payload['milestone_eval_pass']}",
                f"- canary_status: {decision_payload['canary_status']}",
                f"- delta_avg_ante: {delta_avg:.4f}",
                f"- delta_median_ante: {delta_med:.4f}",
                f"- delta_win_rate: {delta_win:.4f}",
                f"- ci95_low(avg_ante_delta): {ci_lb}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if final_decision == "promote":
        _write_json(
            current_path,
            {
                "schema": "champion_pointer_v2",
                "updated_at": _now_iso(),
                "model_id": model_id,
                "model_path": str(Path(args.candidate_model)),
                "previous_model_id": current_model_id or None,
                "previous_model_path": str(current_payload.get("model_path") or ""),
                "source_decision": str(Path(args.decision_out)),
            },
        )
    elif final_decision == "rolled_back":
        rollback_model_id = previous_model_id or current_model_id
        rollback_model_path = previous_model_path or str(current_payload.get("model_path") or "")
        if rollback_model_id and rollback_model_path:
            _write_json(
                current_path,
                {
                    "schema": "champion_pointer_v2",
                    "updated_at": _now_iso(),
                    "model_id": rollback_model_id,
                    "model_path": rollback_model_path,
                    "previous_model_id": current_model_id or None,
                    "previous_model_path": str(current_payload.get("model_path") or ""),
                    "source_decision": str(Path(args.decision_out)),
                },
            )

    _append_jsonl(
        reg_root / "models_registry.jsonl",
        {
            "schema": "model_registry_v2",
            "created_at": _now_iso(),
            "model_id": model_id,
            "parent_model_id": current_model_id or None,
            "dataset_id": args.dataset_id or None,
            "model_path": str(Path(args.candidate_model)),
            "decision": final_decision,
            "status": new_status,
            "probation_started_at": _now_iso() if new_status == "probation" else None,
            "probation_metrics": {"perf_gate_pass": perf_pass, "risk_guard_pass": risk_guard_pass},
            "rollback_parent_model_id": previous_model_id or None,
            "rollback_reason": decision_payload.get("rollback_reason"),
            "metrics_eval_100": c,
            "metrics_eval_milestone": milestone_metrics,
            "git_commit": args.git_commit or None,
            "decision_artifact": str(Path(args.decision_out)),
        },
    )

    # v4: Release channel state management
    release_channel = getattr(args, "release_channel", "") or ""
    if release_channel:
        channel_state = _compute_release_state(
            args, final_decision, perf_pass, risk_guard_pass,
            milestone_present, milestone_pass, canary_pass,
            release_channel, reg_root, model_id, decision_payload,
        )
        _write_json(reg_root / "release_state.json", channel_state)
        decision_dir = Path(args.decision_out).parent
        _write_json(decision_dir / "release_state.json", channel_state)
        release_md = [
            "# Release Channel State",
            "",
            f"- channel: {channel_state.get('channel')}",
            f"- stable: {channel_state.get('stable_model_id')}",
            f"- canary: {channel_state.get('canary_model_id')}",
            f"- action: {channel_state.get('action')}",
            f"- reason: {channel_state.get('reason')}",
        ]
        (decision_dir / "release_decision.md").write_text("\n".join(release_md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": final_decision,
                "perf_gate_pass": perf_pass,
                "risk_guard_pass": risk_guard_pass,
                "decision_out": str(Path(args.decision_out)),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _compute_release_state(
    args, final_decision: str, perf_pass: bool, risk_guard_pass: bool,
    milestone_present: bool, milestone_pass: bool, canary_pass: bool,
    target_channel: str, reg_root: Path, model_id: str,
    decision_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compute release channel state for v4 staged promotion."""
    current = _read_json(reg_root / "release_state.json")
    stable_id = str(current.get("stable_model_id") or "")
    stable_path = str(current.get("stable_model_path") or "")
    canary_id = str(current.get("canary_model_id") or "")
    canary_path = str(current.get("canary_model_path") or "")

    det_pass_str = str(getattr(args, "determinism_pass", "") or "").lower()
    det_pass = det_pass_str in ("true", "1", "yes")
    det_skip = det_pass_str in ("skip", "")
    pkg_pass_str = str(getattr(args, "package_verify_pass", "") or "").lower()
    pkg_pass = pkg_pass_str in ("true", "1", "yes")

    action = "hold"
    reason = ""

    if target_channel == "dev":
        action = "register_dev"
        reason = "registered as dev candidate"
    elif target_channel == "canary":
        if perf_pass and risk_guard_pass:
            action = "promote_to_canary"
            reason = "perf+risk gates passed; promoting to canary"
            canary_id = model_id
            canary_path = str(Path(args.candidate_model))
        else:
            action = "hold"
            reason = f"canary promotion blocked: perf={perf_pass} risk={risk_guard_pass}"
    elif target_channel == "stable":
        if final_decision in ("promote", "hold_for_promote"):
            if milestone_present and milestone_pass and canary_pass:
                action = "promote_to_stable"
                reason = "all gates passed; promoting to stable"
                stable_id = model_id
                stable_path = str(Path(args.candidate_model))
            else:
                action = "hold"
                reason = f"stable promotion blocked: milestone={milestone_pass} canary={canary_pass}"
        elif final_decision == "rolled_back":
            action = "rollback_to_stable"
            reason = decision_payload.get("reason") or "rollback triggered"
        else:
            action = "hold"
            reason = f"decision={final_decision}; not promoting"

    return {
        "schema": "release_state_v1",
        "generated_at": _now_iso(),
        "channel": target_channel,
        "action": action,
        "reason": reason,
        "stable_model_id": stable_id,
        "stable_model_path": stable_path,
        "canary_model_id": canary_id,
        "canary_model_path": canary_path,
        "dev_model_id": model_id if target_channel == "dev" else "",
        "perf_gate_pass": perf_pass,
        "risk_guard_pass": risk_guard_pass,
        "milestone_pass": milestone_pass if milestone_present else None,
        "canary_pass": canary_pass,
        "determinism_pass": det_pass if not det_skip else "skip",
        "package_verify_pass": pkg_pass,
    }


if __name__ == "__main__":
    raise SystemExit(main())
