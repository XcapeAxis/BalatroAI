from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pick_top_policy(summary_rows: list[dict[str, Any]], candidate_policy: str | None) -> dict[str, Any] | None:
    if not summary_rows:
        return None
    if candidate_policy:
        token = str(candidate_policy).strip().lower()
        for row in summary_rows:
            if str(row.get("policy_id") or "").strip().lower() == token:
                return row
    return summary_rows[0]


def _find_policy(summary_rows: list[dict[str, Any]], policy_id: str | None) -> dict[str, Any] | None:
    if not policy_id:
        return None
    token = str(policy_id).strip().lower()
    for row in summary_rows:
        if str(row.get("policy_id") or "").strip().lower() == token:
            return row
    return None


def _read_summary_rows(path: Path) -> list[dict[str, Any]]:
    obj = _read_json(path)
    if not isinstance(obj, list):
        return []
    out: list[dict[str, Any]] = []
    for row in obj:
        if isinstance(row, dict):
            out.append(row)
    return out


def _legacy_champion_policy(champion_payload: dict[str, Any]) -> str | None:
    for key in ("policy_id", "champion_policy_id", "exp_id"):
        token = str(champion_payload.get(key) or "").strip()
        if token:
            return token
    return None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    qv = min(1.0, max(0.0, float(q)))
    pos = (len(ordered) - 1) * qv
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _bootstrap_mean_diff(
    a: list[float],
    b: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(a) < 2 or len(b) < 2:
        return {
            "status": "insufficient_samples",
            "mean_diff": _mean(a) - _mean(b),
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    rng = random.Random(int(seed))
    diffs: list[float] = []
    for _ in range(max(32, int(iterations))):
        sample_a = [_safe_float(a[rng.randrange(0, len(a))], 0.0) for _ in range(len(a))]
        sample_b = [_safe_float(b[rng.randrange(0, len(b))], 0.0) for _ in range(len(b))]
        diffs.append(_mean(sample_a) - _mean(sample_b))
    return {
        "status": "ok",
        "mean_diff": _mean(a) - _mean(b),
        "ci_low": _quantile(diffs, 0.025),
        "ci_high": _quantile(diffs, 0.975),
    }


def _extract_slice_label(row: dict[str, Any], slice_key: str) -> str:
    labels = row.get("slice_labels") if isinstance(row.get("slice_labels"), dict) else {}
    token = labels.get(slice_key)
    if token not in {None, ""}:
        return str(token)
    bucket_counts = row.get("bucket_counts") if isinstance(row.get("bucket_counts"), dict) else {}
    bucket = bucket_counts.get(slice_key) if isinstance(bucket_counts.get(slice_key), dict) else {}
    if bucket:
        ordered = sorted(bucket.items(), key=lambda kv: (-int(_safe_float(kv[1], 0.0)), str(kv[0])))
        if ordered:
            return str(ordered[0][0])
    return "unknown"


def _build_slice_breakdown(
    *,
    episode_rows: list[dict[str, Any]],
    candidate_policy: str,
    champion_policy: str,
    bootstrap_iterations: int,
    min_samples: int,
) -> dict[str, Any]:
    slice_keys = [
        "slice_resource_pressure",
        "slice_action_type",
        "slice_stage",
        "slice_position_sensitive",
        "slice_stateful_joker_present",
    ]
    candidate_rows = [r for r in episode_rows if str(r.get("policy_id") or "") == candidate_policy]
    champion_rows = [r for r in episode_rows if str(r.get("policy_id") or "") == champion_policy]

    payload_rows: list[dict[str, Any]] = []
    for slice_key in slice_keys:
        labels = sorted(
            {
                _extract_slice_label(r, slice_key)
                for r in candidate_rows + champion_rows
            }
        )
        for label in labels:
            cand_subset = [r for r in candidate_rows if _extract_slice_label(r, slice_key) == label]
            champ_subset = [r for r in champion_rows if _extract_slice_label(r, slice_key) == label]
            cand_scores = [_safe_float(r.get("total_score"), 0.0) for r in cand_subset]
            champ_scores = [_safe_float(r.get("total_score"), 0.0) for r in champ_subset]
            cand_rounds = [_safe_float(r.get("rounds_survived"), 0.0) for r in cand_subset]
            champ_rounds = [_safe_float(r.get("rounds_survived"), 0.0) for r in champ_subset]
            cand_wins = [_safe_float(r.get("win_proxy"), 0.0) for r in cand_subset]
            champ_wins = [_safe_float(r.get("win_proxy"), 0.0) for r in champ_subset]

            score_ci = _bootstrap_mean_diff(
                cand_scores,
                champ_scores,
                iterations=bootstrap_iterations,
                seed=(hash(f"{slice_key}:{label}:score") & 0xFFFFFFFF),
            )
            rounds_ci = _bootstrap_mean_diff(
                cand_rounds,
                champ_rounds,
                iterations=bootstrap_iterations,
                seed=(hash(f"{slice_key}:{label}:rounds") & 0xFFFFFFFF),
            )
            win_ci = _bootstrap_mean_diff(
                cand_wins,
                champ_wins,
                iterations=bootstrap_iterations,
                seed=(hash(f"{slice_key}:{label}:wins") & 0xFFFFFFFF),
            )
            ci_status = (
                "ok"
                if score_ci.get("status") == "ok" and rounds_ci.get("status") == "ok" and win_ci.get("status") == "ok"
                else "insufficient_samples"
            )
            degraded_significant = (
                ci_status == "ok"
                and (
                    float(score_ci.get("ci_high") or 0.0) < 0.0
                    or float(win_ci.get("ci_high") or 0.0) < 0.0
                )
            )
            improved_significant = (
                ci_status == "ok"
                and (
                    float(score_ci.get("ci_low") or 0.0) > 0.0
                    or float(win_ci.get("ci_low") or 0.0) > 0.0
                )
            )
            payload_rows.append(
                {
                    "slice_key": slice_key,
                    "slice_label": label,
                    "candidate_count": int(len(cand_subset)),
                    "champion_count": int(len(champ_subset)),
                    "sufficient_samples": bool(len(cand_subset) >= min_samples and len(champ_subset) >= min_samples),
                    "ci_status": ci_status,
                    "metrics": {
                        "mean_total_score_delta": float(score_ci.get("mean_diff") or 0.0),
                        "mean_rounds_survived_delta": float(rounds_ci.get("mean_diff") or 0.0),
                        "win_rate_delta": float(win_ci.get("mean_diff") or 0.0),
                    },
                    "ci": {
                        "mean_total_score": {"low": float(score_ci.get("ci_low") or 0.0), "high": float(score_ci.get("ci_high") or 0.0)},
                        "mean_rounds_survived": {"low": float(rounds_ci.get("ci_low") or 0.0), "high": float(rounds_ci.get("ci_high") or 0.0)},
                        "win_rate": {"low": float(win_ci.get("ci_low") or 0.0), "high": float(win_ci.get("ci_high") or 0.0)},
                    },
                    "signals": {
                        "degraded_significant": bool(degraded_significant),
                        "improved_significant": bool(improved_significant),
                    },
                }
            )
    payload_rows.sort(
        key=lambda row: (
            not bool((row.get("signals") or {}).get("degraded_significant")),
            -abs(_safe_float(((row.get("metrics") or {}).get("mean_total_score_delta")), 0.0)),
            str(row.get("slice_key") or ""),
            str(row.get("slice_label") or ""),
        )
    )
    return {
        "schema": "p41_slice_decision_breakdown_v1",
        "generated_at": _now_iso(),
        "candidate_policy_id": candidate_policy,
        "champion_policy_id": champion_policy,
        "rows": payload_rows,
    }


def _build_slice_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Slice Decision Breakdown",
        "",
        f"- candidate_policy: `{payload.get('candidate_policy_id')}`",
        f"- champion_policy: `{payload.get('champion_policy_id')}`",
        "",
        "| slice_key | slice_label | cand_n | champ_n | ci_status | score_delta | score_ci_low | score_ci_high | win_delta | degraded_sig | improved_sig |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("rows") if isinstance(payload.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        ci = row.get("ci") if isinstance(row.get("ci"), dict) else {}
        score_ci = ci.get("mean_total_score") if isinstance(ci.get("mean_total_score"), dict) else {}
        signals = row.get("signals") if isinstance(row.get("signals"), dict) else {}
        lines.append(
            "| {slice_key} | {slice_label} | {cand_n} | {champ_n} | {ci_status} | {score_delta:.6f} | {score_low:.6f} | {score_high:.6f} | {win_delta:.6f} | {degraded} | {improved} |".format(
                slice_key=row.get("slice_key"),
                slice_label=row.get("slice_label"),
                cand_n=int(row.get("candidate_count") or 0),
                champ_n=int(row.get("champion_count") or 0),
                ci_status=row.get("ci_status"),
                score_delta=_safe_float(metrics.get("mean_total_score_delta"), 0.0),
                score_low=_safe_float(score_ci.get("low"), 0.0),
                score_high=_safe_float(score_ci.get("high"), 0.0),
                win_delta=_safe_float(metrics.get("win_rate_delta"), 0.0),
                degraded=str(bool(signals.get("degraded_significant", False))).lower(),
                improved=str(bool(signals.get("improved_significant", False))).lower(),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Candidate Decision",
        "",
        f"- generated_at: {decision.get('generated_at')}",
        f"- decision: `{decision.get('decision')}`",
        f"- recommend_promotion: `{decision.get('recommend_promotion')}`",
        f"- candidate_policy: `{decision.get('candidate_policy_id')}`",
        f"- champion_policy: `{decision.get('champion_policy_id')}`",
        "",
        "## Key Deltas",
        "",
    ]
    deltas = decision.get("deltas") if isinstance(decision.get("deltas"), dict) else {}
    for key in ("mean_total_score", "win_rate", "invalid_action_rate", "timeout_rate"):
        if key in deltas:
            lines.append(f"- {key}: {deltas.get(key):.6f}")
    lines.append("")
    lines.append("## Reasons")
    reasons = decision.get("reasons") if isinstance(decision.get("reasons"), list) else []
    if reasons:
        lines.extend([f"- {str(reason)}" for reason in reasons])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Risks")
    risks = decision.get("risks") if isinstance(decision.get("risks"), list) else []
    if risks:
        lines.extend([f"- {str(risk)}" for risk in risks])
    else:
        lines.append("- none")
    lines.append("")
    slice_ref = str(decision.get("slice_decision_breakdown_json") or "").strip()
    if slice_ref:
        lines.append(f"- slice_decision_breakdown_json: `{slice_ref}`")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Champion/candidate decision rules (slice-aware v2).")
    parser.add_argument("--summary-json", required=True, help="Path to summary_table.json from arena runner.")
    parser.add_argument("--champion-json", default="docs/artifacts/p22/champion.json")
    parser.add_argument("--episode-records-jsonl", default="", help="Optional episode_records.jsonl for slice-aware analysis.")
    parser.add_argument("--bucket-metrics-json", default="", help="Optional bucket_metrics.json (reference only).")
    parser.add_argument("--out-dir", default="docs/artifacts/p39")
    parser.add_argument("--candidate-policy", default="", help="Optional candidate policy id.")
    parser.add_argument("--champion-policy", default="", help="Optional champion policy id override.")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--max-invalid-increase", type=float, default=0.02)
    parser.add_argument("--max-timeout-increase", type=float, default=0.01)
    parser.add_argument("--min-score-improvement", type=float, default=0.01)
    parser.add_argument("--max-score-regression", type=float, default=0.02)
    parser.add_argument("--min-win-improvement", type=float, default=0.00)
    parser.add_argument("--bootstrap-iterations", type=int, default=250)
    parser.add_argument("--slice-min-samples", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (repo_root / summary_path).resolve()
    if not summary_path.exists():
        raise SystemExit(f"summary json not found: {summary_path}")

    champion_path = Path(args.champion_json)
    if not champion_path.is_absolute():
        champion_path = (repo_root / champion_path).resolve()

    episode_records_path = Path(args.episode_records_jsonl) if str(args.episode_records_jsonl).strip() else (summary_path.parent / "episode_records.jsonl")
    if not episode_records_path.is_absolute():
        episode_records_path = (repo_root / episode_records_path).resolve()

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    out_dir = out_root / f"champion_eval_{_now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_summary_rows(summary_path)
    candidate = _pick_top_policy(summary_rows, args.candidate_policy or None)
    champion_payload = _read_json(champion_path) if champion_path.exists() else None
    if not isinstance(champion_payload, dict):
        champion_payload = {}
    champion_policy_id = args.champion_policy or _legacy_champion_policy(champion_payload)

    champion_row = _find_policy(summary_rows, champion_policy_id)
    reasons: list[str] = []
    risks: list[str] = []
    deltas: dict[str, float] = {}

    decision = "hold"
    recommend_promotion = False
    candidate_policy_id = str((candidate or {}).get("policy_id") or "")

    if candidate is None:
        decision = "hold"
        reasons.append("no candidate row found in summary table")
    else:
        seed_count = _safe_int(candidate.get("seed_count"), 0)
        if seed_count < int(args.min_seeds):
            decision = "observe"
            reasons.append(f"candidate seed_count={seed_count} below min_seeds={int(args.min_seeds)}")
            risks.append("sample_size_insufficient")

        if champion_row is None:
            if not champion_policy_id:
                risks.append("champion_policy_not_specified")
            else:
                risks.append(f"champion_policy_missing_in_current_summary:{champion_policy_id}")
            if decision != "observe":
                decision = "observe"
                reasons.append("champion baseline unavailable in this arena run")
        else:
            score_delta = _safe_float(candidate.get("mean_total_score")) - _safe_float(champion_row.get("mean_total_score"))
            win_delta = _safe_float(candidate.get("win_rate")) - _safe_float(champion_row.get("win_rate"))
            invalid_delta = _safe_float(candidate.get("invalid_action_rate")) - _safe_float(champion_row.get("invalid_action_rate"))
            timeout_delta = _safe_float(candidate.get("timeout_rate")) - _safe_float(champion_row.get("timeout_rate"))
            deltas.update(
                {
                    "mean_total_score": float(score_delta),
                    "win_rate": float(win_delta),
                    "invalid_action_rate": float(invalid_delta),
                    "timeout_rate": float(timeout_delta),
                }
            )

            hard_ok = True
            if invalid_delta > float(args.max_invalid_increase):
                hard_ok = False
                reasons.append(
                    f"invalid_action_rate delta={invalid_delta:.6f} exceeds max_invalid_increase={float(args.max_invalid_increase):.6f}"
                )
            if timeout_delta > float(args.max_timeout_increase):
                hard_ok = False
                reasons.append(
                    f"timeout_rate delta={timeout_delta:.6f} exceeds max_timeout_increase={float(args.max_timeout_increase):.6f}"
                )

            score_good = score_delta >= float(args.min_score_improvement)
            score_not_bad = score_delta >= -float(args.max_score_regression)
            win_good = win_delta >= float(args.min_win_improvement)

            if hard_ok and score_good:
                decision = "promote"
                recommend_promotion = True
                reasons.append(
                    f"score delta={score_delta:.6f} >= min_score_improvement={float(args.min_score_improvement):.6f}"
                )
            elif hard_ok and win_good and score_not_bad:
                decision = "promote"
                recommend_promotion = True
                reasons.append("win_rate improved and score does not regress beyond threshold")
            elif decision != "observe":
                decision = "hold"
                reasons.append("candidate does not satisfy promotion criteria")

            if _safe_float(candidate.get("std_total_score")) > max(1.0, abs(_safe_float(candidate.get("mean_total_score"))) * 0.6):
                risks.append("candidate_variance_high")

    slice_breakdown_payload: dict[str, Any] = {
        "schema": "p41_slice_decision_breakdown_v1",
        "generated_at": _now_iso(),
        "candidate_policy_id": candidate_policy_id,
        "champion_policy_id": str(champion_policy_id or ""),
        "rows": [],
        "ci_status": "insufficient_samples",
    }
    if episode_records_path.exists() and candidate_policy_id and champion_policy_id:
        episode_rows = _read_jsonl(episode_records_path)
        if episode_rows:
            slice_breakdown_payload = _build_slice_breakdown(
                episode_rows=episode_rows,
                candidate_policy=candidate_policy_id,
                champion_policy=str(champion_policy_id),
                bootstrap_iterations=max(32, int(args.bootstrap_iterations)),
                min_samples=max(2, int(args.slice_min_samples)),
            )
            if any(
                bool(((row.get("signals") or {}).get("degraded_significant")))
                for row in (slice_breakdown_payload.get("rows") if isinstance(slice_breakdown_payload.get("rows"), list) else [])
            ):
                slice_breakdown_payload["ci_status"] = "ok"
            else:
                slice_breakdown_payload["ci_status"] = "mixed"
        else:
            risks.append("episode_records_empty")
    else:
        risks.append("episode_records_missing_for_slice_aware")

    degraded_rows = [
        row
        for row in (slice_breakdown_payload.get("rows") if isinstance(slice_breakdown_payload.get("rows"), list) else [])
        if isinstance(row, dict) and bool((row.get("signals") or {}).get("degraded_significant"))
    ]
    critical_degraded = [
        row
        for row in degraded_rows
        if str(row.get("slice_key") or "") == "slice_resource_pressure" and str(row.get("slice_label") or "") in {"high", "resource_tight"}
    ]
    if critical_degraded and decision == "promote":
        decision = "observe"
        recommend_promotion = False
        reasons.append("critical slice degraded: high resource pressure")
        risks.append("slice_regression_critical")
    elif degraded_rows and decision == "promote":
        decision = "observe"
        recommend_promotion = False
        reasons.append("slice degradation detected despite global uplift")
        risks.append("slice_regression_detected")

    improved_rows = [
        row
        for row in (slice_breakdown_payload.get("rows") if isinstance(slice_breakdown_payload.get("rows"), list) else [])
        if isinstance(row, dict) and bool((row.get("signals") or {}).get("improved_significant"))
    ]
    if improved_rows and decision == "hold":
        decision = "observe"
        reasons.append("some slices improved significantly; keep under observation")

    slice_json_path = out_dir / "slice_decision_breakdown.json"
    slice_md_path = out_dir / "slice_decision_breakdown.md"
    slice_json_path.write_text(json.dumps(slice_breakdown_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    slice_md_path.write_text(_build_slice_markdown(slice_breakdown_payload), encoding="utf-8")

    payload = {
        "schema": "p41_candidate_decision_v2",
        "generated_at": _now_iso(),
        "summary_json": str(summary_path),
        "champion_json": str(champion_path),
        "episode_records_jsonl": str(episode_records_path) if episode_records_path.exists() else "",
        "decision": decision,
        "recommend_promotion": bool(recommend_promotion),
        "candidate_policy_id": candidate_policy_id,
        "champion_policy_id": str(champion_policy_id or ""),
        "candidate_row": candidate,
        "champion_row": champion_row,
        "deltas": deltas,
        "reasons": reasons,
        "risks": risks,
        "slice_decision_breakdown_json": str(slice_json_path),
        "slice_decision_breakdown_md": str(slice_md_path),
        "thresholds": {
            "min_seeds": int(args.min_seeds),
            "max_invalid_increase": float(args.max_invalid_increase),
            "max_timeout_increase": float(args.max_timeout_increase),
            "min_score_improvement": float(args.min_score_improvement),
            "max_score_regression": float(args.max_score_regression),
            "min_win_improvement": float(args.min_win_improvement),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "slice_min_samples": int(args.slice_min_samples),
        },
    }

    json_path = out_dir / "candidate_decision.json"
    md_path = out_dir / "candidate_decision.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "recommend_promotion": recommend_promotion,
                "json": str(json_path),
                "md": str(md_path),
                "slice_json": str(slice_json_path),
                "slice_md": str(slice_md_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

