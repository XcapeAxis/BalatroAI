from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
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


def _build_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# P39 Candidate Decision",
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
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P39 champion/candidate decision rules.")
    parser.add_argument("--summary-json", required=True, help="Path to summary_table.json from arena runner.")
    parser.add_argument("--champion-json", default="docs/artifacts/p22/champion.json")
    parser.add_argument("--out-dir", default="docs/artifacts/p39")
    parser.add_argument("--candidate-policy", default="", help="Optional candidate policy id.")
    parser.add_argument("--champion-policy", default="", help="Optional champion policy id override.")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--max-invalid-increase", type=float, default=0.02)
    parser.add_argument("--max-timeout-increase", type=float, default=0.01)
    parser.add_argument("--min-score-improvement", type=float, default=0.01)
    parser.add_argument("--max-score-regression", type=float, default=0.02)
    parser.add_argument("--min-win-improvement", type=float, default=0.00)
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

    payload = {
        "schema": "p39_candidate_decision_v1",
        "generated_at": _now_iso(),
        "summary_json": str(summary_path),
        "champion_json": str(champion_path),
        "decision": decision,
        "recommend_promotion": bool(recommend_promotion),
        "candidate_policy_id": candidate_policy_id,
        "champion_policy_id": str(champion_policy_id or ""),
        "candidate_row": candidate,
        "champion_row": champion_row,
        "deltas": deltas,
        "reasons": reasons,
        "risks": risks,
        "thresholds": {
            "min_seeds": int(args.min_seeds),
            "max_invalid_increase": float(args.max_invalid_increase),
            "max_timeout_increase": float(args.max_timeout_increase),
            "min_score_improvement": float(args.min_score_improvement),
            "max_score_regression": float(args.max_score_regression),
            "min_win_improvement": float(args.min_win_improvement),
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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

