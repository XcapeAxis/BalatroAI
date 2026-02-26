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
    p.add_argument("--decision-out", required=True)
    p.add_argument("--median-threshold", type=float, default=0.5)
    p.add_argument("--avg-threshold", type=float, default=0.3)
    p.add_argument("--allow-promote", action="store_true", help="Actually update current_champion.json.")
    p.add_argument("--git-commit", default="")
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

    if perf_pass and risk_guard_pass:
        final_decision = "promote" if args.allow_promote else "hold_for_promote"
        reason = "perf gate passed and risk guard passed"
    elif perf_pass and (not risk_guard_pass):
        final_decision = "hold_for_more_data"
        reason = "risk guard blocked promotion"
    else:
        final_decision = "reject"
        reason = "perf gate did not pass thresholds"

    model_id = args.candidate_model_id or f"p17_pv_challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    decision_payload = {
        "schema": "champion_decision_v2",
        "generated_at": _now_iso(),
        "decision": final_decision,
        "final_decision": final_decision,
        "reason": reason,
        "perf_gate_pass": perf_pass,
        "risk_guard_pass": risk_guard_pass,
        "decision_reasons": [reason],
        "risk_regressions": regressions,
        "thresholds": {
            "median_ante_delta": median_thr,
            "avg_ante_delta": avg_thr,
            "ci_lower_gt_zero": True,
            "critical_bucket_regression_max_pct_points": risk_thr,
        },
        "metrics": {
            "baseline": b,
            "candidate": c,
            "delta": {"avg_ante": delta_avg, "median_ante": delta_med, "win_rate": delta_win, "avg_ante_ci95_low": ci_lb},
        },
        "artifacts": {
            "baseline_eval": str(Path(args.baseline)),
            "candidate_eval": str(Path(args.candidate)),
            "compare_summary": str(Path(args.compare_summary)) if args.compare_summary else None,
            "candidate_model": str(Path(args.candidate_model)),
        },
        "model_id": model_id,
        "dataset_id": args.dataset_id or None,
        "git_commit": args.git_commit or None,
    }
    _write_json(Path(args.decision_out), decision_payload)
    md_path = Path(args.decision_out).with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                "# P17 Champion Decision",
                "",
                f"- decision: {decision_payload['decision']}",
                f"- reason: {decision_payload['reason']}",
                f"- perf_gate_pass: {decision_payload['perf_gate_pass']}",
                f"- risk_guard_pass: {decision_payload['risk_guard_pass']}",
                f"- delta_avg_ante: {delta_avg:.4f}",
                f"- delta_median_ante: {delta_med:.4f}",
                f"- delta_win_rate: {delta_win:.4f}",
                f"- ci95_low(avg_ante_delta): {ci_lb}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    current_path = reg_root / "current_champion.json"
    if final_decision == "promote":
        _write_json(
            current_path,
            {
                "schema": "p17_current_champion_v1",
                "updated_at": _now_iso(),
                "model_id": model_id,
                "model_path": str(Path(args.candidate_model)),
                "source_decision": str(Path(args.decision_out)),
            },
        )

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


if __name__ == "__main__":
    raise SystemExit(main())
