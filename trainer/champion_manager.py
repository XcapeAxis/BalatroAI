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
    p = argparse.ArgumentParser(description="P17 champion/challenger decision manager.")
    p.add_argument("--registry-root", default="docs/artifacts/p17/registry")
    p.add_argument("--baseline", required=True, help="Champion eval json (100 seeds).")
    p.add_argument("--candidate", required=True, help="Challenger eval json (100 or 500 seeds).")
    p.add_argument("--compare-summary", default="", help="Optional eval_compare summary json path.")
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
    if isinstance(compare.get("delta"), dict):
        ci = compare.get("delta", {}).get("avg_ante_ci95")
        if isinstance(ci, dict):
            ci_lb = ci.get("low")

    perf_pass = bool(
        (delta_med >= float(args.median_threshold))
        or (delta_avg >= float(args.avg_threshold))
        or (ci_lb is not None and float(ci_lb) > 0.0)
    )

    if perf_pass:
        decision = "promote" if args.allow_promote else "hold_for_promote"
        reason = "perf gate passed"
    else:
        decision = "reject"
        reason = "perf gate did not pass thresholds"

    model_id = args.candidate_model_id or f"p17_pv_challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    decision_payload = {
        "schema": "p17_promotion_decision_v1",
        "generated_at": _now_iso(),
        "decision": decision,
        "reason": reason,
        "perf_gate_pass": perf_pass,
        "thresholds": {
            "median_ante_delta": float(args.median_threshold),
            "avg_ante_delta": float(args.avg_threshold),
            "ci_lower_gt_zero": True,
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
    if decision == "promote":
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

    print(json.dumps({"decision": decision, "perf_gate_pass": perf_pass, "decision_out": str(Path(args.decision_out))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

