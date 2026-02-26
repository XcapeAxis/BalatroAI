from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from trainer.utils import setup_logger, warn_if_unstable_python


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _health_ok(base_url: str) -> bool:
    body = {"jsonrpc": "2.0", "id": 1, "method": "health", "params": {}}
    try:
        r = requests.post(base_url, json=body, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P17 real shadow canary recorder (readonly).")
    p.add_argument("--base-url", default="http://127.0.0.1:12346")
    p.add_argument("--model", default="")
    p.add_argument("--models", default="", help="Optional multi-policy model map, e.g. pv=...,hybrid=...,rl=...")
    p.add_argument("--risk-aware-config", default="", help="Optional risk-aware config path for divergence report.")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logger = setup_logger("trainer.real_shadow_canary")
    warn_if_unstable_python(logger)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _health_ok(args.base_url):
        skip = {
            "schema": "p17_real_canary_skip_v1",
            "status": "SKIPPED",
            "reason": "real unavailable",
            "base_url": args.base_url,
            "generated_at": _now_iso(),
        }
        (out_dir / "canary_skip.json").write_text(json.dumps(skip, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(skip, ensure_ascii=False))
        return 0

    session_path = out_dir / f"canary_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    cmd = [
        sys.executable,
        "-B",
        "trainer/record_real_session.py",
        "--base-url",
        args.base_url,
        "--steps",
        str(args.steps),
        "--interval",
        str(args.interval),
        "--topk",
        str(args.topk),
        "--out",
        str(session_path),
    ]
    if args.model:
        cmd += ["--model", args.model]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out_dir / "record_stdout.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        fail = {
            "schema": "p17_real_canary_v1",
            "status": "FAIL",
            "reason": f"record_real_session failed rc={proc.returncode}",
            "base_url": args.base_url,
            "generated_at": _now_iso(),
        }
        (out_dir / "canary_summary.json").write_text(json.dumps(fail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fail, ensure_ascii=False))
        return 1

    rows = _read_jsonl(session_path)
    phase_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    divergence_total = 0
    divergence_same_top1 = 0
    risk_fallback_count = 0
    high_risk_states = 0
    for row in rows:
        phase_counter[str(row.get("phase") or "UNKNOWN")] += 1
        topk = row.get("model_suggestions_topk") or []
        if isinstance(topk, list) and topk:
            first = topk[0] if isinstance(topk[0], dict) else {}
            action_counter[str(first.get("action_type") or first.get("action") or "UNKNOWN")] += 1
            if args.models:
                a1 = str(first.get("action_type") or "WAIT")
                a2 = a1
                if len(topk) > 1 and isinstance(topk[1], dict):
                    a2 = str(topk[1].get("action_type") or a1)
                step_idx = int(row.get("step_idx") or 0)
                pv = a1
                hybrid = a2
                rl = a2 if (step_idx % 2 == 1 and a2) else a1
                risk_score = 0.2 + (0.6 if step_idx % 5 == 0 else 0.0)
                if risk_score >= 0.75:
                    risk_action = "WAIT" if str(row.get("phase") or "").upper() not in {"SELECTING_HAND", "SHOP"} else hybrid
                    risk_fallback_count += 1
                    high_risk_states += 1
                elif risk_score >= 0.45:
                    risk_action = hybrid
                else:
                    risk_action = rl
                chosen = {"pv": pv, "hybrid": hybrid, "rl": rl, "risk_aware": risk_action}
                vals = list(chosen.values())
                divergence_total += 1
                if len(set(vals)) == 1:
                    divergence_same_top1 += 1

    advice = {
        "schema": "p17_canary_advice_distribution_v1",
        "generated_at": _now_iso(),
        "phase_distribution": dict(sorted(phase_counter.items())),
        "top1_action_distribution": dict(sorted(action_counter.items())),
        "steps": len(rows),
        "session": str(session_path),
    }
    (out_dir / "advice_distribution.json").write_text(json.dumps(advice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.models:
        div = {
            "schema": "p19_canary_divergence_v1",
            "generated_at": _now_iso(),
            "steps": divergence_total,
            "top1_agreement_rate": (divergence_same_top1 / divergence_total) if divergence_total else 0.0,
            "top1_divergence_rate": (1.0 - (divergence_same_top1 / divergence_total)) if divergence_total else 0.0,
            "risk_aware_fallback_rate": (risk_fallback_count / divergence_total) if divergence_total else 0.0,
            "high_risk_state_rate": (high_risk_states / divergence_total) if divergence_total else 0.0,
            "models_arg": args.models,
            "metrics_source": "synthetic",
            "metrics_note": "Divergence and risk_aware rates are derived from synthetic rules (topk[0]/topk[1] and step-based risk_score), not from separate pv/rl/risk_aware model inference.",
        }
        (out_dir / "canary_divergence_summary.json").write_text(json.dumps(div, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out_dir / "canary_divergence_summary.md").write_text(
            "\n".join(
                [
                    "# Canary Divergence Summary",
                    "",
                    f"- steps: {div['steps']}",
                    f"- top1_agreement_rate: {div['top1_agreement_rate']:.4f}",
                    f"- top1_divergence_rate: {div['top1_divergence_rate']:.4f}",
                    f"- risk_aware_fallback_rate: {div['risk_aware_fallback_rate']:.4f}",
                    f"- high_risk_state_rate: {div['high_risk_state_rate']:.4f}",
                    "",
                    f"- **metrics_source**: {div['metrics_source']}",
                    f"- **metrics_note**: {div['metrics_note']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    drift_cmd = [
        sys.executable,
        "-B",
        "trainer/sim_real_drift.py",
        "--base-url",
        args.base_url,
        "--samples",
        "10",
        "--interval",
        "0.2",
        "--out",
        str(out_dir / "drift_summary.json"),
    ]
    subprocess.run(drift_cmd, capture_output=True, text=True, check=False)

    summary = {
        "schema": "p17_real_canary_v1",
        "status": "PASS",
        "generated_at": _now_iso(),
        "base_url": args.base_url,
        "steps": len(rows),
        "session": str(session_path),
        "advice_distribution": str(out_dir / "advice_distribution.json"),
        "drift_summary": str(out_dir / "drift_summary.json"),
        "divergence_summary": str(out_dir / "canary_divergence_summary.json") if args.models else "",
        "divergence_metrics_source": "synthetic" if args.models else "",
    }
    (out_dir / "canary_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
