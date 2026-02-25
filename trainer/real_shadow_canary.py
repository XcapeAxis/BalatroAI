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
    for row in rows:
        phase_counter[str(row.get("phase") or "UNKNOWN")] += 1
        topk = row.get("model_suggestions_topk") or []
        if isinstance(topk, list) and topk:
            first = topk[0] if isinstance(topk[0], dict) else {}
            action_counter[str(first.get("action_type") or first.get("action") or "UNKNOWN")] += 1

    advice = {
        "schema": "p17_canary_advice_distribution_v1",
        "generated_at": _now_iso(),
        "phase_distribution": dict(sorted(phase_counter.items())),
        "top1_action_distribution": dict(sorted(action_counter.items())),
        "steps": len(rows),
        "session": str(session_path),
    }
    (out_dir / "advice_distribution.json").write_text(json.dumps(advice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
    }
    (out_dir / "canary_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

