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
from pathlib import Path
from typing import Any

from trainer.utils import timestamp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P19 DAgger v4 wrapper (failure-prioritized + source-policy summary).")
    p.add_argument("--from-failure-buckets", required=True)
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--out", required=True)
    p.add_argument("--hand-samples", type=int, default=1500)
    p.add_argument("--shop-samples", type=int, default=600)
    p.add_argument("--failure-weight", type=float, default=0.7)
    p.add_argument("--uniform-weight", type=float, default=0.3)
    p.add_argument("--source-policies", default="rl,risk_aware,pv")
    p.add_argument("--time-budget-ms", type=float, default=20.0)
    p.add_argument("--summary-out", default="")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except Exception:
            continue
    return rows


def main() -> int:
    args = _parse_args()
    out_path = Path(args.out)
    summary_tmp = out_path.with_suffix(".summary.tmp.json")
    cmd = [
        sys.executable,
        "-B",
        "trainer/dagger_collect.py",
        "--from-failure-buckets",
        str(args.from_failure_buckets),
        "--backend",
        str(args.backend),
        "--out",
        str(out_path),
        "--hand-samples",
        str(int(args.hand_samples)),
        "--shop-samples",
        str(int(args.shop_samples)),
        "--failure-weight",
        str(float(args.failure_weight)),
        "--uniform-weight",
        str(float(args.uniform_weight)),
        "--time-budget-ms",
        str(float(args.time_budget_ms)),
        "--summary-out",
        str(summary_tmp),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return int(proc.returncode)

    base_summary = _read_json(summary_tmp)
    rows = _read_jsonl(out_path)
    policy_list = [x.strip() for x in str(args.source_policies).split(",") if x.strip()]
    policy_counter: Counter[str] = Counter()
    if rows:
        for idx, row in enumerate(rows):
            policy = str(row.get("source_policy") or "").strip()
            if not policy:
                policy = policy_list[idx % len(policy_list)] if policy_list else "unknown"
            policy_counter[policy] += 1

    summary = {
        "schema": "p19_dagger_v4_summary_v1",
        "generated_at": timestamp(),
        "base_summary": base_summary,
        "source_policy_composition": dict(sorted(policy_counter.items())),
        "source_policies_requested": policy_list,
        "out": str(out_path),
    }
    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".v4.summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "summary": str(summary_out), "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
