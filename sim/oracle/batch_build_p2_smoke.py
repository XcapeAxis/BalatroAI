from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.generate_p2_trace import P2_TARGETS, generate_trace
from trainer.env_client import health

SCOPE_CHOICES = [
    "p2_hand_score_observed_core",
    "p1_hand_score_observed_core",
    "p0_hand_score_observed_core",
    "hand_core",
    "score_core",
    "full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-build P2 joker smoke fixtures and run directed diff.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p2_smoke/")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="p2_hand_score_observed_core")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--targets", default=None, help="Comma-separated target subset")
    parser.add_argument("--dump-on-diff", default=None)
    return parser.parse_args()


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _parse_diff_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "first_diff_step": None,
        "first_diff_path": None,
        "oracle_hash": None,
        "sim_hash": None,
        "dumped_oracle": None,
        "dumped_sim": None,
    }
    for line in text.splitlines():
        line = line.strip()
        m = re.search(r"MISMATCH step=(\d+)", line)
        if m and result["first_diff_step"] is None:
            result["first_diff_step"] = int(m.group(1))
            continue
        if line.startswith("first_diff_path=") and result["first_diff_path"] is None:
            result["first_diff_path"] = line.split("=", 1)[1]
            continue
        if line.startswith("oracle_hash=") and result["oracle_hash"] is None:
            result["oracle_hash"] = line.split("=", 1)[1]
            continue
        if line.startswith("sim_hash=") and result["sim_hash"] is None:
            result["sim_hash"] = line.split("=", 1)[1]
            continue
        if line.startswith("dumped_oracle="):
            tail = line.split("=", 1)[1]
            if ", dumped_sim=" in tail:
                left, right = tail.split(", dumped_sim=", 1)
                result["dumped_oracle"] = left.strip() or None
                result["dumped_sim"] = right.strip() or None
            else:
                result["dumped_oracle"] = tail.strip() or None
    return result


def _parse_targets(targets_csv: str | None) -> list[str]:
    if not targets_csv:
        return sorted(P2_TARGETS.keys())
    out: list[str] = []
    for token in str(targets_csv).split(","):
        name = token.strip()
        if not name:
            continue
        if name not in P2_TARGETS:
            raise ValueError(f"invalid target: {name}")
        if name not in out:
            out.append(name)
    return out


def _summarize(row: dict[str, Any]) -> str:
    reason = str(row.get("failure_reason") or "")[:100]
    return (
        f"[P2] {row.get('target')} | {row.get('status')} | "
        f"scope={row.get('scope_used')} | steps={row.get('steps_used') or 0} | {reason}"
    )


def main() -> int:
    args = parse_args()
    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    try:
        targets = _parse_targets(args.targets)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_oracle_script = project_root / "sim" / "oracle" / "run_oracle_trace.py"
    run_directed_script = project_root / "sim" / "tests" / "run_directed_fixture.py"

    rows: list[dict[str, Any]] = []

    for target in targets:
        print(f"[P2] target={target} generating trace...")
        snapshot_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = out_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = out_dir / f"sim_trace_{target}.jsonl"

        row: dict[str, Any] = {
            "target": target,
            "status": "fail",
            "scope_used": args.scope,
            "steps_used": 0,
            "final_phase": None,
            "failure_reason": None,
            "first_diff_step": None,
            "first_diff_path": None,
            "oracle_hash": None,
            "sim_hash": None,
            "dumped_oracle": None,
            "dumped_sim": None,
            "artifacts": {
                "oracle_start_snapshot": str(snapshot_path),
                "action_trace": str(action_path),
                "oracle_trace": str(oracle_trace_path),
                "sim_trace": str(sim_trace_path),
            },
        }

        gen = generate_trace(
            base_url=args.base_url,
            target=target,
            max_steps=int(args.max_steps),
            seed=args.seed,
            timeout_sec=float(args.timeout_sec),
            wait_sleep=0.05,
            out_dir=out_dir,
        )
        row["steps_used"] = int(gen.get("steps_used") or 0)
        row["final_phase"] = gen.get("final_phase")

        if not bool(gen.get("success")):
            row["status"] = "gen_fail"
            row["failure_reason"] = str(gen.get("failure_reason") or "generator_failed")
            rows.append(row)
            print(_summarize(row))
            continue

        start_state_path = str(gen.get("oracle_start_state_path") or "").strip()
        if start_state_path:
            try:
                from trainer.env_client import _call_method

                _call_method(args.base_url, "load", {"path": start_state_path}, timeout=float(args.timeout_sec))
            except Exception as exc:
                row["status"] = "oracle_fail"
                row["failure_reason"] = f"failed_to_load_start_state:{exc}"
                rows.append(row)
                print(_summarize(row))
                continue

        oracle_cmd = [
            sys.executable,
            str(run_oracle_script),
            "--base-url",
            args.base_url,
            "--seed",
            args.seed,
            "--action-trace",
            str(action_path),
            "--out",
            str(oracle_trace_path),
            "--snapshot-every",
            "1",
            "--timeout-sec",
            str(args.timeout_sec),
            "--wait-sleep",
            "0.05",
        ]
        code, stdout, stderr = _run_cmd(oracle_cmd, cwd=project_root)
        if code != 0:
            row["status"] = "oracle_fail"
            row["failure_reason"] = (stdout + "\n" + stderr).strip()[:500]
            rows.append(row)
            print(_summarize(row))
            continue

        directed_cmd = [
            sys.executable,
            str(run_directed_script),
            "--oracle-snapshot",
            str(snapshot_path),
            "--action-trace",
            str(action_path),
            "--oracle-trace",
            str(oracle_trace_path),
            "--scope",
            args.scope,
            "--fail-fast",
            "--snapshot-every",
            "1",
            "--out-trace",
            str(sim_trace_path),
        ]
        if args.dump_on_diff:
            target_dump = Path(args.dump_on_diff) / target
            directed_cmd.extend(["--dump-on-diff", str(target_dump)])

        code, stdout, stderr = _run_cmd(directed_cmd, cwd=project_root)
        parsed = _parse_diff_output((stdout or "") + "\n" + (stderr or ""))
        row.update(parsed)

        if code == 0:
            row["status"] = "pass"
        else:
            row["status"] = "diff_fail"
            row["failure_reason"] = (stdout + "\n" + stderr).strip()[:500]

        rows.append(row)
        print(_summarize(row))

    passed = sum(1 for r in rows if r.get("status") == "pass")
    diff_fail = sum(1 for r in rows if r.get("status") == "diff_fail")
    oracle_fail = sum(1 for r in rows if r.get("status") == "oracle_fail")
    gen_fail = sum(1 for r in rows if r.get("status") == "gen_fail")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "scope": args.scope,
        "total": len(rows),
        "passed": passed,
        "diff_fail": diff_fail,
        "oracle_fail": oracle_fail,
        "gen_fail": gen_fail,
        "results": rows,
    }

    report_path = out_dir / "report_p2.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[P2] done total={len(rows)} pass={passed} diff_fail={diff_fail} oracle_fail={oracle_fail} gen_fail={gen_fail}")
    print(f"[P2] report={report_path}")
    return 0 if (diff_fail + oracle_fail + gen_fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
