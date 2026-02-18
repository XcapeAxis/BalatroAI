if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.generate_p0_trace import TARGETS, generate_trace
from trainer.env_client import health

SCOPE_CHOICES = ["hand_core", "score_core", "p0_hand_score_core", "zones_core", "zones_counts_core", "economy_core", "rng_events_core", "full"]
TARGET_SCOPE_FALLBACK = {
    "p0_07_discard_resource": "zones_counts_core",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-build oracle P0 fixtures and run directed diff.",
        epilog=(
            "Example:\n"
            "python sim\\oracle\\batch_build_p0_oracle_fixtures.py "
            "--base-url http://127.0.0.1:12346 "
            "--out-dir sim\\tests\\fixtures_runtime\\oracle_p0 "
            "--max-steps 80 --scope score_core --seed AAAAAAA --resume "
            "--dump-on-diff sim\\tests\\fixtures_runtime\\oracle_p0\\dumps"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p0/")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="score_core")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--targets", default=None, help="Comma-separated target subset. Default is all TARGETS.")
    parser.add_argument("--resume", action="store_true", help="Skip targets if all expected artifacts already exist.")
    parser.add_argument("--dump-on-diff", default=None, help="Directory for minimal diff dump artifacts.")
    parser.add_argument(
        "--dump-scope-only",
        action="store_true",
        help="Pass through to directed diff to dump scope projection only.",
    )
    parser.add_argument(
        "--dump-per-target",
        dest="dump_per_target",
        action="store_true",
        default=True,
        help="When dump is enabled, use a per-target subdirectory (default: True).",
    )
    parser.add_argument(
        "--no-dump-per-target",
        dest="dump_per_target",
        action="store_false",
        help="When dump is enabled, write all dumps into one directory.",
    )
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
            continue
        if line.startswith("dumped_sim=") and result["dumped_sim"] is None:
            result["dumped_sim"] = line.split("=", 1)[1].strip() or None
            continue
    return result


def _parse_targets(targets_csv: str | None) -> list[str]:
    if not targets_csv:
        return list(TARGETS)
    values = [x.strip() for x in str(targets_csv).split(",") if x.strip()]
    uniq: list[str] = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    invalid = [v for v in uniq if v not in TARGETS]
    if invalid:
        raise ValueError(f"invalid targets: {invalid}; supported={TARGETS}")
    return uniq


def _summarize_failure(text: str | None, max_len: int = 400) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", " ", raw)
    return compact[:max_len]


def _extract_step_id(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"step=(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as fp:
        for line in fp:
            if line.strip():
                count += 1
    return count


def _load_target_preferred_scope(project_root: Path, target: str, global_scope: str) -> str:
    meta_path = project_root / "sim" / "tests" / "fixtures_directed" / f"meta_{target}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            preferred = str((meta or {}).get("preferred_scope") or "").strip()
            if preferred in SCOPE_CHOICES:
                return preferred
        except Exception:
            pass

    fallback = TARGET_SCOPE_FALLBACK.get(target)
    if fallback in SCOPE_CHOICES:
        return fallback
    return global_scope


def _print_row_summary(row: dict[str, Any]) -> None:
    reason = str(row.get("failure_reason") or "")[:80]
    msg = (
        f"[P0] {row.get('target')} | {row.get('status')} | "
        f"scope={row.get('scope_used')} | steps={row.get('steps_used') or 0} | {reason}"
    )
    if row.get("status") == "diff_fail" and row.get("dump_dir"):
        msg += f" | dump_dir={str(row.get('dump_dir'))[:80]}"
    print(msg)


def main() -> int:
    args = parse_args()
    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    try:
        selected_targets = _parse_targets(args.targets)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_oracle_script = project_root / "sim" / "oracle" / "run_oracle_trace.py"
    run_directed_script = project_root / "sim" / "tests" / "run_directed_fixture.py"

    rows: list[dict[str, Any]] = []

    for target in selected_targets:
        print(f"[P0] target={target} generating trace...")

        snapshot_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = out_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = out_dir / f"sim_trace_{target}.jsonl"
        scope_used = _load_target_preferred_scope(project_root, target, args.scope)

        row: dict[str, Any] = {
            "target": target,
            "status": "fail",
            "scope_used": scope_used,
            "steps_used": 0,
            "final_phase": None,
            "index_base": None,
            "failure_reason": None,
            "first_diff_step": None,
            "first_diff_path": None,
            "oracle_hash": None,
            "sim_hash": None,
            "dump_dir": None,
            "dumped_oracle": None,
            "dumped_sim": None,
            "artifacts": {
                "oracle_start_snapshot": _safe_relpath(snapshot_path, project_root),
                "action_trace": _safe_relpath(action_path, project_root),
                "oracle_trace": _safe_relpath(oracle_trace_path, project_root),
                "sim_trace": _safe_relpath(sim_trace_path, project_root),
            },
            "error": None,
        }

        all_artifacts_exist = snapshot_path.exists() and action_path.exists() and oracle_trace_path.exists() and sim_trace_path.exists()
        if args.resume and all_artifacts_exist:
            row["status"] = "skipped"
            row["steps_used"] = _count_jsonl_lines(action_path)
            row["failure_reason"] = "resume: artifacts already exist"
            row["error"] = row["failure_reason"]
            rows.append(row)
            _print_row_summary(row)
            continue

        try:
            gen = generate_trace(
                base_url=args.base_url,
                target=target,
                max_steps=int(args.max_steps),
                seed=args.seed,
                timeout_sec=float(args.timeout_sec),
                wait_sleep=0.05,
                out_dir=out_dir,
            )
        except Exception as exc:
            row["status"] = "gen_fail"
            row["failure_reason"] = _summarize_failure(f"generate_exception:{type(exc).__name__}:{exc}")
            row["error"] = row["failure_reason"]
            rows.append(row)
            _print_row_summary(row)
            continue

        row["steps_used"] = int(gen.get("steps_used") or 0)
        row["final_phase"] = gen.get("final_phase")
        row["index_base"] = int(gen.get("index_base") or 0)

        if not bool(gen.get("success")):
            row["status"] = "gen_fail"
            row["failure_reason"] = _summarize_failure(str(gen.get("failure_reason") or "generator_failed"))
            row["error"] = row["failure_reason"]
            rows.append(row)
            _print_row_summary(row)
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
            merged = (stdout + "\n" + stderr).strip()
            step_id = _extract_step_id(merged)
            summary = _summarize_failure(merged)
            if step_id is not None:
                row["failure_reason"] = f"oracle_fail step={step_id}: {summary}"
            else:
                row["failure_reason"] = f"oracle_fail: {summary}" if summary else "oracle_fail"
            row["error"] = row["failure_reason"]
            rows.append(row)
            _print_row_summary(row)
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
            scope_used,
            "--fail-fast",
            "--snapshot-every",
            "1",
            "--out-trace",
            str(sim_trace_path),
        ]

        target_dump_dir: Path | None = None
        if args.dump_on_diff:
            base_dump_dir = (project_root / args.dump_on_diff).resolve() if not Path(args.dump_on_diff).is_absolute() else Path(args.dump_on_diff)
            target_dump_dir = (base_dump_dir / target) if args.dump_per_target else base_dump_dir
            directed_cmd.extend(["--dump-on-diff", str(target_dump_dir)])
            if args.dump_scope_only:
                directed_cmd.append("--dump-scope-only")

        code, stdout, stderr = _run_cmd(directed_cmd, cwd=project_root)
        parsed = _parse_diff_output((stdout or "") + "\n" + (stderr or ""))
        row.update(parsed)

        if row.get("dumped_oracle"):
            row["dumped_oracle"] = _safe_relpath(Path(str(row["dumped_oracle"])), project_root)
        if row.get("dumped_sim"):
            row["dumped_sim"] = _safe_relpath(Path(str(row["dumped_sim"])), project_root)
        if target_dump_dir is not None and (row.get("dumped_oracle") or row.get("dumped_sim")):
            row["dump_dir"] = _safe_relpath(target_dump_dir, project_root)

        if code == 0:
            row["status"] = "pass"
            row["failure_reason"] = None
            row["error"] = None
            _print_row_summary(row)
        else:
            row["status"] = "diff_fail"
            merged = (stdout + "\n" + stderr).strip()
            if row.get("first_diff_step") is not None:
                row["failure_reason"] = (
                    f"diff_fail step={row.get('first_diff_step')} path={row.get('first_diff_path')} "
                    f"oracle_hash={row.get('oracle_hash')} sim_hash={row.get('sim_hash')}"
                )
            else:
                summary = _summarize_failure(merged)
                row["failure_reason"] = f"diff_fail: {summary}" if summary else "diff_fail"
            row["error"] = row["failure_reason"]
            _print_row_summary(row)

        rows.append(row)

    passed = sum(1 for r in rows if r.get("status") == "pass")
    skipped = sum(1 for r in rows if r.get("status") == "skipped")
    failed = len(rows) - passed - skipped

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "scope": args.scope,
        "targets_requested": selected_targets,
        "resume": bool(args.resume),
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": rows,
    }

    report_path = out_dir / "report_p0.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[P0] done total={len(rows)} passed={passed} failed={failed} skipped={skipped}")
    print(f"[P0] report={report_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
