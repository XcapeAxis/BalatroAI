from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sim.oracle.generate_p3_joker_trace import generate_one_trace, load_supported_entries
from sim.oracle.p3_joker_classifier import build_and_write
from trainer.env_client import _call_method, health

SCOPE_CHOICES = [
    "p3_hand_score_observed_core",
    "p2b_hand_score_observed_core",
    "p2_hand_score_observed_core",
    "p1_hand_score_observed_core",
    "p0_hand_score_observed_core",
    "hand_core",
    "score_core",
    "full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-build P3 joker fixtures and run directed oracle/sim diff.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p3_jokers_v1")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="p3_hand_score_observed_core")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--targets", default=None, help="Comma-separated target subset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--dump-on-diff", default=None)
    parser.add_argument("--dump-scope-only", action="store_true")
    parser.add_argument("--dump-per-target", dest="dump_per_target", action="store_true", default=True)
    parser.add_argument("--no-dump-per-target", dest="dump_per_target", action="store_false")
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


def _summarize_failure(text: str | None, max_len: int = 500) -> str:
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


def _is_transport_failure(text: str | None) -> bool:
    low = str(text or "").lower()
    keywords = [
        "connection refused",
        "actively refused",
        "10061",
        "timed out",
        "timeout",
        "health check failed",
        "failed to establish a new connection",
    ]
    return any(k in low for k in keywords)


def _base_url_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else 80


def _kill_service_processes() -> None:
    for name in ("balatrobot.exe", "uvx.exe", "Balatro.exe"):
        subprocess.run(["taskkill", "/IM", name, "/F"], capture_output=True, text=True)


def _start_service(project_root: Path, base_url: str) -> tuple[bool, str]:
    port = _base_url_port(base_url)
    love = Path(r"D:\SteamLibrary\steamapps\common\Balatro\Balatro.exe")
    lovely = Path(r"D:\SteamLibrary\steamapps\common\Balatro\version.dll")

    cmd = ["uvx", "balatrobot", "serve", "--headless", "--fast", "--port", str(port)]
    if love.exists() and lovely.exists():
        cmd += ["--love-path", str(love), "--lovely-path", str(lovely)]
    elif love.exists():
        cmd += ["--balatro-path", str(love)]

    try:
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except Exception as exc:
        return False, f"start_failed:{exc}"

    for _ in range(60):
        if health(base_url):
            return True, "ok"
        time.sleep(1.0)
    return False, "start_timeout"


def _restart_service(project_root: Path, base_url: str) -> tuple[bool, str]:
    _kill_service_processes()
    time.sleep(1.0)
    return _start_service(project_root, base_url)


def _print_row_summary(row: dict[str, Any]) -> None:
    reason = str(row.get("failure_reason") or "")[:100]
    msg = (
        f"[P3] {row.get('target')} | {row.get('status')} | "
        f"template={row.get('template')} | steps={row.get('steps_used') or 0} | {reason}"
    )
    if row.get("status") == "diff_fail" and row.get("dump_dir"):
        msg += f" | dump_dir={str(row.get('dump_dir'))[:100]}"
    print(msg)


def _resolve_entries(project_root: Path, targets_csv: str | None, limit: int | None) -> list[dict[str, Any]]:
    all_entries = load_supported_entries(project_root)
    by_target = {str(e.get("target") or ""): e for e in all_entries}

    if targets_csv:
        ordered: list[str] = []
        for token in str(targets_csv).split(","):
            t = token.strip()
            if not t:
                continue
            if t not in by_target:
                raise ValueError(f"invalid target: {t}")
            if t not in ordered:
                ordered.append(t)
        entries = [by_target[t] for t in ordered]
    else:
        entries = [by_target[t] for t in sorted(by_target.keys())]

    if limit is not None and int(limit) > 0:
        entries = entries[: int(limit)]

    return entries


def _write_status_doc(project_root: Path, report: dict[str, Any], classifier_summary: dict[str, Any]) -> Path:
    doc_path = project_root / "docs" / "COVERAGE_P3_STATUS.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    rows = report.get("results") if isinstance(report.get("results"), list) else []
    template_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    fail_template_counter: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        tpl = str(row.get("template") or "unknown")
        status = str(row.get("status") or "unknown")
        template_counter[tpl] += 1
        status_counter[status] += 1
        if status != "pass":
            fail_template_counter[tpl] += 1

    lines: list[str] = []
    lines.append("# P3 Joker Fixture Coverage Status")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Base URL: `{report.get('base_url')}`")
    lines.append(f"- Scope: `{report.get('scope')}`")
    lines.append(f"- Classifier total jokers: **{classifier_summary.get('total', 0)}**")
    lines.append(f"- Classifier supported templates: **{classifier_summary.get('supported', 0)}**")
    lines.append(f"- Classifier unsupported: **{classifier_summary.get('unsupported', 0)}**")
    lines.append("")
    lines.append("## Batch Result")
    lines.append(f"- total: **{report.get('total', 0)}**")
    lines.append(f"- pass: **{report.get('passed', 0)}**")
    lines.append(f"- diff_fail: **{report.get('diff_fail', 0)}**")
    lines.append(f"- oracle_fail: **{report.get('oracle_fail', 0)}**")
    lines.append(f"- gen_fail: **{report.get('gen_fail', 0)}**")
    lines.append(f"- skipped: **{report.get('skipped', 0)}**")
    lines.append("")
    lines.append("## Template Counts In This Batch")
    for tpl, count in sorted(template_counter.items()):
        lines.append(f"- `{tpl}`: {count}")
    lines.append("")
    lines.append("## Top Failing Templates")
    if fail_template_counter:
        for tpl, count in fail_template_counter.most_common(12):
            lines.append(f"- `{tpl}`: {count}")
    else:
        lines.append("- none")

    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc_path


def main() -> int:
    args = parse_args()

    if int(args.workers) != 1:
        print(f"[P3] workers={args.workers} requested; forcing workers=1 for deterministic oracle alignment.")

    project_root = Path(__file__).resolve().parent.parent.parent

    classifier_summary = build_and_write(project_root)
    try:
        entries = _resolve_entries(
            project_root=project_root,
            targets_csv=args.targets,
            limit=(int(args.limit) if int(args.limit) > 0 else None),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not entries:
        print("ERROR: no P3 targets selected")
        return 2

    out_dir = (project_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not health(args.base_url):
        ok, note = _restart_service(project_root, args.base_url)
        if not ok:
            print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first. ({note})")
            return 2

    run_oracle_script = project_root / "sim" / "oracle" / "run_oracle_trace.py"
    run_directed_script = project_root / "sim" / "tests" / "run_directed_fixture.py"

    rows: list[dict[str, Any]] = []

    for entry in entries:
        target = str(entry.get("target") or "")
        template = str(entry.get("template") or "")
        joker_key = str(entry.get("joker_key") or "")

        snapshot_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = out_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = out_dir / f"sim_trace_{target}.jsonl"
        oracle_start_state_path = out_dir / f"oracle_start_state_{target}.jkr"

        row: dict[str, Any] = {
            "target": target,
            "template": template,
            "joker_key": joker_key,
            "status": "fail",
            "scope_used": args.scope,
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
            "retries_used": 0,
            "artifacts": {
                "oracle_start_snapshot": _safe_relpath(snapshot_path, project_root),
                "oracle_start_state": _safe_relpath(oracle_start_state_path, project_root),
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

        success_terminal = False
        for attempt in range(int(args.max_retries) + 1):
            row["retries_used"] = attempt

            if not health(args.base_url):
                ok, note = _restart_service(project_root, args.base_url)
                if not ok:
                    row["status"] = "gen_fail"
                    row["failure_reason"] = f"service_restart_failed:{note}"
                    row["error"] = row["failure_reason"]
                    continue

            print(f"[P3] target={target} attempt={attempt + 1}/{int(args.max_retries) + 1} generating trace...")

            try:
                gen = generate_one_trace(
                    base_url=args.base_url,
                    entry=entry,
                    max_steps=int(args.max_steps),
                    seed=args.seed,
                    timeout_sec=float(args.timeout_sec),
                    wait_sleep=0.05,
                    max_attempts=int(args.max_attempts),
                    out_dir=out_dir,
                )
            except Exception as exc:
                row["status"] = "gen_fail"
                row["failure_reason"] = _summarize_failure(f"generate_exception:{type(exc).__name__}:{exc}")
                row["error"] = row["failure_reason"]
                if attempt < int(args.max_retries) and _is_transport_failure(row["failure_reason"]):
                    _restart_service(project_root, args.base_url)
                    continue
                success_terminal = True
                break

            row["steps_used"] = int(gen.get("steps_used") or 0)
            row["final_phase"] = gen.get("final_phase")
            row["index_base"] = int(gen.get("index_base") or 0)

            if not bool(gen.get("success")):
                row["status"] = "gen_fail"
                row["failure_reason"] = _summarize_failure(str(gen.get("failure_reason") or "generator_failed"))
                row["error"] = row["failure_reason"]
                if attempt < int(args.max_retries) and _is_transport_failure(row["failure_reason"]):
                    _restart_service(project_root, args.base_url)
                    continue
                success_terminal = True
                break

            start_state_path = str(gen.get("oracle_start_state_path") or "").strip()
            if start_state_path:
                try:
                    _call_method(args.base_url, "load", {"path": start_state_path}, timeout=float(args.timeout_sec))
                except Exception as exc:
                    row["status"] = "oracle_fail"
                    row["failure_reason"] = _summarize_failure(f"failed_to_load_start_state:{exc}")
                    row["error"] = row["failure_reason"]
                    if attempt < int(args.max_retries) and _is_transport_failure(row["failure_reason"]):
                        _restart_service(project_root, args.base_url)
                        continue
                    success_terminal = True
                    break

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
                merged = (stdout + "\n" + stderr).strip()
                row["status"] = "oracle_fail"
                step_id = _extract_step_id(merged)
                summary = _summarize_failure(merged)
                if step_id is not None:
                    row["failure_reason"] = f"oracle_fail step={step_id}: {summary}"
                else:
                    row["failure_reason"] = f"oracle_fail: {summary}" if summary else "oracle_fail"
                row["error"] = row["failure_reason"]
                if attempt < int(args.max_retries) and _is_transport_failure(merged):
                    _restart_service(project_root, args.base_url)
                    continue
                success_terminal = True
                break

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

            target_dump_dir: Path | None = None
            if args.dump_on_diff:
                base_dump_dir = (project_root / args.dump_on_diff).resolve() if not Path(args.dump_on_diff).is_absolute() else Path(args.dump_on_diff)
                target_dump_dir = (base_dump_dir / target) if args.dump_per_target else base_dump_dir
                directed_cmd.extend(["--dump-on-diff", str(target_dump_dir)])
                if args.dump_scope_only:
                    directed_cmd.append("--dump-scope-only")

            code, stdout, stderr = _run_cmd(directed_cmd, cwd=project_root)
            merged_directed = (stdout or "") + "\n" + (stderr or "")
            parsed = _parse_diff_output(merged_directed)
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
                success_terminal = True
                break

            if attempt < int(args.max_retries) and _is_transport_failure(merged_directed):
                _restart_service(project_root, args.base_url)
                continue

            row["status"] = "diff_fail"
            if row.get("first_diff_step") is not None:
                row["failure_reason"] = (
                    f"diff_fail step={row.get('first_diff_step')} path={row.get('first_diff_path')} "
                    f"oracle_hash={row.get('oracle_hash')} sim_hash={row.get('sim_hash')}"
                )
            else:
                summary = _summarize_failure(merged_directed)
                row["failure_reason"] = f"diff_fail: {summary}" if summary else "diff_fail"
            row["error"] = row["failure_reason"]
            success_terminal = True
            break

        if not success_terminal and row.get("status") == "fail":
            row["status"] = "gen_fail"
            row["failure_reason"] = row.get("failure_reason") or "unknown_failure"
            row["error"] = row["failure_reason"]

        rows.append(row)
        _print_row_summary(row)

    passed = sum(1 for r in rows if r.get("status") == "pass")
    diff_fail = sum(1 for r in rows if r.get("status") == "diff_fail")
    oracle_fail = sum(1 for r in rows if r.get("status") == "oracle_fail")
    gen_fail = sum(1 for r in rows if r.get("status") == "gen_fail")
    skipped = sum(1 for r in rows if r.get("status") == "skipped")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "scope": args.scope,
        "targets_requested": [str(e.get("target") or "") for e in entries],
        "resume": bool(args.resume),
        "max_retries": int(args.max_retries),
        "workers": 1,
        "classifier": {
            "total": int(classifier_summary.get("total") or 0),
            "supported": int(classifier_summary.get("supported") or 0),
            "unsupported": int(classifier_summary.get("unsupported") or 0),
            "map_path": _safe_relpath(Path(str(classifier_summary.get("map_path"))), project_root) if classifier_summary.get("map_path") else None,
            "unsupported_path": _safe_relpath(Path(str(classifier_summary.get("unsupported_path"))), project_root) if classifier_summary.get("unsupported_path") else None,
        },
        "total": len(rows),
        "passed": passed,
        "diff_fail": diff_fail,
        "oracle_fail": oracle_fail,
        "gen_fail": gen_fail,
        "skipped": skipped,
        "results": rows,
    }

    report_path = out_dir / "report_p3.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    status_doc = _write_status_doc(project_root, report, classifier_summary)

    print(
        f"[P3] done total={len(rows)} pass={passed} diff_fail={diff_fail} "
        f"oracle_fail={oracle_fail} gen_fail={gen_fail} skipped={skipped} unsupported={classifier_summary.get('unsupported', 0)}"
    )
    print(f"[P3] report={report_path}")
    print(f"[P3] status_doc={status_doc}")
    return 0 if (diff_fail + oracle_fail + gen_fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
