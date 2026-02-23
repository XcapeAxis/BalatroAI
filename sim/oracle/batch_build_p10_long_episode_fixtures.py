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

from sim.oracle.generate_p10_long_episode_trace import (
    generate_one_trace,
    load_supported_entries,
    select_entries,
)
from trainer.env_client import _call_method, health

SCOPE_CHOICES = [
    "p10_long_episode_observed_core",
    "p9_episode_observed_core",
    "p8_rng_observed_core",
    "p8_shop_observed_core",
    "p7_stateful_observed_core",
    "p5_voucher_pack_observed_core",
    "p5_modifier_observed_core",
    "p4_consumable_observed_core",
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
    parser = argparse.ArgumentParser(description="Batch-build P10 long-episode fixtures and run directed oracle/sim diff.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p10_long_v1")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="p10_long_episode_observed_core")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--targets", default=None, help="Comma-separated target subset")
    parser.add_argument("--targets-file", default=None, help="Optional text file with one target per line")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
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


def _summarize_failure(text: str | None, max_len: int = 600) -> str:
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


def _write_generated_trace_artifacts(
    *,
    snapshot_path: Path,
    action_path: Path,
    start_snapshot: dict[str, Any] | None,
    action_trace: list[dict[str, Any]] | None,
) -> None:
    if start_snapshot is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(start_snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    action_path.parent.mkdir(parents=True, exist_ok=True)
    with action_path.open("w", encoding="utf-8", newline="\n") as fp:
        for action in action_trace or []:
            fp.write(json.dumps(action, ensure_ascii=False) + "\n")


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


def _clear_lovely_dump() -> None:
    dump_dir = Path(r"C:\Users\Administrator\AppData\Roaming\Balatro\Mods\lovely\dump")
    if not dump_dir.exists():
        return
    try:
        for child in dump_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                import shutil

                shutil.rmtree(child, ignore_errors=True)
    except Exception:
        pass


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

    for _ in range(70):
        if health(base_url):
            return True, "ok"
        time.sleep(1.0)
    return False, "start_timeout"


def _restart_service(project_root: Path, base_url: str) -> tuple[bool, str]:
    _kill_service_processes()
    time.sleep(1.0)
    _clear_lovely_dump()
    return _start_service(project_root, base_url)


def _print_row_summary(row: dict[str, Any]) -> None:
    reason = str(row.get("failure_reason") or "")[:120]
    msg = (
        f"[P10-episode] {row.get('target')} | {row.get('status')} | "
        f"steps={row.get('steps_used') or 0} | {reason}"
    )
    if row.get("status") == "diff_fail" and row.get("dump_dir"):
        msg += f" | dump_dir={str(row.get('dump_dir'))[:120]}"
    print(msg)


def _read_targets_file(path_value: str | None, project_root: Path) -> str | None:
    if not path_value:
        return None
    p = Path(path_value)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    if not p.exists():
        raise ValueError(f"targets-file not found: {p}")
    targets: list[str] = []
    seen: set[str] = set()
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            targets.append(line)
    return ",".join(targets) if targets else None


def _write_status_docs(project_root: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    status_path = project_root / "docs" / "COVERAGE_P10_STATUS.md"
    episodes_path = project_root / "docs" / "COVERAGE_P10_EPISODES.md"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    rows = report.get("results") if isinstance(report.get("results"), list) else []
    fail_counter: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") != "pass":
            fail_counter[str(row.get("status") or "unknown")] += 1

    status_lines: list[str] = []
    status_lines.append("# P10 Long Episode Coverage Status")
    status_lines.append("")
    status_lines.append(f"- generated_at: `{report.get('generated_at')}`")
    status_lines.append(f"- base_url: `{report.get('base_url')}`")
    status_lines.append(f"- scope: `{report.get('scope')}`")
    status_lines.append(f"- total: **{report.get('total', 0)}**")
    status_lines.append(f"- pass: **{report.get('passed', 0)}**")
    status_lines.append(f"- diff_fail: **{report.get('diff_fail', 0)}**")
    status_lines.append(f"- oracle_fail: **{report.get('oracle_fail', 0)}**")
    status_lines.append(f"- gen_fail: **{report.get('gen_fail', 0)}**")
    status_lines.append(f"- skipped: **{report.get('skipped', 0)}**")
    status_lines.append(f"- unsupported: **{report.get('unsupported', 0)}**")
    status_lines.append("")
    status_lines.append("## Failure Breakdown")
    if fail_counter:
        for key, value in fail_counter.most_common():
            status_lines.append(f"- `{key}`: {value}")
    else:
        status_lines.append("- none")

    episodes_lines: list[str] = []
    episodes_lines.append("# P10 Episode Fixture Targets")
    episodes_lines.append("")
    for entry in load_supported_entries(project_root):
        episodes_lines.append(
            f"- `{entry.get('target')}` [stake={entry.get('stake') or '-'} category={entry.get('category') or '-'}]"
        )

    status_path.write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    episodes_path.write_text("\n".join(episodes_lines) + "\n", encoding="utf-8")
    return status_path, episodes_path


def main() -> int:
    args = parse_args()

    if int(args.workers) != 1:
        print("[P10-episode] forcing workers=1 for deterministic/stable replay")

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    try:
        targets_csv = _read_targets_file(args.targets_file, project_root)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    merged_targets = args.targets or targets_csv
    entries = select_entries(merged_targets, args.limit, project_root)
    if not entries:
        print("ERROR: no targets selected")
        return 2

    run_oracle_script = project_root / "sim" / "oracle" / "run_oracle_trace.py"
    run_directed_script = project_root / "sim" / "tests" / "run_directed_fixture.py"

    rows: list[dict[str, Any]] = []
    for entry in entries:
        target = str(entry.get("target") or "")
        snapshot_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = out_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = out_dir / f"sim_trace_{target}.jsonl"
        oracle_start_state_path = out_dir / f"oracle_start_state_{target}.jkr"

        row: dict[str, Any] = {
            "target": target,
            "status": "fail",
            "category": entry.get("category"),
            "template": "p10_long_episode_v1",
            "stake": entry.get("stake"),
            "steps_used": 0,
            "final_phase": None,
            "index_base": None,
            "first_diff_step": None,
            "first_diff_path": None,
            "oracle_hash": None,
            "sim_hash": None,
            "failure_reason": None,
            "error": None,
            "dump_dir": None,
            "dumped_oracle": None,
            "dumped_sim": None,
            "artifacts": {
                "oracle_start_snapshot": _safe_relpath(snapshot_path, project_root),
                "oracle_start_state": _safe_relpath(oracle_start_state_path, project_root),
                "action_trace": _safe_relpath(action_path, project_root),
                "oracle_trace": _safe_relpath(oracle_trace_path, project_root),
                "sim_trace": _safe_relpath(sim_trace_path, project_root),
            },
        }

        if args.resume and snapshot_path.exists() and action_path.exists() and oracle_trace_path.exists() and sim_trace_path.exists():
            row["status"] = "skipped"
            row["steps_used"] = _count_jsonl_lines(action_path)
            row["failure_reason"] = "resume: artifacts already exist"
            row["error"] = row["failure_reason"]
            rows.append(row)
            _print_row_summary(row)
            continue

        success_terminal = False
        for attempt in range(int(args.max_retries) + 1):
            print(f"[P10-episode] target={target} attempt={attempt + 1}/{int(args.max_retries) + 1} generating trace...")
            try:
                gen = generate_one_trace(
                    base_url=args.base_url,
                    entry=entry,
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

            _write_generated_trace_artifacts(
                snapshot_path=snapshot_path,
                action_path=action_path,
                start_snapshot=gen.get("start_snapshot") if isinstance(gen, dict) else None,
                action_trace=gen.get("action_trace") if isinstance(gen, dict) else None,
            )

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
                "--check-start",
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

            d_code, d_stdout, d_stderr = _run_cmd(directed_cmd, cwd=project_root)
            if d_code != 0:
                merged = (d_stdout + "\n" + d_stderr).strip()
                diff_info = _parse_diff_output(merged)
                if diff_info.get("first_diff_step") is not None:
                    row["status"] = "diff_fail"
                    row["first_diff_step"] = diff_info.get("first_diff_step")
                    row["first_diff_path"] = diff_info.get("first_diff_path")
                    row["oracle_hash"] = diff_info.get("oracle_hash")
                    row["sim_hash"] = diff_info.get("sim_hash")
                    row["dumped_oracle"] = diff_info.get("dumped_oracle")
                    row["dumped_sim"] = diff_info.get("dumped_sim")
                    row["failure_reason"] = _summarize_failure(
                        f"diff_fail step={row['first_diff_step']} path={row['first_diff_path']} "
                        f"oracle_hash={row['oracle_hash']} sim_hash={row['sim_hash']}"
                    )
                    row["error"] = row["failure_reason"]
                else:
                    row["status"] = "diff_fail"
                    row["failure_reason"] = _summarize_failure(f"diff_fail: {merged}")
                    row["error"] = row["failure_reason"]

                if target_dump_dir is not None:
                    row["dump_dir"] = _safe_relpath(target_dump_dir, project_root)
                if diff_info.get("dumped_oracle"):
                    row["dumped_oracle"] = diff_info.get("dumped_oracle")
                if diff_info.get("dumped_sim"):
                    row["dumped_sim"] = diff_info.get("dumped_sim")
                success_terminal = True
                break

            row["status"] = "pass"
            row["failure_reason"] = None
            row["error"] = None
            if target_dump_dir is not None:
                row["dump_dir"] = _safe_relpath(target_dump_dir, project_root)
            success_terminal = True
            break

        if not success_terminal and row["status"] == "fail":
            row["status"] = "gen_fail"
            row["failure_reason"] = "exhausted_retries"
            row["error"] = row["failure_reason"]

        rows.append(row)
        _print_row_summary(row)

    counter = Counter(str(r.get("status") or "") for r in rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "seed": args.seed,
        "scope": args.scope,
        "targets_requested": [str(e.get("target") or "") for e in entries],
        "resume": bool(args.resume),
        "total": len(rows),
        "passed": int(counter.get("pass", 0)),
        "diff_fail": int(counter.get("diff_fail", 0)),
        "oracle_fail": int(counter.get("oracle_fail", 0)),
        "gen_fail": int(counter.get("gen_fail", 0)),
        "skipped": int(counter.get("skipped", 0)),
        "unsupported": 0,
        "results": rows,
    }

    report_path = out_dir / "report_p10.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    status_doc, episodes_doc = _write_status_docs(project_root, report)

    print(
        f"P10 long-episode summary: pass={report['passed']}/{report['total']} "
        f"diff_fail={report['diff_fail']} oracle_fail={report['oracle_fail']} "
        f"gen_fail={report['gen_fail']} skipped={report['skipped']}"
    )
    print(f"report: {report_path}")
    print(f"status: {status_doc}")
    print(f"coverage: {episodes_doc}")

    return 0 if (report["diff_fail"] == 0 and report["oracle_fail"] == 0 and report["gen_fail"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

