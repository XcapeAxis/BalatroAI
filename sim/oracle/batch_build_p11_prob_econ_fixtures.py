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

from sim.oracle.generate_p11_prob_econ_trace import generate_one_trace, load_supported_entries
from sim.oracle.p11_prob_econ_joker_classifier import build_and_write
from trainer.env_client import _call_method, health

SCOPE_CHOICES = [
    "p11_prob_econ_observed_core",
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
    parser = argparse.ArgumentParser(description="Batch-build P11 prob/econ fixtures and run directed oracle/sim diff.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--out-dir", default="sim/tests/fixtures_runtime/oracle_p11_prob_econ_v1")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="p11_prob_econ_observed_core")
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
        f"[P11] {row.get('target')} | {row.get('status')} | "
        f"template={row.get('template')} | steps={row.get('steps_used') or 0} | {reason}"
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
    doc_path = project_root / "docs" / "COVERAGE_P11_STATUS.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    rows = report.get("results") if isinstance(report.get("results"), list) else []
    template_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        template_counter[str(row.get("template") or "unknown")] += 1
        status_counter[str(row.get("status") or "unknown")] += 1
        category_counter[str(row.get("category") or "unknown")] += 1

    lines: list[str] = []
    lines.append("# P11 Prob/Econ Fixture Coverage Status")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Base URL: `{report.get('base_url')}`")
    lines.append(f"- Scope: `{report.get('scope')}`")
    lines.append(f"- Classifier supported: **{classifier_summary.get('supported', 0)}**")
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
    lines.append("## Category Counts")
    for k, v in sorted(category_counter.items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Template Counts")
    for k, v in sorted(template_counter.items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Status Counts")
    for k, v in sorted(status_counter.items()):
        lines.append(f"- `{k}`: {v}")
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc_path


def main() -> int:
    args = parse_args()

    if int(args.workers) != 1:
        print(f"[P11] workers={args.workers} requested; forcing workers=1 for deterministic oracle alignment.")

    project_root = Path(__file__).resolve().parent.parent.parent

    # Ensure p11 pick payload exists and classifier outputs are up to date.
    picker_cmd = [
        "python",
        "-B",
        "sim/oracle/p11_pick_prob_econ_targets.py",
        "--out-derived",
        "balatro_mechanics/derived",
        "--out-docs",
        "docs",
    ]
    picker_code, picker_out, picker_err = _run_cmd(picker_cmd, project_root)
    if picker_code != 0:
        print("ERROR: failed to refresh p11 picker outputs")
        if picker_out:
            print(picker_out)
        if picker_err:
            print(picker_err)
        return 2
    classifier_summary = build_and_write(project_root)

    if args.targets and args.targets_file:
        print("ERROR: use either --targets or --targets-file, not both")
        return 2

    try:
        targets_csv = args.targets
        if args.targets_file:
            targets_csv = _read_targets_file(args.targets_file, project_root)
        entries = _resolve_entries(project_root, targets_csv, args.limit)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not health(args.base_url):
        print(f"base_url unhealthy: {args.base_url}. Start balatrobot serve first.")
        return 2

    report_rows: list[dict[str, Any]] = []
    for entry in entries:
        target = str(entry.get("target") or "")
        template = str(entry.get("template") or "")
        category = str((entry.get("params") or {}).get("category") or "")

        snapshot_path = out_dir / f"oracle_start_snapshot_{target}.json"
        action_path = out_dir / f"action_trace_{target}.jsonl"
        oracle_trace_path = out_dir / f"oracle_trace_{target}.jsonl"
        sim_trace_path = out_dir / f"sim_trace_{target}.jsonl"

        dump_dir_rel: str | None = None
        dump_target_dir: Path | None = None
        if args.dump_on_diff:
            base_dump = Path(args.dump_on_diff)
            if not base_dump.is_absolute():
                base_dump = project_root / base_dump
            dump_target_dir = base_dump / target if args.dump_per_target else base_dump
            dump_dir_rel = _safe_relpath(dump_target_dir, project_root)

        row = {
            "target": target,
            "template": template,
            "category": category,
            "status": "fail",
            "failure_reason": None,
            "steps_used": 0,
            "final_phase": None,
            "index_base": None,
            "first_diff_step": None,
            "first_diff_path": None,
            "oracle_hash": None,
            "sim_hash": None,
            "dump_dir": dump_dir_rel,
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

        if args.resume and snapshot_path.exists() and action_path.exists() and oracle_trace_path.exists() and sim_trace_path.exists():
            row["status"] = "skipped"
            row["failure_reason"] = "resume: artifacts already exist"
            row["steps_used"] = _count_jsonl_lines(action_path)
            report_rows.append(row)
            _print_row_summary(row)
            continue

        gen_result: dict[str, Any] | None = None
        max_attempts = max(1, int(args.max_retries) + 1)
        for attempt in range(1, max_attempts + 1):
            print(f"[P11] target={target} attempt={attempt}/{max_attempts} generating trace...")
            gen_result = generate_one_trace(
                base_url=args.base_url,
                entry=entry,
                max_steps=int(args.max_steps),
                seed=args.seed,
                timeout_sec=float(args.timeout_sec),
                wait_sleep=0.05,
                out_dir=out_dir,
            )
            row["steps_used"] = int(gen_result.get("steps_used") or 0)
            row["final_phase"] = gen_result.get("final_phase")
            row["index_base"] = gen_result.get("index_base")

            if bool(gen_result.get("success")):
                break

            reason = _summarize_failure(gen_result.get("failure_reason"))
            if _is_transport_failure(reason):
                ok, note = _restart_service(project_root, args.base_url)
                if not ok:
                    row["status"] = "gen_fail"
                    row["failure_reason"] = f"gen_fail_restart:{note}"
                    row["error"] = row["failure_reason"]
                    break
                continue

            row["status"] = "gen_fail"
            row["failure_reason"] = reason or "gen_fail"
            row["error"] = row["failure_reason"]
            break

        if row["status"] == "gen_fail":
            report_rows.append(row)
            _print_row_summary(row)
            continue

        if not gen_result or not bool(gen_result.get("success")):
            row["status"] = "gen_fail"
            row["failure_reason"] = _summarize_failure((gen_result or {}).get("failure_reason"))
            row["error"] = row["failure_reason"]
            report_rows.append(row)
            _print_row_summary(row)
            continue

        if not snapshot_path.exists() or not action_path.exists():
            row["status"] = "gen_fail"
            row["failure_reason"] = "generator did not produce required artifacts"
            row["error"] = row["failure_reason"]
            report_rows.append(row)
            _print_row_summary(row)
            continue

        start_state_path = Path(str(gen_result.get("oracle_start_state_path") or "")).resolve() if gen_result else None
        if start_state_path and start_state_path.exists():
            try:
                _call_method(args.base_url, "load", {"path": str(start_state_path)}, timeout=float(args.timeout_sec))
            except Exception as exc:
                row["status"] = "oracle_fail"
                row["failure_reason"] = f"oracle_fail: failed to load oracle start state {start_state_path}: {exc}"
                row["error"] = row["failure_reason"]
                report_rows.append(row)
                _print_row_summary(row)
                continue

        oracle_cmd = [
            "python",
            "-B",
            "sim/oracle/run_oracle_trace.py",
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
        ]
        code, out_text, err_text = _run_cmd(oracle_cmd, project_root)
        if code != 0:
            combined = _summarize_failure((out_text + "\n" + err_text).strip())
            step_id = _extract_step_id(combined)
            row["status"] = "oracle_fail"
            row["failure_reason"] = f"oracle_fail step={step_id}: {combined}" if step_id is not None else f"oracle_fail: {combined}"
            row["error"] = row["failure_reason"]
            report_rows.append(row)
            _print_row_summary(row)
            continue

        directed_cmd = [
            "python",
            "-B",
            "sim/tests/run_directed_fixture.py",
            "--oracle-snapshot",
            str(snapshot_path),
            "--action-trace",
            str(action_path),
            "--oracle-trace",
            str(oracle_trace_path),
            "--scope",
            args.scope,
            "--out-trace",
            str(sim_trace_path),
            "--snapshot-every",
            "1",
        ]
        if args.dump_on_diff and dump_target_dir is not None:
            directed_cmd += ["--dump-on-diff", str(dump_target_dir)]
            if args.dump_scope_only:
                directed_cmd += ["--dump-scope-only"]

        code, out_text, err_text = _run_cmd(directed_cmd, project_root)
        combined = (out_text + "\n" + err_text).strip()
        parsed_diff = _parse_diff_output(combined)

        row["first_diff_step"] = parsed_diff.get("first_diff_step")
        row["first_diff_path"] = parsed_diff.get("first_diff_path")
        row["oracle_hash"] = parsed_diff.get("oracle_hash")
        row["sim_hash"] = parsed_diff.get("sim_hash")
        row["dumped_oracle"] = parsed_diff.get("dumped_oracle")
        row["dumped_sim"] = parsed_diff.get("dumped_sim")

        if code != 0:
            if parsed_diff.get("first_diff_step") is not None or "MISMATCH" in combined:
                row["status"] = "diff_fail"
                if parsed_diff.get("first_diff_step") is not None:
                    row["failure_reason"] = (
                        f"diff_fail step={parsed_diff.get('first_diff_step')} "
                        f"path={parsed_diff.get('first_diff_path')} "
                        f"oracle_hash={parsed_diff.get('oracle_hash')} sim_hash={parsed_diff.get('sim_hash')}"
                    )
                else:
                    row["failure_reason"] = f"diff_fail: {_summarize_failure(combined)}"
            else:
                row["status"] = "oracle_fail"
                row["failure_reason"] = f"directed_runner_fail: {_summarize_failure(combined)}"
            row["error"] = row["failure_reason"]
        else:
            row["status"] = "pass"
            row["failure_reason"] = None
            row["error"] = None

        report_rows.append(row)
        _print_row_summary(row)

    counters = Counter(str(r.get("status") or "unknown") for r in report_rows)
    classifier_jsonsafe = json.loads(json.dumps(classifier_summary, ensure_ascii=False, default=str))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "scope": args.scope,
        "targets_requested": [str(e.get("target") or "") for e in entries],
        "resume": bool(args.resume),
        "total": len(report_rows),
        "passed": counters.get("pass", 0),
        "failed": counters.get("gen_fail", 0) + counters.get("oracle_fail", 0) + counters.get("diff_fail", 0),
        "skipped": counters.get("skipped", 0),
        "gen_fail": counters.get("gen_fail", 0),
        "oracle_fail": counters.get("oracle_fail", 0),
        "diff_fail": counters.get("diff_fail", 0),
        "classifier": classifier_jsonsafe,
        "results": report_rows,
    }

    report_path = out_dir / "report_p11.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    status_doc = _write_status_doc(project_root, report, classifier_summary)

    print(
        "[P11] done total={total} pass={passed} diff_fail={diff_fail} oracle_fail={oracle_fail} gen_fail={gen_fail} skipped={skipped}".format(
            **report
        )
    )
    print(f"[P11] report={report_path}")
    print(f"[P11] status_doc={status_doc}")
    return 0 if (report["diff_fail"] + report["oracle_fail"] + report["gen_fail"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
