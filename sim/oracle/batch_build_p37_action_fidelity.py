from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    label: str
    ok: bool
    command: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "output_path": self.output_path,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and replay P37 action-fidelity fixture suites.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", default="P37BATCH01")
    parser.add_argument("--scope", default="p37_action_fidelity_core")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path, label: str, output_path: str = "") -> StepResult:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    stdout_tail = (proc.stdout or "")[-4000:]
    stderr_tail = (proc.stderr or "")[-4000:]
    return StepResult(
        label=label,
        ok=proc.returncode == 0,
        command=cmd,
        returncode=int(proc.returncode),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        output_path=output_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_sim_fixture_suite(repo_root: Path, out_dir: Path, *, py: str, seed: str, scope: str) -> list[dict[str, Any]]:
    scenarios = ["hand_move_play", "joker_move_play", "shop_ops"]
    suite: list[dict[str, Any]] = []
    for scenario in scenarios:
        fixture_dir = out_dir / "fixtures_sim" / scenario
        report_path = out_dir / "reports" / f"sim_{scenario}.json"
        trace_path = out_dir / "reports" / f"sim_trace_{scenario}.jsonl"

        build_cmd = [py, "-B", "sim/tests/build_p37_action_fixture.py", "--out-dir", str(fixture_dir), "--seed", f"{seed}_{scenario}", "--scenario", scenario]
        build_res = _run(build_cmd, cwd=repo_root, label=f"build_sim_{scenario}")

        replay_res = StepResult(label=f"replay_sim_{scenario}", ok=False, command=[], returncode=1, stdout_tail="", stderr_tail="", output_path=str(report_path))
        report_obj: dict[str, Any] = {}
        if build_res.ok:
            replay_cmd = [
                py,
                "-B",
                "sim/tests/run_real_action_replay_fixture.py",
                "--fixture-dir",
                str(fixture_dir),
                "--scope",
                scope,
                "--out",
                str(report_path),
                "--out-trace",
                str(trace_path),
                "--fail-fast",
            ]
            replay_res = _run(replay_cmd, cwd=repo_root, label=f"replay_sim_{scenario}", output_path=str(report_path))
            if report_path.exists():
                report_obj = _read_json(report_path)

        suite.append(
            {
                "suite": "sim_only",
                "scenario": scenario,
                "fixture_dir": str(fixture_dir),
                "report_path": str(report_path),
                "build": build_res.to_dict(),
                "replay": replay_res.to_dict(),
                "report": report_obj,
            }
        )
    return suite


def _run_oracle_style_suite(repo_root: Path, out_dir: Path, *, py: str, seed: str, scope: str) -> list[dict[str, Any]]:
    scenarios = ["explicit_position", "inferred_position", "shop_consumable"]
    suite: list[dict[str, Any]] = []
    for scenario in scenarios:
        session_path = out_dir / "sessions" / f"{scenario}.jsonl"
        fixture_dir = out_dir / "fixtures_oracle" / scenario
        report_path = out_dir / "reports" / f"oracle_{scenario}.json"
        trace_path = out_dir / "reports" / f"oracle_trace_{scenario}.jsonl"

        build_session_cmd = [
            py,
            "-B",
            "trainer/build_p37_synthetic_session.py",
            "--out",
            str(session_path),
            "--seed",
            f"{seed}_{scenario}",
            "--scenario",
            scenario,
        ]
        build_session_res = _run(build_session_cmd, cwd=repo_root, label=f"build_session_{scenario}")

        to_fixture_res = StepResult(label=f"to_fixture_{scenario}", ok=False, command=[], returncode=1, stdout_tail="", stderr_tail="")
        replay_res = StepResult(label=f"replay_oracle_{scenario}", ok=False, command=[], returncode=1, stdout_tail="", stderr_tail="", output_path=str(report_path))
        report_obj: dict[str, Any] = {}
        if build_session_res.ok:
            to_fixture_cmd = [py, "-B", "trainer/real_trace_to_fixture.py", "--in", str(session_path), "--out-dir", str(fixture_dir), "--seed", f"{seed}_{scenario}"]
            to_fixture_res = _run(to_fixture_cmd, cwd=repo_root, label=f"to_fixture_{scenario}")
            if to_fixture_res.ok:
                replay_cmd = [
                    py,
                    "-B",
                    "sim/tests/run_real_action_replay_fixture.py",
                    "--fixture-dir",
                    str(fixture_dir),
                    "--scope",
                    scope,
                    "--out",
                    str(report_path),
                    "--out-trace",
                    str(trace_path),
                    "--fail-fast",
                ]
                replay_res = _run(replay_cmd, cwd=repo_root, label=f"replay_oracle_{scenario}", output_path=str(report_path))
                if report_path.exists():
                    report_obj = _read_json(report_path)

        suite.append(
            {
                "suite": "oracle_based",
                "scenario": scenario,
                "session_path": str(session_path),
                "fixture_dir": str(fixture_dir),
                "report_path": str(report_path),
                "build_session": build_session_res.to_dict(),
                "to_fixture": to_fixture_res.to_dict(),
                "replay": replay_res.to_dict(),
                "report": report_obj,
            }
        )
    return suite


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_suite = _run_sim_fixture_suite(repo_root, out_dir, py=args.python, seed=args.seed, scope=args.scope)
    oracle_suite = _run_oracle_style_suite(repo_root, out_dir, py=args.python, seed=args.seed, scope=args.scope)
    all_items = sim_suite + oracle_suite

    replay_reports = [item.get("report") for item in all_items if isinstance(item.get("report"), dict)]
    diff_fail_total = int(sum(int((report or {}).get("diff_fail") or 0) for report in replay_reports))
    failed_items = [
        item
        for item in all_items
        if not bool(item.get("build", item.get("build_session", {})).get("ok", True))
        or not bool(item.get("replay", {}).get("ok", False))
        or str((item.get("report") or {}).get("status") or "").lower() != "pass"
    ]

    report = {
        "schema": "p37_action_fidelity_report_v1",
        "seed": args.seed,
        "scope": args.scope,
        "out_dir": str(out_dir),
        "fixtures_total": len(all_items),
        "fixtures_pass": len(all_items) - len(failed_items),
        "diff_fail": diff_fail_total,
        "status": "PASS" if (len(failed_items) == 0 and diff_fail_total == 0) else "FAIL",
        "sim_only": sim_suite,
        "oracle_based": oracle_suite,
    }
    report_path = out_dir / "report_p37.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# P37 Action Fidelity Report",
        "",
        f"- seed: {args.seed}",
        f"- scope: {args.scope}",
        f"- fixtures_total: {report['fixtures_total']}",
        f"- fixtures_pass: {report['fixtures_pass']}",
        f"- diff_fail: {report['diff_fail']}",
        f"- status: {report['status']}",
        "",
        "## Fixtures",
        "",
        "| suite | scenario | replay_status | diff_fail | report |",
        "|---|---|---|---:|---|",
    ]
    for item in all_items:
        rep = item.get("report") if isinstance(item.get("report"), dict) else {}
        md_lines.append(
            "| {suite} | {scenario} | {status} | {diff_fail} | {path} |".format(
                suite=item.get("suite"),
                scenario=item.get("scenario"),
                status=str(rep.get("status") or "missing"),
                diff_fail=int(rep.get("diff_fail") or 0),
                path=item.get("report_path"),
            )
        )
    (out_dir / "report_p37.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(report_path), "status": report["status"], "diff_fail": report["diff_fail"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
