"""Integration tests for SafeRun orchestration properties."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE_RUN_PS1 = ROOT / "scripts" / "safe_run.ps1"


def _run_safe(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SAFE_RUN_PS1),
    ] + args
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class TestSafeRunIntegration(unittest.TestCase):
    def test_nested_safe_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outer_summary = Path(td) / "outer.summary.json"
            inner_summary = Path(td) / "inner.summary.json"
            inner_summary_ps = str(inner_summary).replace("'", "''")
            safe_run_ps = str(SAFE_RUN_PS1).replace("'", "''")
            nested_cmd = (
                f"& '{safe_run_ps}' "
                f"-NoEcho -SummaryJson '{inner_summary_ps}' "
                "powershell -NoProfile -Command \"exit 0\""
            )
            proc = _run_safe(
                [
                    "-TimeoutSec",
                    "20",
                    "-NoEcho",
                    "-SummaryJson",
                    str(outer_summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    nested_cmd,
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(outer_summary.exists())
            self.assertTrue(inner_summary.exists())
            outer = _load_json(outer_summary)
            inner = _load_json(inner_summary)
            self.assertEqual(int(outer.get("exit_code", -1)), 0)
            self.assertEqual(int(inner.get("exit_code", -1)), 0)

    def test_concurrent_runs_have_distinct_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            s1 = Path(td) / "c1.summary.json"
            s2 = Path(td) / "c2.summary.json"
            cmd1 = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SAFE_RUN_PS1),
                "-TimeoutSec",
                "10",
                "-NoEcho",
                "-SummaryJson",
                str(s1),
                "powershell",
                "-NoProfile",
                "-Command",
                "Start-Sleep -Milliseconds 600; Write-Output 'c1'; exit 0",
            ]
            cmd2 = [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SAFE_RUN_PS1),
                "-TimeoutSec",
                "10",
                "-NoEcho",
                "-SummaryJson",
                str(s2),
                "powershell",
                "-NoProfile",
                "-Command",
                "Start-Sleep -Milliseconds 800; Write-Output 'c2'; exit 0",
            ]
            p1 = subprocess.Popen(cmd1, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p2 = subprocess.Popen(cmd2, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            o1, e1 = p1.communicate(timeout=120)
            o2, e2 = p2.communicate(timeout=120)
            self.assertEqual(p1.returncode, 0, msg=(o1 + e1))
            self.assertEqual(p2.returncode, 0, msg=(o2 + e2))

            j1 = _load_json(s1)
            j2 = _load_json(s2)
            self.assertNotEqual(j1["stdout_log"], j2["stdout_log"])
            self.assertNotEqual(j1["stderr_log"], j2["stderr_log"])

    def test_timeout_kills_child_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "tree.summary.json"
            pid_file = Path(td) / "child.pid"
            pid_path = str(pid_file).replace("'", "''")
            ps_script = (
                "$child = Start-Process powershell -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 20' -PassThru; "
                f"Set-Content -LiteralPath '{pid_path}' -Value $child.Id; "
                "Start-Sleep -Seconds 20"
            )
            proc = _run_safe(
                [
                    "-TimeoutSec",
                    "1",
                    "-NoEcho",
                    "-SummaryJson",
                    str(summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    ps_script,
                ],
                timeout=120,
            )
            self.assertEqual(proc.returncode, 124, msg=proc.stdout + proc.stderr)
            self.assertTrue(pid_file.exists(), "child pid file should be produced")
            child_pid = int(pid_file.read_text(encoding="utf-8").strip())
            probe = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"if (Get-Process -Id {child_pid} -ErrorAction SilentlyContinue) {{ exit 1 }} else {{ exit 0 }}",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                probe.returncode,
                0,
                msg=f"child pid {child_pid} still alive after safe_run timeout",
            )


if __name__ == "__main__":
    unittest.main()
