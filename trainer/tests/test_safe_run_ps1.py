"""Tests for scripts/safe_run.ps1 behavior and summary contract."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE_RUN_PS1 = ROOT / "scripts" / "safe_run.ps1"


def _run_safe_run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
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


class TestSafeRunPs1(unittest.TestCase):
    def test_success_exit_code_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "ok.summary.json"
            proc = _run_safe_run(
                [
                    "-TimeoutSec",
                    "5",
                    "-NoEcho",
                    "-SummaryJson",
                    str(summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Write-Output 'hello'; exit 0",
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            self.assertTrue(summary.exists(), "summary json must exist")
            payload = _load_json(summary)
            self.assertEqual(payload.get("schema"), "safe_run_result_v1")
            self.assertEqual(int(payload.get("exit_code", -1)), 0)
            self.assertFalse(bool(payload.get("timed_out")))
            self.assertTrue(Path(payload["stdout_log"]).exists())
            self.assertTrue(Path(payload["stderr_log"]).exists())

    def test_failure_exit_code_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "fail.summary.json"
            proc = _run_safe_run(
                [
                    "-TimeoutSec",
                    "5",
                    "-NoEcho",
                    "-SummaryJson",
                    str(summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Write-Error 'boom'; exit 7",
                ]
            )
            self.assertEqual(proc.returncode, 7, msg=proc.stdout + proc.stderr)
            payload = _load_json(summary)
            self.assertEqual(int(payload.get("exit_code", -1)), 7)
            self.assertFalse(bool(payload.get("timed_out")))

    def test_timeout_exit_124(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "timeout.summary.json"
            proc = _run_safe_run(
                [
                    "-TimeoutSec",
                    "1",
                    "-NoEcho",
                    "-SummaryJson",
                    str(summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Start-Sleep -Seconds 3",
                ],
                timeout=60,
            )
            self.assertEqual(proc.returncode, 124, msg=proc.stdout + proc.stderr)
            payload = _load_json(summary)
            self.assertEqual(int(payload.get("exit_code", -1)), 124)
            self.assertTrue(bool(payload.get("timed_out")))

    def test_large_output_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "large.summary.json"
            proc = _run_safe_run(
                [
                    "-TimeoutSec",
                    "10",
                    "-NoEcho",
                    "-SummaryJson",
                    str(summary),
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "$x='x'*200; 1..3000 | ForEach-Object { Write-Output $x }; exit 0",
                ]
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = _load_json(summary)
            out_path = Path(payload["stdout_log"])
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 100000)


if __name__ == "__main__":
    unittest.main()
