"""Basic tests for safe_run.sh (skipped when bash is unavailable)."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE_RUN_SH = ROOT / "safe_run.sh"
HAS_BASH = shutil.which("bash") is not None


@unittest.skipUnless(HAS_BASH and SAFE_RUN_SH.exists(), "bash/safe_run.sh not available")
class TestSafeRunSh(unittest.TestCase):
    def test_success_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "ok.summary.json"
            proc = subprocess.run(
                [
                    "bash",
                    str(SAFE_RUN_SH),
                    "--timeout",
                    "5",
                    "--no-echo",
                    "--summary-json",
                    str(summary),
                    "bash",
                    "-lc",
                    "echo hi; exit 0",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema"), "safe_run_result_v1")
            self.assertEqual(int(payload.get("exit_code") or -1), 0)

    def test_timeout_returns_124(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "timeout.summary.json"
            proc = subprocess.run(
                [
                    "bash",
                    str(SAFE_RUN_SH),
                    "--timeout",
                    "1",
                    "--no-echo",
                    "--summary-json",
                    str(summary),
                    "bash",
                    "-lc",
                    "sleep 3",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(proc.returncode, 124, msg=proc.stdout + proc.stderr)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(bool(payload.get("timed_out")))


if __name__ == "__main__":
    unittest.main()

