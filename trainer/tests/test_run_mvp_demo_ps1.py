from __future__ import annotations

import socket
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_MVP_DEMO_PS1 = ROOT / "scripts" / "run_mvp_demo.ps1"


def _run_mvp_demo(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(RUN_MVP_DEMO_PS1),
    ] + args
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_detach_rejects_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        proc = _run_mvp_demo(["-Port", str(port), "-Detach"])

    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode != 0
    assert "already in use" in output
