from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    base = Path("docs/artifacts/p16/baseline")
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": "RunP15",
        "status": "FAIL",
        "exit_code": 1,
        "reason": "timeout_or_terminal_io_error",
    }
    (base / "baseline_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            "# P16 Baseline Summary",
            "",
            "- gate: RunP15",
            "- status: FAIL",
            "- exit_code: 1",
            "- reason: timeout_or_terminal_io_error",
        ]
    )
    (base / "baseline_summary.md").write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
