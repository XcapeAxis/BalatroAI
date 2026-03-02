from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_campaign_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# P24 Campaign Summary",
        "",
        f"- campaign_id: `{summary.get('campaign_id')}`",
        f"- run_id: `{summary.get('run_id')}`",
        f"- status: `{summary.get('status')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- elapsed_sec: `{summary.get('elapsed_sec')}`",
        "",
        "## Stages",
        "",
        "| stage_id | purpose | status | experiments | run_root |",
        "|---|---|---|---:|---|",
    ]
    for row in summary.get("stages", []):
        lines.append(
            "| {stage_id} | {purpose} | {status} | {experiment_count} | {run_root} |".format(
                stage_id=row.get("stage_id"),
                purpose=row.get("purpose"),
                status=row.get("status"),
                experiment_count=row.get("experiment_count", 0),
                run_root=row.get("run_root", ""),
            )
        )
    lines += [
        "",
        "## Post Actions",
    ]
    post = summary.get("post_actions") or {}
    if isinstance(post, dict):
        for key, value in post.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"

