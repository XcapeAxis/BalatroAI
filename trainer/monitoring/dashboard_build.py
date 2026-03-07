from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _campaign_stage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stages = [dict(item) for item in (payload.get("stages") or []) if isinstance(item, dict)]
    active = next((item for item in stages if str(item.get("status") or "") == "running"), None)
    failed = next((item for item in stages if str(item.get("status") or "") == "failed"), None)
    latest = active or failed or (stages[-1] if stages else {})
    return {
        "campaign_id": str(payload.get("campaign_id") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "experiment_id": str(payload.get("experiment_id") or ""),
        "seed": str(payload.get("seed") or ""),
        "stage_id": str((latest or {}).get("stage_id") or ""),
        "status": str((latest or {}).get("status") or ""),
        "state_path": str(payload.get("state_path") or ""),
    }


def _registry_summary(input_root: Path) -> dict[str, Any]:
    registry_path = input_root / "registry" / "checkpoints_registry.json"
    payload = _read_json(registry_path)
    items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
    counts: dict[str, int] = {}
    latest_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        token = str(item.get("status") or "draft")
        counts[token] = int(counts.get(token, 0)) + 1
    for item in items[:8]:
        if isinstance(item, dict):
            latest_items.append(
                {
                    "checkpoint_id": str(item.get("checkpoint_id") or ""),
                    "family": str(item.get("family") or ""),
                    "status": str(item.get("status") or ""),
                }
            )
    return {
        "registry_path": str(registry_path.resolve()),
        "count": len(items),
        "status_counts": counts,
        "latest_items": latest_items,
    }


def collect_dashboard_data(input_root: Path) -> dict[str, Any]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for path in input_root.glob("**/*progress*.jsonl"):
        for row in _read_jsonl(path):
            if str(row.get("schema") or "") != "p49_progress_event_v1":
                continue
            key = (
                str(row.get("run_id") or ""),
                str(row.get("component") or ""),
                str(row.get("seed") or ""),
            )
            latest[key] = row
            if str(row.get("warning") or "").strip():
                warnings.append(row)
    p22_runs_root = input_root / "p22" / "runs"
    latest_p22_summary = []
    if p22_runs_root.exists():
        runs = sorted([path for path in p22_runs_root.iterdir() if path.is_dir()], key=lambda path: path.name)
        if runs:
            summary = _read_json(runs[-1] / "summary_table.json")
            if isinstance(summary, list):
                latest_p22_summary = [row for row in summary if isinstance(row, dict)]
    campaign_states = []
    for path in sorted(input_root.glob("**/campaign_state.json"), key=lambda item: str(item), reverse=True)[:12]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        payload["state_path"] = str(path.resolve())
        campaign_states.append(_campaign_stage_summary(payload))
    return {
        "schema": "p49_dashboard_data_v1",
        "generated_at": _now_iso(),
        "input_root": str(input_root),
        "latest_events": sorted(latest.values(), key=lambda row: (str(row.get("run_id") or ""), str(row.get("component") or ""))),
        "warnings": warnings[-20:],
        "latest_p22_summary": latest_p22_summary,
        "campaign_states": campaign_states,
        "registry_summary": _registry_summary(input_root),
    }


def build_dashboard(input_root: Path, output_dir: Path) -> dict[str, Any]:
    data = collect_dashboard_data(input_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows_html = []
    for row in data.get("latest_events") if isinstance(data.get("latest_events"), list) else []:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('component') or ''))}</td>"
            f"<td>{html.escape(str(row.get('phase') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('learner_device') or ''))}</td>"
            f"<td>{html.escape(str(row.get('rollout_device') or ''))}</td>"
            f"<td>{html.escape(str(row.get('throughput') if row.get('throughput') is not None else '-'))}</td>"
            f"<td>{html.escape(str(row.get('gpu_mem_mb') if row.get('gpu_mem_mb') is not None else '-'))}</td>"
            f"<td><code>{html.escape(json.dumps(metrics, ensure_ascii=False)[:180])}</code></td>"
            "</tr>"
        )

    warnings_html = []
    for row in data.get("warnings") if isinstance(data.get("warnings"), list) else []:
        warnings_html.append(
            "<li>"
            f"{html.escape(str(row.get('run_id') or ''))} / {html.escape(str(row.get('component') or ''))}: "
            f"{html.escape(str(row.get('warning') or ''))}"
            "</li>"
        )

    p22_html = []
    for row in data.get("latest_p22_summary") if isinstance(data.get("latest_p22_summary"), list) else []:
        if not isinstance(row, dict):
            continue
        p22_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('exp_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('mean') or ''))}</td>"
            f"<td>{html.escape(str(row.get('seed_count') or ''))}</td>"
            "</tr>"
        )

    campaign_html = []
    for row in data.get("campaign_states") if isinstance(data.get("campaign_states"), list) else []:
        if not isinstance(row, dict):
            continue
        campaign_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('campaign_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('experiment_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('seed') or ''))}</td>"
            f"<td>{html.escape(str(row.get('stage_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )

    registry_summary = data.get("registry_summary") if isinstance(data.get("registry_summary"), dict) else {}
    registry_counts = registry_summary.get("status_counts") if isinstance(registry_summary.get("status_counts"), dict) else {}
    registry_items = registry_summary.get("latest_items") if isinstance(registry_summary.get("latest_items"), list) else []
    registry_html = []
    for row in registry_items:
        if not isinstance(row, dict):
            continue
        registry_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('checkpoint_id') or ''))}</td>"
            f"<td>{html.escape(str(row.get('family') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>P49 Dashboard</title>
  <style>
    :root {{ --bg: #f4f0e8; --panel: #fffaf2; --ink: #241d16; --muted: #8a745d; --warn: #9d3c2f; }}
    body {{ margin: 0; padding: 24px; font-family: Georgia, 'Times New Roman', serif; background: linear-gradient(180deg, #efe6d6 0%, var(--bg) 100%); color: var(--ink); }}
    h1, h2 {{ margin: 0 0 12px; }}
    .panel {{ background: var(--panel); border: 1px solid #d9ccb6; border-radius: 14px; padding: 18px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(36,29,22,0.08); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e8dcc8; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>P49 GPU Mainline Dashboard</h1>
    <p class="muted">Generated from unified progress events and the latest P22 summary.</p>
    <p><strong>Input:</strong> <code>{html.escape(str(input_root))}</code></p>
    <p><strong>Data:</strong> <code>{html.escape(str((output_dir / "dashboard_data.json").resolve()))}</code></p>
  </div>
  <div class="panel">
    <h2>Active / Latest Progress</h2>
    <table>
      <thead>
        <tr><th>Run</th><th>Component</th><th>Phase</th><th>Status</th><th>Learner</th><th>Rollout</th><th>Throughput</th><th>GPU MB</th><th>Metrics</th></tr>
      </thead>
      <tbody>
        {''.join(rows_html) or '<tr><td colspan="9">No unified progress events found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Warnings</h2>
    <ul>
      {''.join(warnings_html) or '<li>No warnings captured.</li>'}
    </ul>
  </div>
  <div class="panel">
    <h2>Latest P22 Summary</h2>
    <table>
      <thead><tr><th>Experiment</th><th>Status</th><th>Mean</th><th>Seeds</th></tr></thead>
      <tbody>
        {''.join(p22_html) or '<tr><td colspan="4">No P22 summary found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Campaign Stages</h2>
    <table>
      <thead><tr><th>Campaign</th><th>Experiment</th><th>Seed</th><th>Stage</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(campaign_html) or '<tr><td colspan="5">No campaign state found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Checkpoint Registry</h2>
    <p class="muted">registry: <code>{html.escape(str(registry_summary.get('registry_path') or ''))}</code></p>
    <p class="muted">count={html.escape(str(registry_summary.get('count') or 0))} status_counts={html.escape(json.dumps(registry_counts, ensure_ascii=False))}</p>
    <table>
      <thead><tr><th>Checkpoint ID</th><th>Family</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(registry_html) or '<tr><td colspan="3">No registry entries found.</td></tr>'}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "index_html": str((output_dir / "index.html").resolve()),
        "dashboard_data_json": str((output_dir / "dashboard_data.json").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML dashboard from unified P49 progress events.")
    parser.add_argument("--input", default="docs/artifacts")
    parser.add_argument("--output", default="docs/artifacts/dashboard/latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    input_root = Path(args.input)
    if not input_root.is_absolute():
        input_root = (repo_root / input_root).resolve()
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    summary = build_dashboard(input_root, output_dir)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
