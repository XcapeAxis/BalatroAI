from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at line {line_no}: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real session trace into fixture-like artifacts.")
    parser.add_argument("--in", dest="inp", required=True, help="Input session jsonl.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(inp)
    if not rows:
        raise RuntimeError("empty session trace")

    first = rows[0]
    start_snapshot = {
        "schema_version": "real_state_snapshot_v1",
        "source": str(inp),
        "phase": str(first.get("phase") or ""),
        "gamestate_min": first.get("gamestate_min"),
    }
    (out_dir / "oracle_start_snapshot_real.json").write_text(
        json.dumps(start_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    action_trace: list[dict[str, Any]] = []
    state_trace: list[dict[str, Any]] = []
    oracle_trace: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        phase = str(row.get("phase") or "UNKNOWN")
        state_min = row.get("gamestate_min")
        state_trace.append(
            {
                "step_id": i,
                "ts": row.get("ts"),
                "phase": phase,
                "projection": state_min,
            }
        )
        oracle_trace.append(
            {
                "step_id": i,
                "phase": phase,
                "state_hash_real_min": str(hash(json.dumps(state_min, ensure_ascii=False, sort_keys=True))),
                "canonical_state_snapshot": state_min,
            }
        )
        if isinstance(row.get("executed_action"), dict):
            action_trace.append(row["executed_action"])

    _write_jsonl(out_dir / "state_trace.jsonl", state_trace)
    _write_jsonl(out_dir / "oracle_trace_real.jsonl", oracle_trace)
    _write_jsonl(out_dir / "action_trace_real.jsonl", action_trace)

    manifest = {
        "input": str(inp),
        "rows": len(rows),
        "state_trace": str(out_dir / "state_trace.jsonl"),
        "oracle_trace": str(out_dir / "oracle_trace_real.jsonl"),
        "action_trace": str(out_dir / "action_trace_real.jsonl"),
        "actions_count": len(action_trace),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
