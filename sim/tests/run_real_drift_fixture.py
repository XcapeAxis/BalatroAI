from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from collections import Counter
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


def _diff(a: Any, b: Any, prefix: str = "$") -> list[tuple[str, Any, Any]]:
    diffs: list[tuple[str, Any, Any]] = []
    if type(a) is not type(b):
        diffs.append((prefix, a, b))
        return diffs
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            diffs.extend(_diff(a.get(k), b.get(k), f"{prefix}.{k}"))
        return diffs
    if isinstance(a, list):
        if a != b:
            diffs.append((prefix, a, b))
        return diffs
    if a != b:
        diffs.append((prefix, a, b))
    return diffs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two real-state traces and report projection drift.")
    parser.add_argument("--trace-a", required=True)
    parser.add_argument("--trace-b", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    a = _read_jsonl(Path(args.trace_a))
    b = _read_jsonl(Path(args.trace_b))
    n = min(len(a), len(b))

    mismatch_count = 0
    path_counter: Counter[str] = Counter()
    first_diff: dict[str, Any] | None = None

    for i in range(n):
        pa = a[i].get("projection")
        pb = b[i].get("projection")
        diffs = _diff(pa, pb, "$")
        if not diffs:
            continue
        mismatch_count += len(diffs)
        for p, _, _ in diffs:
            path_counter[p] += 1
        if first_diff is None:
            p, va, vb = diffs[0]
            first_diff = {"step": i, "path": p, "a": va, "b": vb}

    if len(a) != len(b):
        path_counter["$.length"] += 1
        mismatch_count += 1
        if first_diff is None:
            first_diff = {"step": n, "path": "$.length", "a": len(a), "b": len(b)}

    report = {
        "trace_a": args.trace_a,
        "trace_b": args.trace_b,
        "count_a": len(a),
        "count_b": len(b),
        "compared": n,
        "mismatch_count": mismatch_count,
        "top_paths": path_counter.most_common(20),
        "first_diff": first_diff,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
