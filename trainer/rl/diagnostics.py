from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def _reward_distribution(rewards: list[float]) -> dict[str, Any]:
    if not rewards:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    ordered = sorted(rewards)
    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * float(q)
        lo = int(pos)
        hi = min(len(ordered) - 1, lo + 1)
        frac = pos - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    return {
        "count": len(rewards),
        "mean": float(statistics.mean(rewards)),
        "std": float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "min": float(min(rewards)),
        "max": float(max(rewards)),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
    }


def run_diagnostics(
    *,
    progress_jsonl: str | Path,
    rollout_buffers: list[str | Path] | None = None,
    rollout_root: str | Path | None = None,
    out_dir: str | Path = "docs/artifacts/p44/diagnostics",
    action_topk: int = 16,
) -> dict[str, Any]:
    progress_rows = _read_jsonl(Path(progress_jsonl).resolve())
    rollout_paths: list[Path] = []
    for raw in rollout_buffers or []:
        if str(raw).strip():
            rollout_paths.append(Path(raw).resolve())
    if rollout_root is not None:
        root = Path(rollout_root).resolve()
        if root.exists():
            rollout_paths.extend(sorted(root.rglob("rollout_buffer.jsonl")))
    unique_rollout_paths = []
    seen = set()
    for path in rollout_paths:
        token = str(path)
        if token in seen or not path.exists():
            continue
        seen.add(token)
        unique_rollout_paths.append(path)

    rollout_rows: list[dict[str, Any]] = []
    for path in unique_rollout_paths:
        rollout_rows.extend(_read_jsonl(path))

    entropies = [
        {
            "ts": str(row.get("ts") or ""),
            "seed": str(row.get("seed") or ""),
            "update": _safe_int(row.get("update"), 0),
            "entropy": _safe_float(row.get("entropy"), 0.0),
            "stage": str(row.get("curriculum_stage") or ""),
        }
        for row in progress_rows
        if row.get("entropy") is not None
    ]
    rewards = [_safe_float(row.get("reward"), 0.0) for row in rollout_rows]
    invalid_steps = sum(1 for row in rollout_rows if bool(row.get("invalid_action")))
    action_counter: Counter[str] = Counter()
    for row in rollout_rows:
        action_counter[str(_safe_int(row.get("action"), 0))] += 1

    payload = {
        "schema": "p44_diagnostics_v1",
        "generated_at": _now_iso(),
        "progress_rows": len(progress_rows),
        "rollout_buffer_count": len(unique_rollout_paths),
        "rollout_rows": len(rollout_rows),
        "policy_entropy_trend": entropies,
        "reward_distribution": _reward_distribution(rewards),
        "invalid_action_rate": float(invalid_steps) / float(max(1, len(rollout_rows))),
        "action_frequency": {
            key: action_counter[key]
            for key in [token for token, _count in action_counter.most_common(max(1, int(action_topk)))]
        },
    }

    output_dir = Path(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnostics.json"
    md_path = output_dir / "diagnostics_report.md"
    _write_json(json_path, payload)
    lines = [
        "# P44 Diagnostics Report",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- progress_rows: `{payload.get('progress_rows')}`",
        f"- rollout_buffer_count: `{payload.get('rollout_buffer_count')}`",
        f"- rollout_rows: `{payload.get('rollout_rows')}`",
        "",
        "## Policy Entropy Trend",
    ]
    if entropies:
        for row in entropies[-12:]:
            lines.append(
                "- update={update} seed={seed} stage={stage} entropy={entropy:.6f}".format(
                    update=_safe_int(row.get("update"), 0),
                    seed=row.get("seed"),
                    stage=row.get("stage") or "n/a",
                    entropy=_safe_float(row.get("entropy"), 0.0),
                )
            )
    else:
        lines.append("- none")

    reward_stats = payload.get("reward_distribution") if isinstance(payload.get("reward_distribution"), dict) else {}
    lines.extend(
        [
            "",
            "## Reward Distribution",
            f"- mean: {_safe_float(reward_stats.get('mean'), 0.0):.6f}",
            f"- std: {_safe_float(reward_stats.get('std'), 0.0):.6f}",
            f"- min: {_safe_float(reward_stats.get('min'), 0.0):.6f}",
            f"- p50: {_safe_float(reward_stats.get('p50'), 0.0):.6f}",
            f"- p90: {_safe_float(reward_stats.get('p90'), 0.0):.6f}",
            f"- max: {_safe_float(reward_stats.get('max'), 0.0):.6f}",
            "",
            "## Invalid Action Rate",
            f"- invalid_action_rate: {_safe_float(payload.get('invalid_action_rate'), 0.0):.6f}",
            "",
            "## Action Frequency",
        ]
    )
    action_frequency = payload.get("action_frequency") if isinstance(payload.get("action_frequency"), dict) else {}
    if action_frequency:
        for action, count in action_frequency.items():
            lines.append(f"- action={action}: count={int(count)}")
    else:
        lines.append("- none")
    _write_markdown(md_path, lines)
    return {
        "status": "ok",
        "diagnostics_json": str(json_path),
        "diagnostics_report_md": str(md_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P44 RL diagnostics generator.")
    parser.add_argument("--progress-jsonl", required=True)
    parser.add_argument("--rollout-root", default="")
    parser.add_argument("--rollout-buffers", default="", help="Optional comma-separated buffer paths.")
    parser.add_argument("--out-dir", default="docs/artifacts/p44/diagnostics")
    parser.add_argument("--action-topk", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    buffers = [token.strip() for token in str(args.rollout_buffers).split(",") if token.strip()]
    summary = run_diagnostics(
        progress_jsonl=args.progress_jsonl,
        rollout_buffers=buffers,
        rollout_root=(args.rollout_root if str(args.rollout_root).strip() else None),
        out_dir=args.out_dir,
        action_topk=max(1, int(args.action_topk)),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
