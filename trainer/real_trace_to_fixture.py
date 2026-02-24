from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sim.core.hashing import (
    state_hash_full,
    state_hash_hand_core,
    state_hash_p14_real_action_observed_core,
)
from sim.core.score_observed import compute_score_observed
from sim.oracle.canonicalize_real import canonicalize_real_state


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
    parser.add_argument("--seed", default="AAAAAAA")
    return parser.parse_args()


def _state_min_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_before(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("gamestate_raw_before")
    if isinstance(raw, dict):
        return raw
    raw_alt = row.get("gamestate_raw")
    if isinstance(raw_alt, dict):
        return raw_alt
    return None


def _raw_after(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("gamestate_raw_after")
    if isinstance(raw, dict):
        return raw
    raw_alt = row.get("gamestate_raw")
    if isinstance(raw_alt, dict):
        return raw_alt
    return None


def _rng_replay(row: dict[str, Any]) -> dict[str, Any]:
    outcomes = row.get("outcome_tokens")
    if not isinstance(outcomes, list):
        outcomes = []
    return {
        "enabled": len(outcomes) > 0,
        "source": "real_recording",
        "outcomes": outcomes,
    }


def main() -> int:
    args = _parse_args()
    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(inp)
    if not rows:
        raise RuntimeError("empty session trace")

    start_raw = None
    for row in rows:
        raw = _raw_before(row)
        if isinstance(raw, dict):
            start_raw = raw
            break
    if start_raw is None:
        raise RuntimeError("session lacks gamestate_raw_before; rerun recorder with --include-raw or --execute")

    start_snapshot = canonicalize_real_state(start_raw, seed=args.seed, rng_events=[], rng_cursor=0)
    (out_dir / "oracle_start_snapshot_real.json").write_text(
        json.dumps(start_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    action_trace: list[dict[str, Any]] = []
    state_trace: list[dict[str, Any]] = []
    oracle_trace: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        phase = str(row.get("phase") or "UNKNOWN")
        state_min = row.get("gamestate_min")
        state_trace.append(
            {
                "step_id": i,
                "ts": row.get("ts"),
                "phase": phase,
                "projection": state_min,
                "state_hash_real_min": _state_min_hash(state_min),
            }
        )

        action = row.get("action_sent")
        if not isinstance(action, dict):
            action = row.get("executed_action")
        if not isinstance(action, dict):
            continue

        raw_before = _raw_before(row)
        raw_after = _raw_after(row)
        if not isinstance(raw_before, dict) or not isinstance(raw_after, dict):
            skipped_actions.append(
                {
                    "step_idx": row.get("step_idx", i),
                    "reason": "missing_raw_before_or_after",
                }
            )
            continue

        state_changed = bool(row.get("state_changed", True))
        if not state_changed:
            skipped_actions.append(
                {
                    "step_idx": row.get("step_idx", i),
                    "reason": "no_state_change",
                }
            )
            continue

        score_observed = compute_score_observed(raw_before, raw_after)
        canonical_after = canonicalize_real_state(raw_after, seed=args.seed, rng_events=[], rng_cursor=0)
        canonical_after_obs = dict(canonical_after)
        canonical_after_obs["score_observed"] = dict(score_observed)
        canonical_after_obs["rng_replay"] = _rng_replay(row)

        action_trace.append(dict(action))
        oracle_trace.append(
            {
                "schema_version": "trace_v1",
                "step_id": len(action_trace) - 1,
                "phase": str(canonical_after.get("phase") or phase),
                "action": dict(action),
                "state_hash_full": state_hash_full(canonical_after),
                "state_hash_hand_core": state_hash_hand_core(canonical_after),
                "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(canonical_after_obs),
                "reward": float((row.get("action_result") or {}).get("reward") or 0.0),
                "done": bool((row.get("action_result") or {}).get("done") or False),
                "score_observed": score_observed,
                "rng_replay": _rng_replay(row),
                "canonical_state_snapshot": canonical_after,
                "info": {
                    "source": "real_trace_to_fixture",
                    "session_step_idx": row.get("step_idx", i),
                    "mode": row.get("mode"),
                },
            }
        )

    _write_jsonl(out_dir / "state_trace.jsonl", state_trace)
    _write_jsonl(out_dir / "oracle_trace_real.jsonl", oracle_trace)
    _write_jsonl(out_dir / "action_trace_real.jsonl", action_trace)

    phase_hist: dict[str, int] = {}
    for row in rows:
        key = str(row.get("phase") or "UNKNOWN")
        phase_hist[key] = int(phase_hist.get(key, 0)) + 1

    manifest = {
        "input": str(inp),
        "rows": len(rows),
        "base_url": str(rows[0].get("base_url") or ""),
        "mode": str(rows[0].get("mode") or "shadow"),
        "seed": args.seed,
        "state_trace": str(out_dir / "state_trace.jsonl"),
        "oracle_trace": str(out_dir / "oracle_trace_real.jsonl"),
        "action_trace": str(out_dir / "action_trace_real.jsonl"),
        "actions_count": len(action_trace),
        "phase_distribution": phase_hist,
        "skipped_actions": skipped_actions,
        "arm_token_used": bool(rows[0].get("mode") == "execute"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
