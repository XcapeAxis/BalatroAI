from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.core.engine import SimEnv
from sim.core.hashing import (
    hand_core_projection,
    p14_real_action_observed_core_projection,
    state_hash_full,
    state_hash_hand_core,
    state_hash_p14_real_action_observed_core,
)
from sim.core.score_observed import compute_score_observed
from sim.oracle.extract_rng_events import extract_rng_events


SCOPE_TO_HASH_KEY = {
    "p14_real_action_observed_core": "state_hash_p14_real_action_observed_core",
    "hand_core": "state_hash_hand_core",
    "full": "state_hash_full",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay executed real action fixture on simulator and diff with oracle trace.")
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument(
        "--scope",
        choices=["p14_real_action_observed_core", "hand_core", "full"],
        default="p14_real_action_observed_core",
    )
    parser.add_argument("--out", required=True, help="Replay report json path.")
    parser.add_argument("--out-trace", default="", help="Optional sim trace output jsonl.")
    parser.add_argument("--dump-on-diff", default="", help="Directory for mismatch dumps.")
    parser.add_argument("--dump-scope-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_no}: invalid json: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_no}: expected object")
            rows.append(item)
    return rows


def _first_diff_path(a: Any, b: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(a) is not type(b):
        return path, a, b
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            if key not in a:
                return f"{path}.{key}", None, b[key]
            if key not in b:
                return f"{path}.{key}", a[key], None
            diff = _first_diff_path(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}.length", len(a), len(b)
        for i, (av, bv) in enumerate(zip(a, b)):
            diff = _first_diff_path(av, bv, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        return path, a, b
    return None


def _parse_path_tokens(path_str: str) -> list[str | int]:
    s = str(path_str or "").strip()
    if s.startswith("$"):
        s = s[1:]
    if s.startswith("."):
        s = s[1:]
    tokens: list[str | int] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == ".":
            i += 1
            continue
        if ch == "[":
            j = s.find("]", i)
            if j == -1:
                break
            t = s[i + 1 : j].strip()
            if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
                tokens.append(int(t))
            else:
                tokens.append(t)
            i = j + 1
            continue
        j = i
        while j < len(s) and s[j] not in ".[":
            j += 1
        token = s[i:j]
        if token:
            tokens.append(token)
        i = j
    return tokens


def _get_by_path(obj: Any, path_str: str) -> tuple[Any, bool]:
    cur = obj
    for token in _parse_path_tokens(path_str):
        if isinstance(token, int):
            if not isinstance(cur, list) or token < 0 or token >= len(cur):
                return None, False
            cur = cur[token]
        else:
            if not isinstance(cur, dict) or token not in cur:
                return None, False
            cur = cur[token]
    return cur, True


def _projection(scope: str, state: dict[str, Any]) -> Any:
    if scope == "p14_real_action_observed_core":
        return p14_real_action_observed_core_projection(state)
    if scope == "hand_core":
        return hand_core_projection(state)
    return state


def _write_dump(
    *,
    path: Path,
    scope: str,
    projection: Any,
    first_diff_path: str | None,
    dump_scope_only: bool,
) -> None:
    payload: dict[str, Any] = {"scope": scope, "projection": projection}
    if first_diff_path:
        payload["first_diff_path"] = first_diff_path
        if not dump_scope_only:
            sub, ok = _get_by_path(projection, first_diff_path)
            payload["first_diff_subtree"] = sub if ok else "__path_not_found__"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    fixture_dir = Path(args.fixture_dir)
    snapshot_path = fixture_dir / "oracle_start_snapshot_real.json"
    action_path = fixture_dir / "action_trace_real.jsonl"
    oracle_trace_path = fixture_dir / "oracle_trace_real.jsonl"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"missing snapshot: {snapshot_path}")
    if not action_path.exists():
        raise FileNotFoundError(f"missing action trace: {action_path}")
    if not oracle_trace_path.exists():
        raise FileNotFoundError(f"missing oracle trace: {oracle_trace_path}")

    snapshot = _load_json(snapshot_path)
    actions = _load_jsonl(action_path)
    oracle_trace = _load_jsonl(oracle_trace_path)
    if not actions:
        raise RuntimeError("action_trace_real.jsonl is empty")

    seed = str((snapshot.get("rng") or {}).get("seed") or "AAAAAAA")
    rng_mode = str((snapshot.get("rng") or {}).get("mode") or "oracle_stream")
    rng_cursor = int((snapshot.get("rng") or {}).get("cursor") or 0)

    env = SimEnv(seed=seed)
    state = env.reset(from_snapshot=snapshot)

    sim_lines: list[dict[str, Any]] = []
    mismatches = 0
    first_diff_step: int | None = None
    first_diff_path: str | None = None
    oracle_hash: str | None = None
    sim_hash: str | None = None
    dumped_oracle: str | None = None
    dumped_sim: str | None = None

    hash_key = SCOPE_TO_HASH_KEY[args.scope]
    dump_dir = Path(args.dump_on_diff) if args.dump_on_diff else None

    for step_id, action in enumerate(actions):
        action_dict = dict(action)
        try:
            next_state, reward, done, info = env.step(action_dict)
        except Exception as exc:
            report = {
                "status": "gen_fail",
                "failure_reason": f"sim_step_failed:{exc}",
                "step_id": step_id,
                "fixture_dir": str(fixture_dir),
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1

        events = extract_rng_events(state, next_state)
        rng_cursor += len(events)
        canonical = to_canonical_state(
            next_state,
            rng_mode=rng_mode,
            seed=seed,
            rng_cursor=rng_cursor,
            rng_events=events,
        )
        score_observed = compute_score_observed(state, next_state)
        canonical_obs = dict(canonical)
        canonical_obs["score_observed"] = dict(score_observed)
        if isinstance(action_dict.get("rng_replay"), dict):
            canonical_obs["rng_replay"] = dict(action_dict.get("rng_replay") or {})

        sim_line = {
            "schema_version": "trace_v1",
            "step_id": step_id,
            "phase": str(canonical.get("phase") or "UNKNOWN"),
            "action": action_dict,
            "state_hash_full": state_hash_full(canonical),
            "state_hash_hand_core": state_hash_hand_core(canonical),
            "state_hash_p14_real_action_observed_core": state_hash_p14_real_action_observed_core(canonical_obs),
            "reward": float(reward),
            "done": bool(done),
            "score_observed": score_observed,
            "rng_replay": dict(action_dict.get("rng_replay")) if isinstance(action_dict.get("rng_replay"), dict) else {"enabled": False, "source": "", "outcomes": []},
            "info": {"source": "sim_real_action_replay", "engine_info": info},
            "canonical_state_snapshot": canonical,
        }
        sim_lines.append(sim_line)

        if step_id < len(oracle_trace):
            oracle_line = oracle_trace[step_id]
            if oracle_line.get(hash_key) != sim_line.get(hash_key):
                mismatches += 1
                if first_diff_step is None:
                    first_diff_step = step_id
                    oracle_hash = str(oracle_line.get(hash_key))
                    sim_hash = str(sim_line.get(hash_key))
                    oracle_snap = oracle_line.get("canonical_state_snapshot")
                    sim_snap = sim_line.get("canonical_state_snapshot")
                    oracle_proj = _projection(args.scope, oracle_snap if isinstance(oracle_snap, dict) else {})
                    sim_proj = _projection(args.scope, sim_snap if isinstance(sim_snap, dict) else {})
                    diff = _first_diff_path(oracle_proj, sim_proj)
                    if diff is not None:
                        first_diff_path = diff[0]
                    if dump_dir is not None:
                        dump_dir.mkdir(parents=True, exist_ok=True)
                        oracle_path = dump_dir / f"oracle_step_{step_id}_{args.scope}.json"
                        sim_path = dump_dir / f"sim_step_{step_id}_{args.scope}.json"
                        _write_dump(
                            path=oracle_path,
                            scope=args.scope,
                            projection=oracle_proj,
                            first_diff_path=first_diff_path,
                            dump_scope_only=bool(args.dump_scope_only),
                        )
                        _write_dump(
                            path=sim_path,
                            scope=args.scope,
                            projection=sim_proj,
                            first_diff_path=first_diff_path,
                            dump_scope_only=bool(args.dump_scope_only),
                        )
                        dumped_oracle = str(oracle_path)
                        dumped_sim = str(sim_path)
                if args.fail_fast:
                    break

        state = next_state

    out_trace = Path(args.out_trace) if args.out_trace else fixture_dir / "sim_trace_real.jsonl"
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    with out_trace.open("w", encoding="utf-8", newline="\n") as f:
        for row in sim_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    status = "pass"
    failure_reason = None
    if len(oracle_trace) != len(actions):
        status = "oracle_fail"
        failure_reason = f"trace_length_mismatch oracle={len(oracle_trace)} actions={len(actions)}"
    if mismatches > 0:
        status = "diff_fail"
        failure_reason = f"hash_mismatch scope={args.scope}"

    report = {
        "status": status,
        "scope": args.scope,
        "fixture_dir": str(fixture_dir),
        "actions_count": len(actions),
        "oracle_states": len(oracle_trace),
        "sim_states": len(sim_lines),
        "diff_fail": mismatches,
        "failure_reason": failure_reason,
        "first_diff_step": first_diff_step,
        "first_diff_path": first_diff_path,
        "oracle_hash": oracle_hash,
        "sim_hash": sim_hash,
        "dumped_oracle": dumped_oracle,
        "dumped_sim": dumped_sim,
        "sim_trace": str(out_trace),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
