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
    economy_core_projection,
    hand_core_projection,
    p0_hand_score_core_projection,
    rng_events_core_projection,
    score_core_projection,
    state_hash_economy_core,
    state_hash_full,
    state_hash_hand_core,
    state_hash_p0_hand_score_core,
    state_hash_rng_events_core,
    state_hash_score_core,
    state_hash_zones_core,
    state_hash_zones_counts_core,
    zones_core_projection,
    zones_counts_core_projection,
)
from sim.core.validate import validate_action, validate_state, validate_trace_line
from sim.oracle.extract_rng_events import extract_rng_events

SCOPE_TO_HASH_KEY = {
    "hand_core": "state_hash_hand_core",
    "score_core": "state_hash_score_core",
    "p0_hand_score_core": "state_hash_p0_hand_score_core",
    "zones_core": "state_hash_zones_core",
    "zones_counts_core": "state_hash_zones_counts_core",
    "economy_core": "state_hash_economy_core",
    "rng_events_core": "state_hash_rng_events_core",
    "full": "state_hash_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run directed fixture from snapshot and action trace.")
    parser.add_argument("--oracle-snapshot", required=True, help="Canonical snapshot JSON file used to initialize simulator.")
    parser.add_argument("--action-trace", required=True, help="Action trace jsonl file.")
    parser.add_argument("--oracle-trace", help="Optional oracle trace jsonl for step-by-step diff.")
    parser.add_argument(
        "--scope",
        choices=["hand_core", "score_core", "p0_hand_score_core", "zones_core", "zones_counts_core", "economy_core", "rng_events_core", "full"],
        default="hand_core",
    )
    parser.add_argument("--check-start", action="store_true", help="Compare oracle snapshot vs simulator reset(from_snapshot) before replay.")
    parser.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--out-trace", default="sim/runtime/directed_sim_trace.jsonl")
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--dump-on-diff", default=None, help="Directory for mismatch dump artifacts.")
    parser.add_argument("--dump-scope-only", action="store_true", help="Dump scope projection only, omit first_diff_path subtree.")
    return parser.parse_args()


def load_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"snapshot not found: {p}")

    if p.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("canonical_state_snapshot"), dict):
                snapshot = obj["canonical_state_snapshot"]
            elif isinstance(obj, dict):
                snapshot = obj
            else:
                continue
            validate_state(snapshot)
            return snapshot
        raise ValueError(f"no usable snapshot found in jsonl: {p}")

    obj = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(obj, dict) and isinstance(obj.get("canonical_state_snapshot"), dict):
        snapshot = obj["canonical_state_snapshot"]
    elif isinstance(obj, dict):
        snapshot = obj
    else:
        raise ValueError(f"snapshot json must be object: {p}")
    validate_state(snapshot)
    return snapshot


def load_action_trace(path: str | Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"action line {line_no} must be object")
            validate_action(obj)
            actions.append(obj)
    return actions


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"trace line {line_no} must be object")
            out.append(obj)
    return out


def _first_diff_path(a: Any, b: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(a) is not type(b):
        return path, a, b
    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for k in keys:
            if k not in a:
                return f"{path}.{k}", None, b[k]
            if k not in b:
                return f"{path}.{k}", a[k], None
            diff = _first_diff_path(a[k], b[k], f"{path}.{k}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}.length", len(a), len(b)
        for i, (ai, bi) in enumerate(zip(a, b)):
            diff = _first_diff_path(ai, bi, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        return path, a, b
    return None


def _scope_projection(scope: str, state: dict[str, Any] | None) -> Any:
    if not isinstance(state, dict):
        return None
    if scope == "hand_core":
        return hand_core_projection(state)
    if scope == "score_core":
        return score_core_projection(state)
    if scope == "p0_hand_score_core":
        return p0_hand_score_core_projection(state)
    if scope == "zones_core":
        return zones_core_projection(state)
    if scope == "zones_counts_core":
        return zones_counts_core_projection(state)
    if scope == "economy_core":
        return economy_core_projection(state)
    if scope == "rng_events_core":
        return rng_events_core_projection(state)
    return state


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
            idx_text = s[i + 1 : j].strip()
            if idx_text.isdigit() or (idx_text.startswith("-") and idx_text[1:].isdigit()):
                tokens.append(int(idx_text))
            else:
                tokens.append(idx_text)
            i = j + 1
            continue

        j = i
        while j < len(s) and s[j] not in ".[":
            j += 1
        key = s[i:j]
        if key:
            tokens.append(key)
        i = j

    return tokens


def _get_by_path(obj: Any, path_str: str) -> tuple[Any, bool]:
    current = obj
    for token in _parse_path_tokens(path_str):
        if isinstance(token, int):
            if not isinstance(current, list):
                return None, False
            if token < 0 or token >= len(current):
                return None, False
            current = current[token]
        else:
            if not isinstance(current, dict):
                return None, False
            if token not in current:
                return None, False
            current = current[token]
    return current, True


def _write_dump_payload(
    *,
    path: Path,
    scope: str,
    projection: Any,
    first_diff_path: str | None,
    dump_scope_only: bool,
) -> None:
    payload: dict[str, Any] = {
        "scope": scope,
        "projection": projection,
    }
    if first_diff_path:
        payload["first_diff_path"] = first_diff_path
        if not dump_scope_only:
            subtree, ok = _get_by_path(projection, first_diff_path)
            payload["first_diff_subtree"] = subtree if ok else "__path_not_found__"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_step_artifacts(
    *,
    step_id: int,
    scope: str,
    dump_dir: Path,
    oracle_projection: Any,
    sim_projection: Any,
    first_diff_path: str | None,
    dump_scope_only: bool,
) -> tuple[str, str]:
    dump_dir.mkdir(parents=True, exist_ok=True)

    oracle_path = dump_dir / f"oracle_step_{step_id}_{scope}.json"
    sim_path = dump_dir / f"sim_step_{step_id}_{scope}.json"

    _write_dump_payload(
        path=oracle_path,
        scope=scope,
        projection=oracle_projection,
        first_diff_path=first_diff_path,
        dump_scope_only=dump_scope_only,
    )
    _write_dump_payload(
        path=sim_path,
        scope=scope,
        projection=sim_projection,
        first_diff_path=first_diff_path,
        dump_scope_only=dump_scope_only,
    )
    return str(oracle_path), str(sim_path)


def _dump_start_artifacts(
    *,
    scope: str,
    dump_dir: Path,
    oracle_projection: Any,
    sim_projection: Any,
    first_diff_path: str | None,
    dump_scope_only: bool,
) -> tuple[str, str]:
    dump_dir.mkdir(parents=True, exist_ok=True)

    oracle_path = dump_dir / f"oracle_start_{scope}.json"
    sim_path = dump_dir / f"sim_start_{scope}.json"

    _write_dump_payload(
        path=oracle_path,
        scope=scope,
        projection=oracle_projection,
        first_diff_path=first_diff_path,
        dump_scope_only=dump_scope_only,
    )
    _write_dump_payload(
        path=sim_path,
        scope=scope,
        projection=sim_projection,
        first_diff_path=first_diff_path,
        dump_scope_only=dump_scope_only,
    )
    return str(oracle_path), str(sim_path)


def main() -> int:
    args = parse_args()

    snapshot = load_snapshot(args.oracle_snapshot)
    actions = load_action_trace(args.action_trace)
    if not actions:
        print("ERROR: empty action trace")
        return 2

    oracle_trace: list[dict[str, Any]] = []
    if args.oracle_trace:
        oracle_trace = load_trace(args.oracle_trace)

    out_path = Path(args.out_trace)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dump_dir: Path | None = None
    if args.dump_on_diff:
        dump_dir = Path(args.dump_on_diff)

    seed = str((snapshot.get("rng") or {}).get("seed") or "AAAAAAA")
    rng_mode = str((snapshot.get("rng") or {}).get("mode") or "native")
    rng_cursor = int((snapshot.get("rng") or {}).get("cursor") or 0)

    env = SimEnv(seed=seed)
    state = env.reset(from_snapshot=snapshot)

    # Optional baseline check: compare start snapshot before replay.
    if args.check_start:
        sim_start_canonical = to_canonical_state(
            state,
            rng_mode=rng_mode,
            seed=seed,
            rng_cursor=rng_cursor,
            rng_events=list((snapshot.get("rng") or {}).get("events") or []),
        )
        oracle_start_projection = _scope_projection(args.scope, snapshot)
        sim_start_projection = _scope_projection(args.scope, sim_start_canonical)
        start_diff = _first_diff_path(oracle_start_projection, sim_start_projection)
        if start_diff is not None:
            path, oracle_val, sim_val = start_diff
            print(f"START_MISMATCH scope={args.scope}")
            print(f"start_first_diff_path={path}")
            print(f"oracle_start_value={oracle_val}")
            print(f"sim_start_value={sim_val}")
            if dump_dir is not None:
                dumped_oracle, dumped_sim = _dump_start_artifacts(
                    scope=args.scope,
                    dump_dir=dump_dir,
                    oracle_projection=oracle_start_projection,
                    sim_projection=sim_start_projection,
                    first_diff_path=path,
                    dump_scope_only=bool(args.dump_scope_only),
                )
                print(f"dumped_oracle={dumped_oracle}, dumped_sim={dumped_sim}")
            if args.fail_fast:
                print(f"wrote directed sim trace: {out_path}")
                return 1

    mismatches = 0
    hash_key = SCOPE_TO_HASH_KEY[args.scope]

    with out_path.open("w", encoding="utf-8-sig", newline="\n") as fp:
        for step_id, action in enumerate(actions):
            executed_action = dict(action)
            try:
                next_state, reward, done, info = env.step(executed_action)
            except Exception as exc:
                print(f"ERROR: step {step_id} action failed: {exc}")
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

            include_snapshot = (
                step_id == 0
                or step_id == len(actions) - 1
                or done
                or (args.snapshot_every > 0 and (step_id + 1) % args.snapshot_every == 0)
            )

            trace_line: dict[str, Any] = {
                "schema_version": "trace_v1",
                "step_id": step_id,
                "phase": str(canonical.get("phase") or "UNKNOWN"),
                "action": executed_action,
                "state_hash_full": state_hash_full(canonical),
                "state_hash_hand_core": state_hash_hand_core(canonical),
                "state_hash_score_core": state_hash_score_core(canonical),
                "state_hash_p0_hand_score_core": state_hash_p0_hand_score_core(canonical),
                "state_hash_zones_core": state_hash_zones_core(canonical),
                "state_hash_zones_counts_core": state_hash_zones_counts_core(canonical),
                "state_hash_economy_core": state_hash_economy_core(canonical),
                "state_hash_rng_events_core": state_hash_rng_events_core(canonical),
                "reward": float(reward),
                "done": bool(done),
                "info": {
                    "source": "directed_sim",
                    "engine_info": info,
                },
            }
            if include_snapshot:
                trace_line["canonical_state_snapshot"] = canonical

            validate_trace_line(trace_line)
            fp.write(json.dumps(trace_line, ensure_ascii=False) + "\n")

            if oracle_trace and step_id < len(oracle_trace):
                oracle_line = oracle_trace[step_id]
                oracle_hash = oracle_line.get(hash_key)
                sim_hash = trace_line.get(hash_key)
                if oracle_hash != sim_hash:
                    mismatches += 1
                    print(f"MISMATCH step={step_id} scope={args.scope}")
                    print(f"oracle_hash={oracle_hash}")
                    print(f"sim_hash={sim_hash}")

                    oracle_snap = oracle_line.get("canonical_state_snapshot")
                    sim_snap = trace_line.get("canonical_state_snapshot")
                    first_diff_path: str | None = None

                    oracle_proj = _scope_projection(args.scope, oracle_snap if isinstance(oracle_snap, dict) else None)
                    sim_proj = _scope_projection(args.scope, sim_snap if isinstance(sim_snap, dict) else None)

                    if oracle_proj is not None and sim_proj is not None:
                        diff = _first_diff_path(oracle_proj, sim_proj)
                        if diff is not None:
                            path, oracle_val, sim_val = diff
                            first_diff_path = path
                            print(f"first_diff_path={path}")
                            print(f"oracle_value={oracle_val}")
                            print(f"sim_value={sim_val}")
                    else:
                        print("No snapshots on mismatch step. Re-run with --snapshot-every 1 and oracle trace snapshots.")

                    if dump_dir is not None:
                        dumped_oracle, dumped_sim = _dump_step_artifacts(
                            step_id=step_id,
                            scope=args.scope,
                            dump_dir=dump_dir,
                            oracle_projection=oracle_proj,
                            sim_projection=sim_proj,
                            first_diff_path=first_diff_path,
                            dump_scope_only=bool(args.dump_scope_only),
                        )
                        print(f"dumped_oracle={dumped_oracle}, dumped_sim={dumped_sim}")

                    if args.fail_fast:
                        print(f"wrote directed sim trace: {out_path}")
                        return 1

            state = next_state
            if done and args.stop_on_done:
                break

    if oracle_trace and len(oracle_trace) != len(actions):
        print(f"WARNING: oracle trace length={len(oracle_trace)} action steps={len(actions)}")

    print(f"wrote directed sim trace: {out_path}")
    if oracle_trace:
        if mismatches:
            print(f"directed diff finished with mismatches={mismatches}")
            return 1
        print(f"directed diff OK for {len(actions)} steps in scope={args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
