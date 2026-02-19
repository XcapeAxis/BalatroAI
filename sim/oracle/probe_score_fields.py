if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim.oracle.canonicalize_real import canonicalize_real_state
from sim.oracle.generate_p0_trace import TARGET_INJECT_KEYS
from trainer.env_client import ConnectionError, RPCError, _call_method, get_state, health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe observable score fields around first oracle action.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--target", required=True)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--action-trace")
    parser.add_argument("--oracle-start-snapshot")
    parser.add_argument("--oracle-start-state")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--wait-sleep", type=float, default=0.05)
    parser.add_argument("--out")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    fixtures_dir = Path(args.fixtures_dir)
    action_trace = Path(args.action_trace) if args.action_trace else fixtures_dir / f"action_trace_{args.target}.jsonl"
    snapshot = Path(args.oracle_start_snapshot) if args.oracle_start_snapshot else fixtures_dir / f"oracle_start_snapshot_{args.target}.json"

    if args.oracle_start_state:
        start_state = Path(args.oracle_start_state)
    else:
        candidate = fixtures_dir / f"oracle_start_state_{args.target}.jkr"
        start_state = candidate if candidate.exists() else None

    return action_trace, snapshot if snapshot.exists() else None, start_state


def _load_first_action(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"action trace not found: {path}")
    with path.open("r", encoding="utf-8-sig") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError("first action line is not object")
            return obj
    raise ValueError(f"action trace is empty: {path}")


def _parse_path_tokens(path: str) -> list[str | int]:
    s = path.strip()
    if s.startswith("$"):
        s = s[1:]
    if s.startswith("."):
        s = s[1:]
    tokens: list[str | int] = []
    i = 0
    while i < len(s):
        if s[i] == ".":
            i += 1
            continue
        if s[i] == "[":
            j = s.find("]", i)
            if j == -1:
                break
            idx = s[i + 1 : j].strip()
            if idx.isdigit() or (idx.startswith("-") and idx[1:].isdigit()):
                tokens.append(int(idx))
            else:
                tokens.append(idx)
            i = j + 1
            continue
        j = i
        while j < len(s) and s[j] not in ".[":
            j += 1
        if j > i:
            tokens.append(s[i:j])
        i = j
    return tokens


def _get_by_path(obj: Any, path: str) -> tuple[Any, bool]:
    cur = obj
    for tok in _parse_path_tokens(path):
        if isinstance(tok, int):
            if not isinstance(cur, list) or tok < 0 or tok >= len(cur):
                return None, False
            cur = cur[tok]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return None, False
            cur = cur[tok]
    return cur, True


def _flatten_numeric(obj: Any, path: str = "$") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_numeric(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten_numeric(v, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = float(obj)
    return out


def _index_offset(action: dict[str, Any]) -> int:
    params = action.get("params")
    if not isinstance(params, dict):
        return 0
    try:
        v = int(params.get("index_base", 0))
    except Exception:
        return 0
    return 1 if v == 1 else 0


def _action_indices_for_rpc(action: dict[str, Any]) -> list[int]:
    offset = _index_offset(action)
    indices = [int(i) for i in (action.get("indices") or [])]
    return [i + offset for i in indices]


def _apply_action(base_url: str, action: dict[str, Any], timeout_sec: float, wait_sleep: float) -> None:
    action_type = str(action.get("action_type") or "WAIT").upper()
    if action_type == "PLAY":
        _call_method(base_url, "play", {"cards": _action_indices_for_rpc(action)}, timeout=timeout_sec)
    elif action_type == "DISCARD":
        _call_method(base_url, "discard", {"cards": _action_indices_for_rpc(action)}, timeout=timeout_sec)
    elif action_type == "SELECT":
        _call_method(base_url, "select", {"index": int(action.get("index", 0))}, timeout=timeout_sec)
    elif action_type == "CASH_OUT":
        _call_method(base_url, "cash_out", {}, timeout=timeout_sec)
    elif action_type == "NEXT_ROUND":
        _call_method(base_url, "next_round", {}, timeout=timeout_sec)
    elif action_type == "START":
        seed = str(action.get("seed") or "AAAAAAA")
        _call_method(base_url, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}, timeout=timeout_sec)
    elif action_type == "MENU":
        _call_method(base_url, "menu", {}, timeout=timeout_sec)
    elif action_type == "SKIP":
        _call_method(base_url, "skip", {}, timeout=timeout_sec)
    elif action_type == "REROLL":
        _call_method(base_url, "reroll", {}, timeout=timeout_sec)
    elif action_type == "WAIT":
        import time

        time.sleep(max(0.0, float(action.get("sleep") or wait_sleep)))
    else:
        raise ValueError(f"unsupported action_type: {action_type}")


def _try_add_target_cards(base_url: str, target: str, timeout_sec: float) -> bool:
    keys = TARGET_INJECT_KEYS.get(target)
    if not keys:
        return False
    added = False
    for key in keys:
        try:
            _call_method(base_url, "add", {"key": key}, timeout=timeout_sec)
            added = True
        except Exception:
            continue
    return added


def main() -> int:
    args = parse_args()

    if not health(args.base_url):
        print(f"ERROR: base_url unhealthy: {args.base_url}")
        return 2

    action_trace_path, snapshot_path, start_state_path = _resolve_paths(args)
    action0 = _load_first_action(action_trace_path)

    if start_state_path and start_state_path.exists():
        _call_method(args.base_url, "load", {"path": str(start_state_path.resolve())}, timeout=args.timeout_sec)
    elif snapshot_path and snapshot_path.exists():
        print(f"WARNING: no .jkr start state found; probing from current live state. snapshot={snapshot_path}")
    else:
        print("WARNING: no start snapshot/state found; probing from current live state.")

    pre_state = get_state(args.base_url, timeout=args.timeout_sec)

    action_error: str | None = None
    retried_with_add = False
    try:
        _apply_action(args.base_url, action0, timeout_sec=args.timeout_sec, wait_sleep=args.wait_sleep)
    except (RPCError, ConnectionError, ValueError) as exc:
        action_error = str(exc)
        if "Invalid card index" in action_error and _try_add_target_cards(args.base_url, args.target, args.timeout_sec):
            retried_with_add = True
            pre_state = get_state(args.base_url, timeout=args.timeout_sec)
            try:
                _apply_action(args.base_url, action0, timeout_sec=args.timeout_sec, wait_sleep=args.wait_sleep)
                action_error = None
            except (RPCError, ConnectionError, ValueError) as exc2:
                action_error = str(exc2)

    post_state = get_state(args.base_url, timeout=args.timeout_sec)

    pre_flat = _flatten_numeric(pre_state)
    post_flat = _flatten_numeric(post_state)
    keys = sorted(set(pre_flat.keys()) | set(post_flat.keys()))

    changed_numeric: list[dict[str, Any]] = []
    for k in keys:
        a = pre_flat.get(k)
        b = post_flat.get(k)
        if a is None or b is None:
            continue
        if abs(a - b) > 1e-9:
            changed_numeric.append({"path": k, "pre": a, "post": b, "delta": b - a})

    probe_fields = [
        "$.score.last_base_chips",
        "$.score.last_base_mult",
        "$.score.last_hand_type",
        "$.round.last_base_chips",
        "$.round.last_base_mult",
        "$.round.last_hand_type",
        "$.round.hand_type",
    ]
    field_probe: dict[str, Any] = {}
    for f in probe_fields:
        pre_v, pre_ok = _get_by_path(pre_state, f)
        post_v, post_ok = _get_by_path(post_state, f)
        field_probe[f] = {
            "pre_exists": pre_ok,
            "post_exists": post_ok,
            "pre": pre_v,
            "post": post_v,
            "changed": (pre_ok and post_ok and pre_v != post_v),
        }

    observable_candidates = [
        "$.round.chips",
        "$.score.chips",
        "$.round.mult",
        "$.score.mult",
        "$.round.hands_left",
        "$.round.discards_left",
        "$.round.hands_played",
        "$.round.discards_used",
    ]
    observed_summary: list[dict[str, Any]] = []
    for path in observable_candidates:
        pre_v, pre_ok = _get_by_path(pre_state, path)
        post_v, post_ok = _get_by_path(post_state, path)
        if pre_ok and post_ok and isinstance(pre_v, (int, float)) and isinstance(post_v, (int, float)):
            observed_summary.append(
                {
                    "path": path,
                    "pre": float(pre_v),
                    "post": float(post_v),
                    "delta": float(post_v) - float(pre_v),
                    "changed": float(post_v) != float(pre_v),
                }
            )

    canonical_pre = canonicalize_real_state(pre_state, seed="probe", rng_events=[], rng_cursor=0)
    canonical_post = canonicalize_real_state(post_state, seed="probe", rng_events=[], rng_cursor=0)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "base_url": args.base_url,
        "action_trace": str(action_trace_path),
        "start_state": str(start_state_path) if start_state_path else None,
        "action0": action0,
        "action_error": action_error,
        "retried_with_add": retried_with_add,
        "last_base_probe": field_probe,
        "observable_candidates": observed_summary,
        "changed_numeric_fields": changed_numeric,
        "canonical_pre_score": canonical_pre.get("score"),
        "canonical_post_score": canonical_post.get("score"),
    }

    out_path = Path(args.out) if args.out else Path("sim/tests/fixtures_runtime/probes") / args.target / "probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"probe target={args.target}")
    print(f"action_error={action_error}")
    for k in ("$.score.last_base_chips", "$.score.last_base_mult", "$.score.last_hand_type"):
        item = field_probe.get(k, {})
        print(
            f"{k}: pre_exists={item.get('pre_exists')} post_exists={item.get('post_exists')} "
            f"pre={item.get('pre')} post={item.get('post')} changed={item.get('changed')}"
        )

    print("observable_candidates:")
    for item in observed_summary:
        print(
            f"  {item['path']}: pre={item['pre']} post={item['post']} "
            f"delta={item['delta']} changed={item['changed']}"
        )

    print(f"wrote probe: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
