if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Any

SCOPE_TO_HASH_KEY = {
    "hand_core": "state_hash_hand_core",
    "score_core": "state_hash_score_core",
    "zones_core": "state_hash_zones_core",
    "full": "state_hash_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff oracle and simulator traces by canonical hash.")
    parser.add_argument("--oracle-trace", required=True)
    parser.add_argument("--sim-trace", required=True)
    parser.add_argument("--scope", choices=["hand_core", "score_core", "zones_core", "full"], default="hand_core")
    parser.add_argument("--fail-fast", action="store_true", default=True)
    return parser.parse_args()


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
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


def main() -> int:
    args = parse_args()

    oracle = load_trace(args.oracle_trace)
    sim = load_trace(args.sim_trace)

    if len(oracle) != len(sim):
        print(f"TRACE LENGTH MISMATCH: oracle={len(oracle)} sim={len(sim)}")

    n = min(len(oracle), len(sim))
    hash_key = SCOPE_TO_HASH_KEY[args.scope]

    for i in range(n):
        o = oracle[i]
        s = sim[i]
        ho = o.get(hash_key)
        hs = s.get(hash_key)
        if ho != hs:
            print(f"MISMATCH step={i} scope={args.scope}")
            print(f"oracle_hash={ho}")
            print(f"sim_hash={hs}")

            o_snap = o.get("canonical_state_snapshot")
            s_snap = s.get("canonical_state_snapshot")
            if isinstance(o_snap, dict) and isinstance(s_snap, dict):
                diff = _first_diff_path(o_snap, s_snap)
                if diff is not None:
                    path, ov, sv = diff
                    print(f"first_diff_path={path}")
                    print(f"oracle_value={ov}")
                    print(f"sim_value={sv}")
                else:
                    print("snapshots present but no structural diff detected")
            else:
                print("No canonical_state_snapshot on mismatch step. Re-run traces with snapshot_every=1 for full diff path.")
            return 1

    if len(oracle) != len(sim):
        return 1

    print(f"OK: traces matched for {n} steps in scope={args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
