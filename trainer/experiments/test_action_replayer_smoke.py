from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.actions.replay import ActionReplayer, normalize_high_level_action
from trainer.env_client import create_backend


def _state_projection(state: dict[str, Any]) -> dict[str, Any]:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
    jokers = state.get("jokers") if isinstance(state.get("jokers"), list) else []
    return {
        "phase": str(state.get("state") or "UNKNOWN"),
        "round": {
            "chips": float(round_info.get("chips") or 0.0),
            "hands_left": int(round_info.get("hands_left") or 0),
            "discards_left": int(round_info.get("discards_left") or 0),
        },
        "hand_keys": [str((c or {}).get("key") or "") for c in hand_cards],
        "jokers": [str((j or {}).get("joker_id") or (j or {}).get("key") or "") if isinstance(j, dict) else str(j) for j in jokers],
        "money": float(state.get("money") or 0.0),
    }


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _find_latest_position_fixture(repo_root: Path) -> Path | None:
    p32_root = repo_root / "docs" / "artifacts" / "p32"
    if not p32_root.exists():
        return None
    candidates = sorted(
        [
            p
            for p in p32_root.glob("*/fixtures/position_contract")
            if (p / "manifest.json").exists() and (p / "action_trace_real.jsonl").exists()
        ],
        key=lambda p: str(p),
    )
    return candidates[-1] if candidates else None


def _advance_to_phase(backend, target_phase: str, *, reset_seed: str, max_steps: int = 80) -> dict[str, Any]:
    state = backend.get_state()
    if str(state.get("state") or "").upper() == target_phase.upper():
        return state
    for _ in range(max_steps):
        state, _, done, _ = backend.step({"action_type": "AUTO"})
        if str(state.get("state") or "").upper() == target_phase.upper():
            return state
        if done:
            state = backend.reset(seed=reset_seed)
    raise RuntimeError(f"unable to reach target phase={target_phase} within {max_steps} steps")


def _build_fixture(repo_root: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-B",
        "sim/tests/build_p32_position_fixture.py",
        "--out-dir",
        str(out_dir),
        "--seed",
        "P33SMOKE",
    ]
    result = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"failed to build fallback fixture: rc={result.returncode} stderr={result.stderr[-2000:]}")
    return out_dir


def _is_action_compatible(state: dict[str, Any], action: dict[str, Any]) -> bool:
    action_type = str(action.get("action_type") or "").upper()
    hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
    jokers = state.get("jokers") if isinstance(state.get("jokers"), list) else []
    hand_n = len(hand_cards)
    joker_n = len(jokers)
    if action_type in {"REORDER_HAND", "PLAY", "DISCARD"}:
        if action_type == "REORDER_HAND":
            return len(action.get("permutation") or []) == hand_n and hand_n > 0
        idxs = action.get("indices") or []
        return bool(idxs) and all(isinstance(x, int) and 0 <= x < hand_n for x in idxs)
    if action_type == "SWAP_HAND_CARD":
        i = int(action.get("i", -1))
        j = int(action.get("j", -1))
        return hand_n > 1 and 0 <= i < hand_n and 0 <= j < hand_n and i != j
    if action_type == "REORDER_JOKERS":
        return len(action.get("permutation") or []) == joker_n and joker_n > 0
    if action_type == "SWAP_JOKER":
        i = int(action.get("i", -1))
        j = int(action.get("j", -1))
        return joker_n > 1 and 0 <= i < joker_n and 0 <= j < joker_n and i != j
    if action_type == "MOVE_HAND_CARD":
        src = int(action.get("src_index", -1))
        dst = int(action.get("dst_index", -1))
        return hand_n > 0 and 0 <= src < hand_n and 0 <= dst < hand_n
    if action_type == "MOVE_JOKER":
        src = int(action.get("src_index", -1))
        dst = int(action.get("dst_index", -1))
        return joker_n > 0 and 0 <= src < joker_n and 0 <= dst < joker_n
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="P33 smoke test for ActionReplayer(sim).")
    parser.add_argument("--fixture-dir", default="", help="Optional fixture dir containing manifest.json + action_trace_real.jsonl")
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--out", default="docs/artifacts/p33/action_replayer_smoke_summary.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    fixture_dir = Path(args.fixture_dir) if str(args.fixture_dir).strip() else None
    if fixture_dir is None:
        fixture_dir = _find_latest_position_fixture(repo_root)
    if fixture_dir is None or not fixture_dir.exists():
        fixture_dir = _build_fixture(repo_root, repo_root / "docs" / "artifacts" / "p33" / "smoke_fixture")

    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = str(manifest.get("seed") or "AAAAAAA")
    actions = _read_jsonl(fixture_dir / "action_trace_real.jsonl")
    if not actions:
        raise RuntimeError(f"empty action trace: {fixture_dir}")
    actions = actions[: max(1, int(args.max_actions))]

    backend_a = create_backend("sim", seed=seed)
    backend_b = create_backend("sim", seed=seed)
    replayer = ActionReplayer(mode="sim", backend=backend_a, strict=True, debug_dir=repo_root / "docs" / "artifacts" / "p33" / "logs")
    mismatches: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    skipped_actions = 0
    try:
        state_a = backend_a.reset(seed=seed)
        state_b = backend_b.reset(seed=seed)
        # Position fixture actions start from SELECTING_HAND context.
        state_a = _advance_to_phase(backend_a, "SELECTING_HAND", reset_seed=seed)
        state_b = _advance_to_phase(backend_b, "SELECTING_HAND", reset_seed=seed)
        for idx, action in enumerate(actions):
            normalized = normalize_high_level_action(action, phase=str(state_a.get("state") or "UNKNOWN"))
            if not _is_action_compatible(state_a, normalized):
                skipped_actions += 1
                continue
            res = replayer.replay_single_action(state_a, normalized)
            if not res.ok:
                raise RuntimeError(f"replay failed at step={idx}: {res.error}")
            after_a = backend_a.get_state()
            after_b, reward_b, done_b, info_b = backend_b.step(normalized)

            hash_a = _stable_hash(_state_projection(after_a))
            hash_b = _stable_hash(_state_projection(after_b))
            reward_a = float(res.reward)
            done_a = bool(res.done)
            mismatch = not (hash_a == hash_b and abs(reward_a - float(reward_b)) < 1e-9 and done_a == bool(done_b))
            executed.append(
                {
                    "step": idx,
                    "action_type": str(normalized.get("action_type") or ""),
                    "hash_a": hash_a,
                    "hash_b": hash_b,
                    "reward_a": reward_a,
                    "reward_b": float(reward_b),
                    "done_a": done_a,
                    "done_b": bool(done_b),
                    "info_b": info_b,
                }
            )
            if mismatch:
                mismatches.append(executed[-1])
            state_a = after_a
            state_b = after_b
    finally:
        replayer.close()
        backend_a.close()
        backend_b.close()

    summary = {
        "schema": "p33_action_replayer_smoke_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_dir": str(fixture_dir),
        "seed": seed,
        "max_actions": int(args.max_actions),
        "actions_executed": len(executed),
        "actions_skipped_incompatible": int(skipped_actions),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "pass": len(mismatches) == 0 and len(executed) > 0,
    }
    out_path = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
