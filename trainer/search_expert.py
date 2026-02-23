if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.pybind.sim_env import SimEnvBackend
from trainer import action_space


@dataclass
class SearchDecision:
    action_type: str
    indices: list[int]
    value: float
    considered: int
    depth_used: int
    reason: str


def _legal_hand_actions(hand_size: int) -> list[tuple[str, list[int]]]:
    legal_ids = action_space.legal_action_ids(hand_size)
    out: list[tuple[str, list[int]]] = []
    for aid in legal_ids:
        atype, mask_int = action_space.decode(hand_size, int(aid))
        indices = action_space.mask_to_indices(mask_int, hand_size)
        out.append((atype, indices))
    return out


def _state_value(state: dict[str, Any]) -> float:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    chips = float(round_info.get("chips") or 0.0)
    hands_left = float(round_info.get("hands_left") or 0.0)
    discards_left = float(round_info.get("discards_left") or 0.0)
    money = float(state.get("money") or 0.0)
    return chips + 1.5 * hands_left + 0.8 * discards_left + 0.05 * money


def _simulate_action(state: dict[str, Any], action: dict[str, Any], seed: str) -> tuple[dict[str, Any], float, bool]:
    canonical = to_canonical_state(state, seed=seed)
    backend = SimEnvBackend(seed=seed)
    backend.reset(from_snapshot=canonical)
    next_state, reward, done, _ = backend.step(action)
    return next_state, float(reward), bool(done)


def _is_selecting_hand(state: dict[str, Any]) -> bool:
    return str(state.get("state") or "") == "SELECTING_HAND"


def _search_value(
    state: dict[str, Any],
    *,
    depth: int,
    max_depth: int,
    max_branch: int,
    seed: str,
    t0: float,
    time_budget_sec: float,
) -> tuple[float, int]:
    if depth >= max_depth:
        return _state_value(state), 0
    if not _is_selecting_hand(state):
        return _state_value(state), 0
    if (time.perf_counter() - t0) > time_budget_sec:
        return _state_value(state), 0

    hand_cards = (state.get("hand") or {}).get("cards") or []
    hand_size = min(len(hand_cards), action_space.MAX_HAND)
    if hand_size <= 0:
        return _state_value(state), 0

    candidates = _legal_hand_actions(hand_size)
    if max_branch > 0:
        candidates = candidates[:max_branch]

    best = float("-inf")
    considered = 0
    for atype, indices in candidates:
        if (time.perf_counter() - t0) > time_budget_sec:
            break
        considered += 1
        action = {"action_type": atype, "indices": indices}
        try:
            next_state, reward, done = _simulate_action(state, action, seed=seed)
        except Exception:
            continue
        cur = float(reward) + _state_value(next_state) * 0.05
        if not done:
            tail, tail_considered = _search_value(
                next_state,
                depth=depth + 1,
                max_depth=max_depth,
                max_branch=max_branch,
                seed=seed,
                t0=t0,
                time_budget_sec=time_budget_sec,
            )
            considered += tail_considered
            cur += 0.9 * tail
        if cur > best:
            best = cur

    if best == float("-inf"):
        return _state_value(state), considered
    return best, considered


def choose_action(
    state: dict[str, Any],
    *,
    max_branch: int = 80,
    max_depth: int = 2,
    time_budget_ms: float = 15.0,
    seed: str = "SEARCH",
) -> SearchDecision:
    hand_cards = (state.get("hand") or {}).get("cards") or []
    hand_size = min(len(hand_cards), action_space.MAX_HAND)
    if hand_size <= 0 or not _is_selecting_hand(state):
        return SearchDecision("PLAY", [0], value=0.0, considered=0, depth_used=0, reason="fallback_non_hand_phase")

    candidates = _legal_hand_actions(hand_size)
    if max_branch > 0:
        candidates = candidates[:max_branch]

    t0 = time.perf_counter()
    budget_sec = max(0.001, float(time_budget_ms) / 1000.0)
    best_value = float("-inf")
    best_action: tuple[str, list[int]] | None = None
    considered_total = 0

    for atype, indices in candidates:
        if (time.perf_counter() - t0) > budget_sec:
            break
        considered_total += 1
        action = {"action_type": atype, "indices": indices}
        try:
            next_state, reward, done = _simulate_action(state, action, seed=seed)
        except Exception:
            continue

        value = float(reward) + _state_value(next_state) * 0.05
        if (not done) and max_depth > 1:
            tail, tail_considered = _search_value(
                next_state,
                depth=1,
                max_depth=max_depth,
                max_branch=max_branch,
                seed=seed,
                t0=t0,
                time_budget_sec=budget_sec,
            )
            considered_total += tail_considered
            value += 0.9 * tail

        if value > best_value:
            best_value = value
            best_action = (atype, indices)

    if best_action is None:
        return SearchDecision("PLAY", [0], value=0.0, considered=considered_total, depth_used=0, reason="fallback_no_candidate")

    return SearchDecision(
        action_type=best_action[0],
        indices=list(best_action[1]),
        value=float(best_value),
        considered=int(considered_total),
        depth_used=int(max_depth),
        reason="search_selected",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Search expert debug helper for SELECTING_HAND state json.")
    parser.add_argument("--state-json", required=True, help="Path to a JSON file containing one gamestate object.")
    parser.add_argument("--max-branch", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--time-budget-ms", type=float, default=15.0)
    parser.add_argument("--seed", default="SEARCH")
    args = parser.parse_args()

    with open(args.state_json, "r", encoding="utf-8-sig") as fp:
        state = json.load(fp)
    decision = choose_action(
        state,
        max_branch=args.max_branch,
        max_depth=args.max_depth,
        time_budget_ms=args.time_budget_ms,
        seed=args.seed,
    )
    print(json.dumps(decision.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
