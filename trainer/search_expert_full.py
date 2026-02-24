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
from trainer import action_space_shop
from trainer.search_expert import SearchDecision, choose_action as choose_hand_action


@dataclass
class FullDecision:
    phase: str
    action: dict[str, Any]
    action_id: int | None
    value: float
    considered: int
    reason: str


def _state_value(state: dict[str, Any]) -> float:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    chips = float(round_info.get("chips") or 0.0)
    hands_left = float(round_info.get("hands_left") or 0.0)
    discards_left = float(round_info.get("discards_left") or 0.0)
    money = float(state.get("money") or 0.0)
    return chips + 1.5 * hands_left + 0.8 * discards_left + 0.2 * money


def _simulate_action(state: dict[str, Any], action: dict[str, Any], seed: str) -> tuple[dict[str, Any], float, bool]:
    canonical = to_canonical_state(state, seed=seed)
    backend = SimEnvBackend(seed=seed)
    backend.reset(from_snapshot=canonical)
    next_state, reward, done, _ = backend.step(action)
    return next_state, float(reward), bool(done)


def _shop_search(state: dict[str, Any], *, max_actions: int, time_budget_ms: float, seed: str) -> FullDecision:
    legal_ids = action_space_shop.legal_action_ids(state)
    if not legal_ids:
        action = {"action_type": "WAIT"}
        return FullDecision(phase=str(state.get("state") or "UNKNOWN"), action=action, action_id=None, value=0.0, considered=0, reason="no_legal_shop_actions")

    candidates = legal_ids[: max(1, int(max_actions))]
    t0 = time.perf_counter()
    budget = max(0.001, float(time_budget_ms) / 1000.0)
    best_val = float("-inf")
    best_id = candidates[0]
    considered = 0

    for aid in candidates:
        if (time.perf_counter() - t0) > budget:
            break
        considered += 1
        action = action_space_shop.action_from_id(state, aid)
        try:
            next_state, reward, done = _simulate_action(state, action, seed=seed)
        except Exception:
            continue
        value = float(reward) + 0.15 * _state_value(next_state) + (0.0 if done else 1.0)
        if value > best_val:
            best_val = value
            best_id = aid

    best_action = action_space_shop.action_from_id(state, best_id)
    return FullDecision(
        phase=str(state.get("state") or "UNKNOWN"),
        action=best_action,
        action_id=int(best_id),
        value=float(best_val if best_val != float("-inf") else 0.0),
        considered=int(considered),
        reason="shop_search_selected",
    )


def choose_full_action(
    state: dict[str, Any],
    *,
    hand_max_branch: int = 80,
    hand_max_depth: int = 2,
    shop_max_actions: int = 24,
    time_budget_ms: float = 20.0,
    seed: str = "SEARCH_FULL",
) -> FullDecision:
    phase = str(state.get("state") or "UNKNOWN")
    if phase == "SELECTING_HAND":
        hand: SearchDecision = choose_hand_action(
            state,
            max_branch=int(hand_max_branch),
            max_depth=int(hand_max_depth),
            time_budget_ms=float(time_budget_ms),
            seed=seed,
        )
        action = {"action_type": hand.action_type, "indices": list(hand.indices)}
        return FullDecision(
            phase=phase,
            action=action,
            action_id=None,
            value=float(hand.value),
            considered=int(hand.considered),
            reason=hand.reason,
        )

    if phase in action_space_shop.SHOP_PHASES:
        return _shop_search(
            state,
            max_actions=int(shop_max_actions),
            time_budget_ms=float(time_budget_ms),
            seed=seed,
        )

    return FullDecision(
        phase=phase,
        action={"action_type": "WAIT"},
        action_id=None,
        value=0.0,
        considered=0,
        reason="non_hand_shop_wait",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Search expert helper for hand+shop decisions.")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--hand-max-branch", type=int, default=80)
    parser.add_argument("--hand-max-depth", type=int, default=2)
    parser.add_argument("--shop-max-actions", type=int, default=24)
    parser.add_argument("--time-budget-ms", type=float, default=20.0)
    parser.add_argument("--seed", default="SEARCH_FULL")
    args = parser.parse_args()

    with open(args.state_json, "r", encoding="utf-8-sig") as fp:
        state = json.load(fp)
    decision = choose_full_action(
        state,
        hand_max_branch=args.hand_max_branch,
        hand_max_depth=args.hand_max_depth,
        shop_max_actions=args.shop_max_actions,
        time_budget_ms=args.time_budget_ms,
        seed=args.seed,
    )
    print(json.dumps(decision.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

