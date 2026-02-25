from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import time
from dataclasses import dataclass
from typing import Any

from sim.core.canonical import to_canonical_state
from sim.pybind.sim_env import SimEnvBackend
from trainer import action_space_shop
from trainer.candidates_hand import generate_hand_candidates
from trainer.candidates_shop import generate_shop_candidates


@dataclass
class BeamAction:
    action: dict[str, Any]
    value: float
    score_breakdown: dict[str, float]


@dataclass
class BeamDecision:
    phase: str
    topk: list[BeamAction]
    considered: int
    reason: str

    @property
    def top1(self) -> dict[str, Any]:
        if self.topk:
            return dict(self.topk[0].action)
        return {"action_type": "WAIT", "sleep": 0.01}


def _state_value(state: dict[str, Any]) -> float:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    chips = float(round_info.get("chips") or 0.0)
    hands_left = float(round_info.get("hands_left") or 0.0)
    discards_left = float(round_info.get("discards_left") or 0.0)
    money = float(state.get("money") or 0.0)
    done_penalty = -80.0 if str(state.get("state") or "") == "GAME_OVER" else 0.0
    return chips + 1.5 * hands_left + 0.9 * discards_left + 0.35 * money + done_penalty


def _simulate(state: dict[str, Any], action: dict[str, Any], seed: str) -> tuple[dict[str, Any], float, bool]:
    canonical = to_canonical_state(state, seed=seed)
    backend = SimEnvBackend(seed=seed)
    backend.reset(from_snapshot=canonical)
    next_state, reward, done, _ = backend.step(action)
    return next_state, float(reward), bool(done)


def _legal_phase(phase: str) -> str:
    if phase == "SELECTING_HAND":
        return "HAND"
    if phase in action_space_shop.SHOP_PHASES:
        return "SHOP"
    return "OTHER"


def choose_topk(
    state: dict[str, Any],
    *,
    topk: int = 3,
    hand_max_candidates: int = 40,
    shop_max_candidates: int = 20,
    max_depth: int = 2,
    time_budget_ms: float = 20.0,
    seed: str = "P15-BEAM",
    fail_fast: bool = True,
) -> BeamDecision:
    phase = str(state.get("state") or "UNKNOWN")
    phase_kind = _legal_phase(phase)

    if phase_kind == "HAND":
        candidates = generate_hand_candidates(state, max_candidates=hand_max_candidates)
    elif phase_kind == "SHOP":
        candidates = generate_shop_candidates(state, max_candidates=shop_max_candidates)
    else:
        return BeamDecision(phase=phase, topk=[BeamAction({"action_type": "WAIT", "sleep": 0.01}, 0.0, {"fallback": 0.0})], considered=0, reason="non_hand_shop")

    t0 = time.perf_counter()
    budget = max(0.001, float(time_budget_ms) / 1000.0)
    scored: list[BeamAction] = []
    considered = 0

    for idx, action in enumerate(candidates):
        if (time.perf_counter() - t0) > budget:
            break
        considered += 1
        try:
            next_state, reward, done = _simulate(state, action, seed=f"{seed}-{idx}")
        except Exception:
            if fail_fast:
                raise RuntimeError(f"beam simulate failed for action={action!r} phase={phase}")
            continue

        immediate = float(reward)
        proxy = 0.08 * _state_value(next_state)
        terminal = -100.0 if done and str(next_state.get("state") or "") == "GAME_OVER" else (8.0 if done else 0.0)
        depth_bonus = 0.0

        if (not done) and max_depth > 1 and (time.perf_counter() - t0) <= budget:
            next_phase = str(next_state.get("state") or "UNKNOWN")
            if _legal_phase(next_phase) in {"HAND", "SHOP"}:
                if _legal_phase(next_phase) == "HAND":
                    next_candidates = generate_hand_candidates(next_state, max_candidates=min(8, hand_max_candidates))
                else:
                    next_candidates = generate_shop_candidates(next_state, max_candidates=min(8, shop_max_candidates))
                best_tail = float("-inf")
                for j, nxt in enumerate(next_candidates):
                    if (time.perf_counter() - t0) > budget:
                        break
                    try:
                        tail_state, tail_reward, tail_done = _simulate(next_state, nxt, seed=f"{seed}-{idx}-d{j}")
                    except Exception:
                        if fail_fast:
                            raise RuntimeError(f"beam tail simulate failed for action={nxt!r} phase={next_phase}")
                        continue
                    tail_val = float(tail_reward) + 0.05 * _state_value(tail_state) + (-30.0 if tail_done and str(tail_state.get("state") or "") == "GAME_OVER" else 0.0)
                    if tail_val > best_tail:
                        best_tail = tail_val
                if best_tail != float("-inf"):
                    depth_bonus = 0.6 * best_tail

        value = immediate + proxy + terminal + depth_bonus
        breakdown = {
            "immediate_score_delta": immediate,
            "state_proxy": proxy,
            "terminal_term": terminal,
            "depth_bonus": depth_bonus,
            "total": value,
        }
        scored.append(BeamAction(action=dict(action), value=float(value), score_breakdown=breakdown))

    if not scored:
        return BeamDecision(phase=phase, topk=[BeamAction({"action_type": "WAIT", "sleep": 0.01}, 0.0, {"fallback": 0.0})], considered=considered, reason="no_candidate_scored")

    scored.sort(key=lambda x: x.value, reverse=True)
    return BeamDecision(
        phase=phase,
        topk=scored[: max(1, int(topk))],
        considered=int(considered),
        reason="beam_selected",
    )
