from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ACTION_ALIASES: dict[str, str] = {
    "REROLL": "SHOP_REROLL",
    "BUY": "SHOP_BUY",
    "PACK": "PACK_OPEN",
    "USE": "CONSUMABLE_USE",
    "SWAP_HAND_CARDS": "SWAP_HAND_CARD",
    "SWAP_JOKERS": "SWAP_JOKER",
}


def phase_from_obs(obs: dict[str, Any] | None) -> str:
    if not isinstance(obs, dict):
        return "UNKNOWN"
    return str(obs.get("state") or obs.get("phase") or "UNKNOWN").upper()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_indices(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        out.append(_safe_int(item, 0))
    return out


def normalize_action(
    action: dict[str, Any] | None,
    *,
    phase: str = "UNKNOWN",
    index_base: int = 0,
) -> dict[str, Any]:
    raw = dict(action or {})
    raw_type = str(raw.get("action_type") or "WAIT").upper()
    action_type = ACTION_ALIASES.get(raw_type, raw_type)

    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    merged_params = dict(params)
    merged_params.setdefault("index_base", 1 if int(index_base) == 1 else 0)

    if "index" in raw:
        raw["index"] = _safe_int(raw.get("index"), 0)
    if "indices" in raw:
        raw["indices"] = _safe_indices(raw.get("indices"))

    normalized: dict[str, Any] = {
        "schema_version": "action_v1",
        "phase": str(raw.get("phase") or phase or "UNKNOWN").upper(),
        "action_type": action_type,
        "params": merged_params,
    }

    passthrough = (
        "index",
        "indices",
        "src_index",
        "dst_index",
        "i",
        "j",
        "seed",
        "stake",
        "deck",
        "sleep",
        "permutation",
    )
    for key in passthrough:
        if key in raw:
            normalized[key] = raw[key]

    return normalized


def phase_default_action(obs: dict[str, Any], *, seed: str = "AAAAAAA") -> dict[str, Any]:
    phase = phase_from_obs(obs)
    if phase == "BLIND_SELECT":
        return normalize_action({"action_type": "SELECT", "index": 0}, phase=phase)
    if phase == "SELECTING_HAND":
        hand_cards = (obs.get("hand") or {}).get("cards") if isinstance(obs.get("hand"), dict) else []
        round_info = obs.get("round") if isinstance(obs.get("round"), dict) else {}
        hands_left = _safe_int(round_info.get("hands_left"), 0)
        discards_left = _safe_int(round_info.get("discards_left"), 0)
        if isinstance(hand_cards, list) and hand_cards and hands_left > 0:
            return normalize_action({"action_type": "PLAY", "indices": [0]}, phase=phase)
        if isinstance(hand_cards, list) and hand_cards and discards_left > 0:
            return normalize_action({"action_type": "DISCARD", "indices": [0]}, phase=phase)
        return normalize_action({"action_type": "WAIT"}, phase=phase)
    if phase == "ROUND_EVAL":
        return normalize_action({"action_type": "CASH_OUT"}, phase=phase)
    if phase == "SHOP":
        return normalize_action({"action_type": "NEXT_ROUND"}, phase=phase)
    if phase in {"MENU", "GAME_OVER"}:
        return normalize_action(
            {"action_type": "START", "seed": seed, "stake": "WHITE", "deck": "RED"},
            phase=phase,
        )
    return normalize_action({"action_type": "WAIT"}, phase=phase)


@dataclass
class AdapterDescriptor:
    name: str
    family: str
    status: str = "active"
    supports_batch: bool = False
    supports_shop: bool = True
    supports_consumables: bool = True
    supports_position_actions: bool = False
    notes: str = ""


class BasePolicyAdapter:
    def __init__(self, *, descriptor: AdapterDescriptor):
        self.name = descriptor.name
        self._descriptor = descriptor
        self._seed = "AAAAAAA"

    def describe(self) -> dict[str, Any]:
        return {
            "schema": "p39_policy_adapter_descriptor_v1",
            "adapter": asdict(self._descriptor),
        }

    def reset(self, seed: str | int | None = None) -> None:
        if seed is None:
            return
        self._seed = str(seed)

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        raise NotImplementedError("BasePolicyAdapter.act must be implemented by subclass")

    def act_batch(
        self,
        obs_batch: list[dict[str, Any]],
        legal_actions_batch: list[list[dict[str, Any]] | None] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, obs in enumerate(obs_batch):
            legal_actions = None
            if isinstance(legal_actions_batch, list) and idx < len(legal_actions_batch):
                legal_actions = legal_actions_batch[idx]
            out.append(self.act(obs, legal_actions=legal_actions))
        return out

    def close(self) -> None:
        return None
