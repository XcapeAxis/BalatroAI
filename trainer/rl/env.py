from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer import action_space
from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.features_shop import extract_shop_features

PHASE_TO_ID: dict[str, int] = {
    "BLIND_SELECT": 0,
    "SELECTING_HAND": 1,
    "ROUND_EVAL": 2,
    "SHOP": 3,
    "SMODS_BOOSTER_OPENED": 4,
    "MENU": 5,
    "GAME_OVER": 6,
    "OTHER": 7,
}
CONTROLLED_PHASES = {"SELECTING_HAND", "SHOP", "SMODS_BOOSTER_OPENED"}
DEFAULT_OBS_DIM = 48


@dataclass
class BalatroEnvConfig:
    backend: str = "sim"
    seed: str = "AAAAAAA"
    stake: str = "WHITE"
    timeout_sec: float = 8.0
    reward_mode: str = "score_delta"  # "score_delta" | "episode_total_score"
    max_steps_per_episode: int = 320
    max_auto_steps: int = 8
    max_ante: int = 0
    auto_advance: bool = True


class BalatroEnv:
    """Gym-like environment wrapper over existing sim/real backends."""

    def __init__(
        self,
        *,
        backend: str = "sim",
        seed: str = "AAAAAAA",
        stake: str = "WHITE",
        timeout_sec: float = 8.0,
        reward_mode: str = "score_delta",
        max_steps_per_episode: int = 320,
        max_auto_steps: int = 8,
        max_ante: int = 0,
        auto_advance: bool = True,
    ) -> None:
        self.config = BalatroEnvConfig(
            backend=backend,
            seed=seed,
            stake=stake,
            timeout_sec=float(timeout_sec),
            reward_mode=str(reward_mode or "score_delta"),
            max_steps_per_episode=max(1, int(max_steps_per_episode)),
            max_auto_steps=max(1, int(max_auto_steps)),
            max_ante=max(0, int(max_ante)),
            auto_advance=bool(auto_advance),
        )
        self._backend = create_backend(
            self.config.backend,
            timeout_sec=self.config.timeout_sec,
            seed=self.config.seed,
        )
        self.action_dim = max(action_space.max_actions(), action_space_shop.max_actions())
        self.obs_dim = DEFAULT_OBS_DIM
        self._seed = self.config.seed
        self._state: dict[str, Any] | None = None
        self._done = False
        self._episode_steps = 0
        self._episode_return = 0.0

    def close(self) -> None:
        self._backend.close()

    def reset(self, seed: str | None = None) -> dict[str, Any]:
        self._seed = str(seed or self._seed or self.config.seed)
        self._done = False
        self._episode_steps = 0
        self._episode_return = 0.0
        state = self._backend.reset(seed=self._seed)
        phase = str(state.get("state") or "")
        if phase in {"MENU", "GAME_OVER"}:
            state, _, _, _ = self._backend.step(
                {"action_type": "START", "seed": self._seed, "stake": self.config.stake}
            )
        self._state = state
        if self.config.auto_advance:
            self._auto_advance_until_controlled()
        return self._build_observation(self._state)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("episode already done; call reset() before step()")

        state_before = self._state
        phase_before = str(state_before.get("state") or "OTHER")
        score_before = self._score(state_before)
        action_payload, applied_action_id, invalid_action, legal_ids = self._decode_action(state_before, action)
        next_state, _, done_flag, backend_info = self._backend.step(action_payload)
        self._episode_steps += 1

        score_after = self._score(next_state)
        reward = self._reward_from_scores(score_before, score_after)
        auto_reward = 0.0
        auto_steps = 0

        self._state = next_state
        if self.config.auto_advance and not done_flag and not self._is_done_state(self._state):
            auto_reward, auto_steps = self._auto_advance_until_controlled()

        total_reward = float(reward + auto_reward)
        self._episode_return += total_reward

        done = bool(done_flag or self._is_done_state(self._state))
        truncated = False
        if not done and self._episode_steps >= self.config.max_steps_per_episode:
            done = True
            truncated = True
        if (
            not done
            and self.config.max_ante > 0
            and int(self._state.get("ante_num") or 0) >= self.config.max_ante
            and str(self._state.get("state") or "") == "BLIND_SELECT"
        ):
            done = True
            truncated = True

        self._done = done
        info = {
            "phase_before": phase_before,
            "phase_after": str(self._state.get("state") or "OTHER"),
            "action_payload": action_payload,
            "applied_action_id": applied_action_id,
            "invalid_action": bool(invalid_action),
            "legal_action_ids": legal_ids,
            "score_before": score_before,
            "score_after": self._score(self._state),
            "score_delta": self._score(self._state) - score_before,
            "episode_return": float(self._episode_return),
            "episode_length": int(self._episode_steps),
            "auto_steps": int(auto_steps),
            "truncated": bool(truncated),
            "backend_info": backend_info if isinstance(backend_info, dict) else {},
        }
        return self._build_observation(self._state), total_reward, done, info

    def render(self) -> str:
        if self._state is None:
            return "BalatroEnv(state=uninitialized)"
        phase = str(self._state.get("state") or "OTHER")
        chips = self._score(self._state)
        money = float(self._state.get("money") or 0.0)
        ante = int(self._state.get("ante_num") or 0)
        return (
            f"BalatroEnv(phase={phase}, chips={chips:.2f}, money={money:.2f}, "
            f"ante={ante}, step={self._episode_steps}, return={self._episode_return:.2f})"
        )

    def _is_done_state(self, state: dict[str, Any]) -> bool:
        phase = str(state.get("state") or "")
        return bool(phase == "GAME_OVER" or state.get("done"))

    @staticmethod
    def _score(state: dict[str, Any]) -> float:
        round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
        return float(round_info.get("chips") or 0.0)

    def _reward_from_scores(self, before: float, after: float) -> float:
        mode = str(self.config.reward_mode or "score_delta").strip().lower()
        if mode == "episode_total_score":
            return float(after)
        return float(after - before)

    def _phase_default_action(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = str(state.get("state") or "OTHER")
        if phase == "BLIND_SELECT":
            idx = 0
            blinds = state.get("blinds") if isinstance(state.get("blinds"), dict) else {}
            for probe_idx, key in enumerate(("small", "big", "boss")):
                info = blinds.get(key) if isinstance(blinds.get(key), dict) else {}
                if str(info.get("status") or "").upper() == "SELECT":
                    idx = probe_idx
                    break
            return {"action_type": "SELECT", "index": idx}
        if phase == "ROUND_EVAL":
            return {"action_type": "CASH_OUT"}
        if phase == "SHOP":
            return {"action_type": "NEXT_ROUND"}
        if phase == "SMODS_BOOSTER_OPENED":
            return {"action_type": "PACK", "params": {"skip": True}}
        if phase in {"MENU", "GAME_OVER"}:
            return {"action_type": "START", "seed": self._seed, "stake": self.config.stake}
        return {"action_type": "WAIT"}

    def _auto_advance_until_controlled(self) -> tuple[float, int]:
        if self._state is None:
            return 0.0, 0
        total_reward = 0.0
        steps = 0
        while steps < self.config.max_auto_steps:
            phase = str(self._state.get("state") or "OTHER")
            if phase in CONTROLLED_PHASES or self._is_done_state(self._state):
                break
            before = self._score(self._state)
            action = self._phase_default_action(self._state)
            next_state, _, _, _ = self._backend.step(action)
            self._episode_steps += 1
            after = self._score(next_state)
            total_reward += self._reward_from_scores(before, after)
            self._state = next_state
            steps += 1
            if self._episode_steps >= self.config.max_steps_per_episode:
                break
        return float(total_reward), steps

    def _decode_action(
        self,
        state: dict[str, Any],
        action: int | float,
    ) -> tuple[dict[str, Any], int | None, bool, list[int]]:
        phase = str(state.get("state") or "OTHER")
        if phase == "SELECTING_HAND":
            hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
            hand_size = min(len(hand_cards), action_space.MAX_HAND)
            legal_ids = self._legal_hand_action_ids(state, hand_size) if hand_size > 0 else []
            if not legal_ids:
                return {"action_type": "WAIT"}, None, True, []
            req_id = int(action)
            invalid = req_id not in legal_ids
            chosen = req_id if not invalid else legal_ids[0]
            atype, mask_int = action_space.decode(hand_size, chosen)
            payload = {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}
            return payload, int(chosen), bool(invalid), legal_ids

        if phase in action_space_shop.SHOP_PHASES:
            legal_ids = action_space_shop.legal_action_ids(state)
            if not legal_ids:
                return {"action_type": "WAIT"}, None, True, []
            req_id = int(action)
            invalid = req_id not in legal_ids
            chosen = req_id if not invalid else legal_ids[0]
            payload = action_space_shop.action_from_id(state, chosen)
            return payload, int(chosen), bool(invalid), legal_ids

        fallback = self._phase_default_action(state)
        return fallback, None, True, []

    def _legal_action_ids(self, state: dict[str, Any]) -> list[int]:
        phase = str(state.get("state") or "OTHER")
        if phase == "SELECTING_HAND":
            hand_cards = ((state.get("hand") or {}).get("cards") or []) if isinstance(state.get("hand"), dict) else []
            hand_size = min(len(hand_cards), action_space.MAX_HAND)
            return self._legal_hand_action_ids(state, hand_size) if hand_size > 0 else []
        if phase in action_space_shop.SHOP_PHASES:
            return action_space_shop.legal_action_ids(state)
        return []

    def _legal_hand_action_ids(self, state: dict[str, Any], hand_size: int) -> list[int]:
        if hand_size <= 0:
            return []
        hands_left = int((state.get("round") or {}).get("hands_left") or 0)
        discards_left = int((state.get("round") or {}).get("discards_left") or 0)
        legal: list[int] = []
        for aid in action_space.legal_action_ids(hand_size):
            atype, mask_int = action_space.decode(hand_size, aid)
            if atype == action_space.PLAY and hands_left <= 0:
                continue
            if atype == action_space.DISCARD and discards_left <= 0:
                continue
            # Keep a valid fallback when no useful action remains.
            if atype == action_space.DISCARD and mask_int == 0 and hands_left > 0:
                continue
            legal.append(int(aid))
        return legal

    def _build_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = str(state.get("state") or "OTHER")
        phase_id = PHASE_TO_ID.get(phase, PHASE_TO_ID["OTHER"])
        legal_ids = self._legal_action_ids(state)
        action_mask = [0] * self.action_dim
        for aid in legal_ids:
            if 0 <= int(aid) < self.action_dim:
                action_mask[int(aid)] = 1

        feat = extract_features(state)
        shop_feat = extract_shop_features(state)

        context = list(feat.get("context") or [0.0] * 12)
        if len(context) < 12:
            context = (context + [0.0] * 12)[:12]
        norm_context = [
            float(context[0]) / 5.0,   # hands_left
            float(context[1]) / 5.0,   # discards_left
            float(context[2]) / 5000.0,  # chips
            float(context[3]) / 100.0,  # money
            float(context[4]) / 20.0,  # reroll_cost
            float(context[5]) / 5000.0,  # small blind
            float(context[6]) / 5000.0,  # big blind
            float(context[7]) / 5000.0,  # boss blind
            float(context[8]) / 10.0,  # jokers_count
            float(context[9]) / 4.0,   # consumables_count
            float(context[10]) / 8.0,  # ante
            float(context[11]) / 20.0,  # round num
        ]

        shop_context = list(shop_feat.get("shop_context") or [0.0] * 16)
        if len(shop_context) < 16:
            shop_context = (shop_context + [0.0] * 16)[:16]

        rank_ids = list(feat.get("card_rank_ids") or [])
        chip_hint = list(feat.get("card_chip_hint") or [])
        enh_flags = list(feat.get("card_has_enhancement") or [])
        edt_flags = list(feat.get("card_has_edition") or [])
        seal_flags = list(feat.get("card_has_seal") or [])
        hand_size = int(feat.get("hand_size") or 0)
        divisor = max(1, hand_size)

        rank_mean = (sum(float(x) for x in rank_ids[: action_space.MAX_HAND]) / divisor) / 14.0
        rank_max = (max([float(x) for x in rank_ids[: action_space.MAX_HAND]] or [0.0])) / 14.0
        chip_mean = (sum(float(x) for x in chip_hint[: action_space.MAX_HAND]) / divisor) / 20.0
        enh_mean = (sum(float(x) for x in enh_flags[: action_space.MAX_HAND]) / divisor)
        edt_mean = (sum(float(x) for x in edt_flags[: action_space.MAX_HAND]) / divisor)
        seal_mean = (sum(float(x) for x in seal_flags[: action_space.MAX_HAND]) / divisor)

        phase_onehot = [0.0] * len(PHASE_TO_ID)
        phase_onehot[min(phase_id, len(phase_onehot) - 1)] = 1.0

        misc = [
            float(hand_size) / float(max(1, action_space.MAX_HAND)),
            float(len(legal_ids)) / float(max(1, self.action_dim)),
            float(state.get("ante_num") or 0.0) / 8.0,
            float(state.get("round_num") or 0.0) / 20.0,
            float((state.get("round") or {}).get("hands_left") or 0.0) / 5.0,
            float((state.get("round") or {}).get("discards_left") or 0.0) / 5.0,
        ]

        vector = [
            *norm_context,
            *shop_context[:16],
            rank_mean,
            rank_max,
            chip_mean,
            enh_mean,
            edt_mean,
            seal_mean,
            *phase_onehot,
            *misc,
        ]
        if len(vector) < self.obs_dim:
            vector.extend([0.0] * (self.obs_dim - len(vector)))
        vector = vector[: self.obs_dim]

        return {
            "vector": vector,
            "phase": phase,
            "phase_id": phase_id,
            "legal_action_ids": legal_ids,
            "action_mask": action_mask,
            "action_dim": self.action_dim,
            "episode_step": self._episode_steps,
            "episode_return": float(self._episode_return),
            "score": self._score(state),
            "ante_num": int(state.get("ante_num") or 0),
        }
