from __future__ import annotations

from pathlib import Path
from typing import Any

from trainer import action_space, action_space_shop
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
from trainer.legal_actions import legal_hand_action_ids_for_state
from trainer.policy_arena.adapters.heuristic_adapter import HeuristicAdapter
from trainer.policy_arena.policy_adapter import AdapterDescriptor, BasePolicyAdapter, normalize_action, phase_from_obs


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - runtime guarded
        raise RuntimeError("PyTorch is required for model arena inference") from exc
    return torch, nn


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_bc_model(nn: Any, *, max_actions: int, max_shop_actions: int):
    class BCMultiModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rank_emb = nn.Embedding(16, 16)
            self.suit_emb = nn.Embedding(8, 8)
            self.card_proj = nn.Sequential(
                nn.Linear(16 + 8 + 4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.ctx_proj = nn.Sequential(
                nn.Linear(12, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.hand_head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, max_actions),
            )
            self.shop_proj = nn.Sequential(
                nn.Linear(SHOP_CONTEXT_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.shop_head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, max_shop_actions),
            )

        def forward_hand(self, batch: dict[str, Any]):
            torch, _nn = _require_torch()
            rank = batch["rank"]
            suit = batch["suit"]
            chip = batch["chip"].unsqueeze(-1)
            enh = batch["enh"].unsqueeze(-1)
            edt = batch["edt"].unsqueeze(-1)
            seal = batch["seal"].unsqueeze(-1)
            pad = batch["pad"]

            r = self.rank_emb(rank)
            s = self.suit_emb(suit)
            card_x = torch.cat([r, s, chip, enh, edt, seal], dim=-1)
            card_h = self.card_proj(card_x)
            pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum
            ctx_h = self.ctx_proj(batch["context"])
            fused = torch.cat([pooled, ctx_h], dim=-1)
            return self.hand_head(fused)

        def forward_shop(self, batch: dict[str, Any]):
            h = self.shop_proj(batch["shop_context"])
            return self.shop_head(h)

    return BCMultiModel()


class ModelAdapter(BasePolicyAdapter):
    def __init__(
        self,
        *,
        name: str = "model_policy",
        model_path: str = "",
        strategy: str = "bc",
    ) -> None:
        self.model_path = str(model_path or "").strip()
        self.strategy = str(strategy or "bc").strip().lower()
        self._fallback = HeuristicAdapter(name=f"{name}_fallback")
        self._available = False
        self._load_error = ""
        self._model = None
        self._device = None
        self._max_actions = action_space.max_actions()
        self._max_shop_actions = action_space_shop.max_actions()
        if self.model_path and Path(self.model_path).exists():
            self._available = self._load_model(Path(self.model_path))

        note = "Model checkpoint loaded."
        status = "active"
        if not self._available:
            status = "stub"
            note = (
                "Model adapter running in fallback mode. "
                + (self._load_error or "Checkpoint unavailable or incompatible.")
            )
        super().__init__(
            descriptor=AdapterDescriptor(
                name=name,
                family="model",
                status=status,
                supports_batch=False,
                supports_shop=True,
                supports_consumables=True,
                supports_position_actions=False,
                notes=note,
            )
        )

    def _load_model(self, checkpoint_path: Path) -> bool:
        try:
            torch, nn = _require_torch()
        except Exception as exc:
            self._load_error = str(exc)
            return False

        config = _load_json(checkpoint_path.parent / "config.json")
        self._max_actions = max(1, _safe_int(config.get("max_actions"), action_space.max_actions()))
        self._max_shop_actions = max(1, _safe_int(config.get("max_shop_actions"), action_space_shop.max_actions()))
        model = _build_bc_model(nn, max_actions=self._max_actions, max_shop_actions=self._max_shop_actions)
        try:
            state_dict = torch.load(str(checkpoint_path), map_location="cpu")
            if not isinstance(state_dict, dict):
                self._load_error = f"unexpected checkpoint format:{checkpoint_path}"
                return False
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            self._load_error = f"checkpoint_load_failed:{exc}"
            return False

        model.eval()
        self._model = model
        self._device = torch.device("cpu")
        return True

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["adapter"]["strategy"] = self.strategy
        payload["adapter"]["model_path"] = self.model_path
        payload["adapter"]["available"] = bool(self._available)
        payload["adapter"]["load_error"] = self._load_error
        return payload

    def reset(self, seed: str | int | None = None) -> None:
        super().reset(seed)
        self._fallback.reset(seed)

    def _legal_hand_ids(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None) -> list[int]:
        state_legal_ids = legal_hand_action_ids_for_state(obs)
        if isinstance(legal_actions, list) and legal_actions:
            ids = [_safe_int(row.get("id"), -1) for row in legal_actions if isinstance(row, dict)]
            ids = [aid for aid in ids if aid >= 0]
            if ids:
                if state_legal_ids:
                    state_legal_set = set(state_legal_ids)
                    filtered = [aid for aid in ids if aid in state_legal_set]
                    if filtered:
                        return filtered
                return ids
        return state_legal_ids

    def _legal_shop_ids(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None) -> list[int]:
        if isinstance(legal_actions, list) and legal_actions:
            ids = [_safe_int(row.get("id"), -1) for row in legal_actions if isinstance(row, dict)]
            ids = [aid for aid in ids if aid >= 0]
            if ids:
                return ids
        return action_space_shop.legal_action_ids(obs)

    def _act_hand(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None) -> dict[str, Any]:
        torch, _nn = _require_torch()
        if self._model is None:
            raise RuntimeError("model_unavailable")
        features = extract_features(obs)
        hand_size = max(1, _safe_int(features.get("hand_size"), 1))
        legal_ids = [aid for aid in self._legal_hand_ids(obs, legal_actions) if 0 <= aid < self._max_actions]
        if not legal_ids:
            raise RuntimeError("no_legal_hand_actions")
        batch = {
            "rank": torch.tensor([list(features.get("card_rank_ids") or [])], dtype=torch.long),
            "suit": torch.tensor([list(features.get("card_suit_ids") or [])], dtype=torch.long),
            "chip": torch.tensor([list(features.get("card_chip_hint") or [])], dtype=torch.float32),
            "enh": torch.tensor([list(features.get("card_has_enhancement") or [])], dtype=torch.float32),
            "edt": torch.tensor([list(features.get("card_has_edition") or [])], dtype=torch.float32),
            "seal": torch.tensor([list(features.get("card_has_seal") or [])], dtype=torch.float32),
            "pad": torch.tensor([list(features.get("hand_pad_mask") or [])], dtype=torch.float32),
            "context": torch.tensor([list(features.get("context") or [])], dtype=torch.float32),
        }
        legal_mask = torch.zeros((1, self._max_actions), dtype=torch.float32)
        for aid in legal_ids:
            legal_mask[0, aid] = 1.0
        with torch.no_grad():
            logits = self._model.forward_hand(batch)
            masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
            action_id = int(masked_logits.argmax(dim=1).item())
        action_type, mask = action_space.decode(hand_size, action_id)
        return normalize_action(
            {"action_type": action_type, "indices": action_space.mask_to_indices(mask, hand_size), "id": action_id},
            phase="SELECTING_HAND",
        )

    def _candidate_rows_hand(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        torch, _nn = _require_torch()
        if self._model is None:
            raise RuntimeError("model_unavailable")
        features = extract_features(obs)
        hand_size = max(1, _safe_int(features.get("hand_size"), 1))
        legal_ids = [aid for aid in self._legal_hand_ids(obs, legal_actions) if 0 <= aid < self._max_actions]
        if not legal_ids:
            return []
        batch = {
            "rank": torch.tensor([list(features.get("card_rank_ids") or [])], dtype=torch.long),
            "suit": torch.tensor([list(features.get("card_suit_ids") or [])], dtype=torch.long),
            "chip": torch.tensor([list(features.get("card_chip_hint") or [])], dtype=torch.float32),
            "enh": torch.tensor([list(features.get("card_has_enhancement") or [])], dtype=torch.float32),
            "edt": torch.tensor([list(features.get("card_has_edition") or [])], dtype=torch.float32),
            "seal": torch.tensor([list(features.get("card_has_seal") or [])], dtype=torch.float32),
            "pad": torch.tensor([list(features.get("hand_pad_mask") or [])], dtype=torch.float32),
            "context": torch.tensor([list(features.get("context") or [])], dtype=torch.float32),
        }
        legal_mask = torch.zeros((1, self._max_actions), dtype=torch.float32)
        for aid in legal_ids:
            legal_mask[0, aid] = 1.0
        with torch.no_grad():
            logits = self._model.forward_hand(batch)
            masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
            probs = torch.softmax(masked_logits, dim=1)
            count = min(max(1, int(top_k)), len(legal_ids))
            values, indices = torch.topk(masked_logits, k=count, dim=1)
        rows: list[dict[str, Any]] = []
        for rank, (logit, action_id) in enumerate(zip(values[0].tolist(), indices[0].tolist()), start=1):
            aid = int(action_id)
            action_type, mask = action_space.decode(hand_size, aid)
            rows.append(
                {
                    "action": normalize_action(
                        {"action_type": action_type, "indices": action_space.mask_to_indices(mask, hand_size), "id": aid},
                        phase="SELECTING_HAND",
                    ),
                    "source": "policy_topk",
                    "source_rank": int(rank),
                    "source_score": float(logit),
                    "legal": True,
                    "metadata": {"probability": float(probs[0, aid].item())},
                }
            )
        return rows

    def _act_shop(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None) -> dict[str, Any]:
        torch, _nn = _require_torch()
        if self._model is None:
            raise RuntimeError("model_unavailable")
        shop_features = extract_shop_features(obs)
        legal_ids = [aid for aid in self._legal_shop_ids(obs, legal_actions) if 0 <= aid < self._max_shop_actions]
        if not legal_ids:
            raise RuntimeError("no_legal_shop_actions")
        batch = {
            "shop_context": torch.tensor([list(shop_features.get("shop_context") or [])], dtype=torch.float32),
        }
        legal_mask = torch.zeros((1, self._max_shop_actions), dtype=torch.float32)
        for aid in legal_ids:
            legal_mask[0, aid] = 1.0
        with torch.no_grad():
            logits = self._model.forward_shop(batch)
            masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
            action_id = int(masked_logits.argmax(dim=1).item())
        action = action_space_shop.action_from_id(obs, action_id)
        action["id"] = action_id
        return normalize_action(action, phase=phase_from_obs(obs))

    def _candidate_rows_shop(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        torch, _nn = _require_torch()
        if self._model is None:
            raise RuntimeError("model_unavailable")
        shop_features = extract_shop_features(obs)
        legal_ids = [aid for aid in self._legal_shop_ids(obs, legal_actions) if 0 <= aid < self._max_shop_actions]
        if not legal_ids:
            return []
        batch = {
            "shop_context": torch.tensor([list(shop_features.get("shop_context") or [])], dtype=torch.float32),
        }
        legal_mask = torch.zeros((1, self._max_shop_actions), dtype=torch.float32)
        for aid in legal_ids:
            legal_mask[0, aid] = 1.0
        with torch.no_grad():
            logits = self._model.forward_shop(batch)
            masked_logits = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
            probs = torch.softmax(masked_logits, dim=1)
            count = min(max(1, int(top_k)), len(legal_ids))
            values, indices = torch.topk(masked_logits, k=count, dim=1)
        rows: list[dict[str, Any]] = []
        phase = phase_from_obs(obs)
        for rank, (logit, action_id) in enumerate(zip(values[0].tolist(), indices[0].tolist()), start=1):
            aid = int(action_id)
            action = action_space_shop.action_from_id(obs, aid)
            action["id"] = aid
            rows.append(
                {
                    "action": normalize_action(action, phase=phase),
                    "source": "policy_topk",
                    "source_rank": int(rank),
                    "source_score": float(logit),
                    "legal": True,
                    "metadata": {"probability": float(probs[0, aid].item())},
                }
            )
        return rows

    def act(self, obs: dict[str, Any], legal_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        phase = phase_from_obs(obs)
        if not self._available or self._model is None:
            action = self._fallback.act(obs, legal_actions=legal_actions)
            return normalize_action(action, phase=phase)
        try:
            if phase == "SELECTING_HAND":
                return self._act_hand(obs, legal_actions)
            if phase in action_space_shop.SHOP_PHASES:
                return self._act_shop(obs, legal_actions)
        except Exception:
            pass
        action = self._fallback.act(obs, legal_actions=legal_actions)
        return normalize_action(action, phase=phase)

    def candidate_actions(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]] | None = None,
        *,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        phase = phase_from_obs(obs)
        if not self._available or self._model is None:
            return self._fallback.candidate_actions(obs, legal_actions=legal_actions, top_k=top_k)
        try:
            if phase == "SELECTING_HAND":
                rows = self._candidate_rows_hand(obs, legal_actions, top_k=top_k)
                if rows:
                    return rows
            if phase in action_space_shop.SHOP_PHASES:
                rows = self._candidate_rows_shop(obs, legal_actions, top_k=top_k)
                if rows:
                    return rows
        except Exception:
            pass
        return self._fallback.candidate_actions(obs, legal_actions=legal_actions, top_k=top_k)

    def close(self) -> None:
        self._fallback.close()
        super().close()
