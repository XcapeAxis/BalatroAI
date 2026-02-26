if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from trainer import action_space
from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
from trainer.risk_controller import PolicySignal, load_config as load_risk_config, select_policy as select_risk_policy
from trainer.search_expert import choose_action as choose_search_action
from trainer.utils import setup_logger, timestamp, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for BC policy in eval_long_horizon.py") from exc
    return torch, nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-horizon end-to-end evaluation on sim backend.")
    parser.add_argument("--backend", choices=["sim"], default="sim")
    parser.add_argument("--stake", default="gold")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seed-prefix", default="AAAAAAA")
    parser.add_argument("--policy", choices=["heuristic", "bc", "search", "pv", "hybrid", "risk_aware"], default="heuristic")
    parser.add_argument("--model", default=None, help="Required when --policy bc/pv/hybrid/risk_aware.")
    parser.add_argument("--rl-model", default=None, help="Optional RL model path for --policy risk_aware.")
    parser.add_argument("--risk-config", default="trainer/config/p19_risk_controller.yaml")
    parser.add_argument("--search-budget-ms", type=float, default=20.0)
    parser.add_argument("--seeds-file", default=None, help="Optional seeds file, one seed per line.")
    parser.add_argument("--save-episode-logs", default=None, help="Optional output jsonl for episode-level logs.")
    parser.add_argument("--max-steps-per-episode", type=int, default=600)
    parser.add_argument("--target-ante", type=int, default=8)
    parser.add_argument("--out", required=True, help="Output JSON summary path.")
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())
    parser.add_argument("--max-shop-actions", type=int, default=action_space_shop.max_actions())
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _build_hand_model(nn, max_actions: int):
    class BCHandModel(nn.Module):
        def __init__(self, max_actions: int):
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
            self.head = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, max_actions),
            )

        def forward(self, batch):
            import torch

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
            return self.head(fused)

    return BCHandModel(max_actions)


def _build_multi_model(nn, max_actions: int, max_shop_actions: int):
    class BCMultiModel(nn.Module):
        def __init__(self, max_actions: int, max_shop_actions: int):
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

        def forward_hand(self, batch):
            import torch

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

        def forward_shop(self, batch):
            h = self.shop_proj(batch["shop_context"])
            return self.shop_head(h)

        def forward(self, batch):
            return self.forward_hand(batch)

    return BCMultiModel(max_actions, max_shop_actions)


def _state_to_hand_batch(state, torch):
    f = extract_features(state)
    chip_hint = list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND)
    return {
        "rank": torch.tensor([f["card_rank_ids"]], dtype=torch.long),
        "suit": torch.tensor([f["card_suit_ids"]], dtype=torch.long),
        "chip": torch.tensor([chip_hint], dtype=torch.float32),
        "enh": torch.tensor([f["card_has_enhancement"]], dtype=torch.float32),
        "edt": torch.tensor([f["card_has_edition"]], dtype=torch.float32),
        "seal": torch.tensor([f["card_has_seal"]], dtype=torch.float32),
        "pad": torch.tensor([f["hand_pad_mask"]], dtype=torch.float32),
        "context": torch.tensor([f["context"]], dtype=torch.float32),
    }


def _state_to_shop_batch(state, torch):
    sf = extract_shop_features(state)
    ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
    if len(ctx) != SHOP_CONTEXT_DIM:
        ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
    return {"shop_context": torch.tensor([ctx], dtype=torch.float32)}


def _masked_argmax(logits, legal_ids: list[int], max_actions: int, torch, device):
    valid = [int(a) for a in legal_ids if 0 <= int(a) < max_actions]
    if not valid:
        return None
    mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
    for aid in valid:
        mask[0, aid] = 1.0
    masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    return int(masked.argmax(dim=1).item())


def _masked_topk(logits, legal_ids: list[int], max_actions: int, k: int, torch, device) -> list[int]:
    valid = [int(a) for a in legal_ids if 0 <= int(a) < max_actions]
    if not valid:
        return []
    mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
    for aid in valid:
        mask[0, aid] = 1.0
    masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    kk = min(max(1, int(k)), masked.shape[1])
    return [int(x) for x in torch.topk(masked, k=kk, dim=1).indices[0].tolist()]


def _masked_top1_with_conf(logits, legal_ids: list[int], max_actions: int, torch, device) -> tuple[int | None, float]:
    valid = [int(a) for a in legal_ids if 0 <= int(a) < max_actions]
    if not valid:
        return None, 0.0
    mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
    for aid in valid:
        mask[0, aid] = 1.0
    masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(masked, dim=1)
    top_vals, top_ids = torch.topk(probs, k=1, dim=1)
    return int(top_ids[0, 0].item()), float(top_vals[0, 0].item())


def _load_model(args, logger):
    torch, nn = _require_torch()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if not args.model:
        raise RuntimeError("--model is required for policy=bc/pv")
    state_dict = torch.load(args.model, map_location=device)
    pv_load_error = None
    if args.policy in {"pv", "hybrid", "risk_aware"}:
        from trainer.models.policy_value import PolicyValueModel

        model = PolicyValueModel(args.max_actions, args.max_shop_actions).to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            logger.info("Loaded PV model=%s", args.model)
            return torch, device, model, True, "pv"
        except Exception as exc:
            pv_load_error = exc
            logger.warning("PV load failed for %s; attempting BC fallback: %s", args.model, exc)

    is_multi = any(str(k).startswith("hand_head.") for k in state_dict.keys())
    if is_multi:
        model = _build_multi_model(nn, args.max_actions, args.max_shop_actions).to(device)
    else:
        model = _build_hand_model(nn, args.max_actions).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # If caller explicitly requested BC and this fails, surface the direct error.
        # If caller requested PV/hybrid and both loaders fail, surface both contexts.
        if args.policy in {"pv", "hybrid", "risk_aware"} and pv_load_error is not None:
            raise RuntimeError(
                f"PV and BC fallback both failed for checkpoint {args.model}: "
                f"pv_error={pv_load_error!s}"
            )
        raise
    model.eval()
    if args.policy in {"pv", "hybrid", "risk_aware"} and pv_load_error is not None:
        logger.warning("Using BC fallback model for requested policy=%s", args.policy)
    logger.info("Loaded BC model=%s multi_head=%s", args.model, is_multi)
    return torch, device, model, is_multi, "bc"


def _load_model_from_path(model_path: str, args, logger):
    class _Shim:
        pass

    shim = _Shim()
    shim.device = args.device
    shim.model = model_path
    shim.policy = "pv"
    shim.max_actions = args.max_actions
    shim.max_shop_actions = args.max_shop_actions
    return _load_model(shim, logger)


def _phase_macro_action(state: dict, seed: str) -> dict:
    decision = choose_action(state, start_seed=seed)
    macro_action = str(decision.macro_action or "WAIT").upper()
    action = {"action_type": macro_action}
    action.update(dict(decision.macro_params or {}))
    return action


def _pick_hand_action(
    state: dict,
    args,
    torch,
    device,
    model,
    is_multi,
    model_kind: str,
    episode_seed: str,
    rl_bundle: tuple[Any, bool, str] | None = None,
    risk_cfg: dict[str, Any] | None = None,
) -> tuple[dict, str]:
    hand = (state.get("hand") or {}).get("cards") or []
    hand_size = min(len(hand), action_space.MAX_HAND)
    if hand_size <= 0:
        return {"action_type": "WAIT", "sleep": 0.01}, "no_cards_wait"

    if args.policy == "heuristic":
        d = choose_action(state, start_seed=episode_seed)
        if d.action_type and d.mask_int is not None:
            idxs = action_space.mask_to_indices(d.mask_int, hand_size)
            return {"action_type": d.action_type, "indices": idxs}, "heuristic"
        return {"action_type": "WAIT", "sleep": 0.01}, "heuristic_fallback_wait"

    if args.policy == "search":
        d = choose_search_action(
            state,
            max_branch=80,
            max_depth=2,
            time_budget_ms=float(args.search_budget_ms),
            seed=episode_seed,
        )
        return {"action_type": d.action_type, "indices": d.indices}, "search"

    legal_ids = action_space.legal_action_ids(hand_size)
    batch = _state_to_hand_batch(state, torch)
    batch = {k: v.to(device) for k, v in batch.items()}

    if args.policy == "hybrid":
        with torch.no_grad():
            if model_kind == "pv":
                logits, _ = model.forward_hand(batch)
            elif is_multi:
                logits = model.forward_hand(batch)
            else:
                logits = model(batch)
        top_ids = _masked_topk(logits, legal_ids, args.max_actions, 3, torch, device)
        if not top_ids:
            return {"action_type": "WAIT", "sleep": 0.01}, "hybrid_no_legal_wait"

        d = choose_search_action(
            state,
            max_branch=80,
            max_depth=2,
            time_budget_ms=float(args.search_budget_ms),
            seed=episode_seed,
        )
        search_id = None
        try:
            if d.action_type in {"PLAY", "DISCARD"}:
                search_id = action_space.encode(hand_size, d.action_type, action_space.indices_to_mask(list(d.indices or [])))
        except Exception:
            search_id = None

        chosen_id = int(top_ids[0])
        reason = "hybrid_model_top1"
        if search_id is not None and int(search_id) in top_ids:
            chosen_id = int(search_id)
            reason = "hybrid_search_rerank"
        atype, mask_int = action_space.decode(hand_size, int(chosen_id))
        return {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}, reason

    if args.policy == "risk_aware":
        if rl_bundle is None:
            return {"action_type": "WAIT", "sleep": 0.01}, "risk_missing_rl_bundle"
        rl_model, rl_is_multi, rl_kind = rl_bundle

        with torch.no_grad():
            if model_kind == "pv":
                pv_logits, pv_value_t = model.forward_hand(batch)
                pv_value = float(pv_value_t.squeeze().item())
            elif is_multi:
                pv_logits = model.forward_hand(batch)
                pv_value = None
            else:
                pv_logits = model(batch)
                pv_value = None

            if rl_kind == "pv":
                rl_logits, rl_value_t = rl_model.forward_hand(batch)
                rl_value = float(rl_value_t.squeeze().item())
            elif rl_is_multi:
                rl_logits = rl_model.forward_hand(batch)
                rl_value = None
            else:
                rl_logits = rl_model(batch)
                rl_value = None

        pv_id, pv_conf = _masked_top1_with_conf(pv_logits, legal_ids, args.max_actions, torch, device)
        rl_id, rl_conf = _masked_top1_with_conf(rl_logits, legal_ids, args.max_actions, torch, device)
        pv_top3 = _masked_topk(pv_logits, legal_ids, args.max_actions, 3, torch, device)

        heur = choose_action(state, start_seed=episode_seed)
        heur_id = None
        try:
            if heur.action_type and heur.mask_int is not None:
                heur_id = action_space.encode(hand_size, heur.action_type, int(heur.mask_int))
        except Exception:
            heur_id = None

        hybrid_id = pv_id

        context_vals = [float(x) for x in batch["context"][0].detach().cpu().tolist()]
        decision = select_risk_policy(
            phase_group="hand",
            signals=[
                PolicySignal("pv", pv_id, pv_conf, pv_value, pv_id is not None),
                PolicySignal("rl", rl_id, rl_conf, rl_value, rl_id is not None),
                PolicySignal("hybrid", hybrid_id, max(pv_conf, rl_conf), pv_value, hybrid_id is not None),
                PolicySignal("heuristic", heur_id, 0.50, None, heur_id is not None),
            ],
            context=context_vals,
            cfg=risk_cfg or {},
            available_policies={"pv", "rl", "hybrid", "search", "heuristic"},
        )
        selected = str(decision.get("selected_policy") or "heuristic")
        chosen_id = None
        if selected == "pv":
            chosen_id = pv_id
        elif selected == "rl":
            chosen_id = rl_id
        elif selected == "hybrid":
            chosen_id = hybrid_id
        elif selected == "search":
            search_id = None
            try:
                search_decision = choose_search_action(
                    state,
                    max_branch=80,
                    max_depth=2,
                    time_budget_ms=float(args.search_budget_ms),
                    seed=episode_seed,
                )
                if search_decision.action_type in {"PLAY", "DISCARD"}:
                    search_id = action_space.encode(
                        hand_size,
                        search_decision.action_type,
                        action_space.indices_to_mask(list(search_decision.indices or [])),
                    )
            except Exception:
                search_id = None
            chosen_id = search_id
        elif selected == "heuristic":
            chosen_id = heur_id
        if chosen_id is None:
            chosen_id = pv_id if pv_id is not None else rl_id
        if chosen_id is None:
            return {"action_type": "WAIT", "sleep": 0.01}, "risk_no_legal_wait"
        atype, mask_int = action_space.decode(hand_size, int(chosen_id))
        return {
            "action_type": atype,
            "indices": action_space.mask_to_indices(mask_int, hand_size),
        }, f"risk_aware:{selected}:risk={float(decision.get('risk_score') or 0.0):.3f}"

    # bc / pv policy
    with torch.no_grad():
        if model_kind == "pv":
            logits, _ = model.forward_hand(batch)
        elif is_multi:
            logits = model.forward_hand(batch)
        else:
            logits = model(batch)
    pred = _masked_argmax(logits, legal_ids, args.max_actions, torch, device)
    if pred is None:
        return {"action_type": "WAIT", "sleep": 0.01}, "bc_no_legal_wait"
    atype, mask_int = action_space.decode(hand_size, int(pred))
    return {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}, "bc"


def _pick_shop_action(
    state: dict,
    args,
    torch,
    device,
    model,
    is_multi,
    model_kind: str,
    rl_bundle: tuple[Any, bool, str] | None = None,
    risk_cfg: dict[str, Any] | None = None,
) -> tuple[dict, str]:
    if args.policy in {"heuristic", "search"} or ((args.policy in {"bc", "pv", "hybrid"}) and not is_multi):
        d = choose_shop_action(state)
        return dict(d.action), "heuristic_shop"

    if args.policy == "hybrid":
        legal_ids = action_space_shop.legal_action_ids(state)
        batch = _state_to_shop_batch(state, torch)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if model_kind == "pv":
                logits, _ = model.forward_shop(batch)
            else:
                logits = model.forward_shop(batch)
        top_ids = _masked_topk(logits, legal_ids, args.max_shop_actions, 3, torch, device)
        if not top_ids:
            return {"action_type": "WAIT", "sleep": 0.01}, "hybrid_shop_wait"
        chosen = int(top_ids[0])
        reason = "hybrid_shop_model_top1"
        heur = choose_shop_action(state)
        try:
            heur_id = action_space_shop.encode(str((heur.action or {}).get("action_type") or "WAIT"), (heur.action or {}).get("params") or {})
            if int(heur_id) in top_ids:
                chosen = int(heur_id)
                reason = "hybrid_shop_heuristic_rerank"
        except Exception:
            pass
        return action_space_shop.action_from_id(state, int(chosen)), reason

    if args.policy == "risk_aware":
        heur_action = dict((choose_shop_action(state).action) or {"action_type": "WAIT"})
        heur_id = None
        try:
            heur_id = action_space_shop.encode(str(heur_action.get("action_type") or "WAIT"), dict(heur_action.get("params") or {}))
        except Exception:
            heur_id = None

        legal_ids = action_space_shop.legal_action_ids(state)
        pv_id, pv_conf = None, 0.0
        rl_id, rl_conf = None, 0.0
        batch = _state_to_shop_batch(state, torch)
        batch = {k: v.to(device) for k, v in batch.items()}

        if is_multi:
            with torch.no_grad():
                if model_kind == "pv":
                    pv_logits, _ = model.forward_shop(batch)
                else:
                    pv_logits = model.forward_shop(batch)
            pv_id, pv_conf = _masked_top1_with_conf(pv_logits, legal_ids, args.max_shop_actions, torch, device)

        if rl_bundle is not None:
            rl_model, rl_is_multi, rl_kind = rl_bundle
            if rl_is_multi:
                with torch.no_grad():
                    if rl_kind == "pv":
                        rl_logits, _ = rl_model.forward_shop(batch)
                    else:
                        rl_logits = rl_model.forward_shop(batch)
                rl_id, rl_conf = _masked_top1_with_conf(rl_logits, legal_ids, args.max_shop_actions, torch, device)

        ctx_vals = [float(x) for x in batch["shop_context"][0].detach().cpu().tolist()]
        decision = select_risk_policy(
            phase_group="shop",
            signals=[
                PolicySignal("pv", pv_id, pv_conf, None, pv_id is not None),
                PolicySignal("rl", rl_id, rl_conf, None, rl_id is not None),
                PolicySignal("hybrid", pv_id, max(pv_conf, rl_conf), None, pv_id is not None),
                PolicySignal("heuristic", heur_id, 0.50, None, heur_id is not None),
            ],
            context=ctx_vals,
            cfg=risk_cfg or {},
            available_policies={"pv", "rl", "hybrid", "heuristic"},
        )
        selected = str(decision.get("selected_policy") or "heuristic")
        chosen_id = None
        if selected in {"pv", "hybrid"}:
            chosen_id = pv_id
        elif selected == "rl":
            chosen_id = rl_id
        elif selected == "heuristic":
            chosen_id = heur_id

        if chosen_id is not None:
            return action_space_shop.action_from_id(state, int(chosen_id)), f"risk_aware_shop:{selected}:risk={float(decision.get('risk_score') or 0.0):.3f}"
        return heur_action, "risk_aware_shop:heuristic_fallback"

    legal_ids = action_space_shop.legal_action_ids(state)
    batch = _state_to_shop_batch(state, torch)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        if model_kind == "pv":
            logits, _ = model.forward_shop(batch)
        else:
            logits = model.forward_shop(batch)
    pred = _masked_argmax(logits, legal_ids, args.max_shop_actions, torch, device)
    if pred is None:
        return {"action_type": "WAIT", "sleep": 0.01}, "bc_shop_no_legal_wait"
    return action_space_shop.action_from_id(state, int(pred)), "bc_shop"


def _ante_value(state: dict) -> int:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    return int(round_info.get("ante") or state.get("ante_num") or 0)


def _failure_reason(state: dict) -> str:
    round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
    hands_left = int(round_info.get("hands_left") or 0)
    discards_left = int(round_info.get("discards_left") or 0)
    chips = float(round_info.get("chips") or 0.0)
    target = float((state.get("score") or {}).get("target_chips") or state.get("target_chips") or 0.0)
    money = float(state.get("money") or 0.0)
    if hands_left <= 0 and discards_left <= 0 and chips < target:
        return "blind_score_shortfall"
    if hands_left <= 0 and discards_left <= 0:
        return "resource_exhausted"
    if money <= 0:
        return "economy_collapse"
    return "other"


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.eval_long_horizon")
    warn_if_unstable_python(logger)

    torch = None
    device = None
    model = None
    is_multi = False
    model_kind = "heuristic"
    rl_bundle = None
    risk_cfg = {}
    if args.policy in {"bc", "pv", "hybrid", "risk_aware"}:
        try:
            torch, device, model, is_multi, model_kind = _load_model(args, logger)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return 2
    if args.policy == "risk_aware":
        rl_model_path = str(args.rl_model or args.model or "").strip()
        if not rl_model_path:
            logger.error("--rl-model (or --model) is required for policy=risk_aware")
            return 2
        try:
            _, _, rl_model, rl_is_multi, rl_kind = _load_model_from_path(rl_model_path, args, logger)
            rl_bundle = (rl_model, rl_is_multi, rl_kind)
            risk_cfg = load_risk_config(args.risk_config)
        except Exception as exc:
            logger.error("Failed to load rl model: %s", exc)
            return 2

    backend = create_backend("sim", seed=args.seed_prefix, timeout_sec=8.0, logger=logger)

    wins = 0
    ante_values: list[int] = []
    money_ends: list[float] = []
    chips_ends: list[float] = []
    rounds_list: list[int] = []
    actions_list: list[int] = []
    degraded = False
    failure_counter: Counter[str] = Counter()
    episode_logs: list[dict] = []

    seeds: list[str] = []
    if args.seeds_file:
        try:
            seeds = [line.strip() for line in Path(args.seeds_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            seeds = []
    if not seeds:
        seeds = [f"{args.seed_prefix}-{args.seed}-{ep}" for ep in range(args.episodes)]

    for ep in range(args.episodes):
        episode_seed = seeds[ep % len(seeds)]
        state = backend.reset(seed=episode_seed)

        if str(args.stake or "").strip().lower() not in {"", "white"}:
            try:
                state, _, _, _ = backend.step({"action_type": "START", "seed": episode_seed, "stake": str(args.stake).upper()})
            except Exception:
                pass

        actions_taken = 0
        for _step in range(args.max_steps_per_episode):
            phase = str(state.get("state") or "UNKNOWN")
            if phase == "GAME_OVER":
                break
            if phase == "SELECTING_HAND":
                action, _ = _pick_hand_action(
                    state,
                    args,
                    torch,
                    device,
                    model,
                    is_multi,
                    model_kind,
                    episode_seed,
                    rl_bundle=rl_bundle,
                    risk_cfg=risk_cfg,
                )
            elif phase in action_space_shop.SHOP_PHASES:
                action, _ = _pick_shop_action(
                    state,
                    args,
                    torch,
                    device,
                    model,
                    is_multi,
                    model_kind,
                    rl_bundle=rl_bundle,
                    risk_cfg=risk_cfg,
                )
            else:
                action = _phase_macro_action(state, episode_seed)
            try:
                state, _, done, _ = backend.step(action)
            except Exception:
                fallback = _phase_macro_action(state, episode_seed)
                try:
                    state, _, done, _ = backend.step(fallback)
                except Exception:
                    failure_counter["step_error"] += 1
                    break
            actions_taken += 1
            if done:
                break

        round_info = state.get("round") if isinstance(state.get("round"), dict) else {}
        ante = _ante_value(state)
        round_num = int(round_info.get("round_num") or state.get("round_num") or 0)
        money = float(state.get("money") or 0.0)
        chips = float(round_info.get("chips") or 0.0)
        done = str(state.get("state") or "") == "GAME_OVER"
        won = bool(state.get("won") or (state.get("flags") or {}).get("won"))
        reach_target = ante >= int(args.target_ante)
        if won or reach_target:
            wins += 1
            failure_reason = "win"
        elif done:
            failure_reason = _failure_reason(state)
            failure_counter[failure_reason] += 1
        else:
            failure_reason = "max_steps_cutoff"
            failure_counter["max_steps_cutoff"] += 1

        rules = state.get("rules") if isinstance(state.get("rules"), dict) else {}
        degraded = degraded or bool(rules.get("degraded") or False)

        ante_values.append(ante)
        money_ends.append(money)
        chips_ends.append(chips)
        rounds_list.append(round_num)
        actions_list.append(actions_taken)
        episode_logs.append(
            {
                "episode_id": int(ep),
                "seed": str(episode_seed),
                "result": "win" if failure_reason == "win" else "loss",
                "final_ante": int(ante),
                "final_money": float(money),
                "final_chips": float(chips),
                "failure_reason": str(failure_reason),
                "failure_phase": str(state.get("state") or ""),
                "boss_blind_id": str((state.get("round") or {}).get("boss_blind") or ""),
                "steps": int(actions_taken),
                "actions": int(actions_taken),
            }
        )

    backend.close()

    if not ante_values:
        logger.error("No episode results produced.")
        return 2

    ante_sorted = sorted(ante_values)
    median_ante = ante_sorted[len(ante_sorted) // 2]
    summary = {
        "generated_at": timestamp(),
        "policy": args.policy,
        "stake": str(args.stake),
        "episodes": int(args.episodes),
        "target_ante": int(args.target_ante),
        "win_rate": wins / len(ante_values),
        "avg_ante_reached": sum(ante_values) / len(ante_values),
        "median_ante": float(median_ante),
        "avg_money_end": sum(money_ends) / len(money_ends),
        "avg_chips_end": sum(chips_ends) / len(chips_ends),
        "avg_rounds": sum(rounds_list) / len(rounds_list),
        "avg_actions": sum(actions_list) / len(actions_list),
        "failure_breakdown": dict(sorted(failure_counter.items())),
        "episode_log_values": {
            "ante_values": ante_values,
            "money_end_values": money_ends,
            "chips_end_values": chips_ends,
        },
        "stake_coverage": {
            "degraded": bool(degraded),
            "note": "stake modifiers partially degraded" if degraded else "stake modifiers fully applied",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.save_episode_logs:
        logs_path = Path(args.save_episode_logs)
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with logs_path.open("w", encoding="utf-8") as f:
            for row in episode_logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    txt_path = out_path.with_suffix(".txt")
    lines = [
        "P10 Long Horizon Evaluation",
        f"generated_at={summary['generated_at']}",
        f"policy={summary['policy']} stake={summary['stake']} episodes={summary['episodes']}",
        f"win_rate={summary['win_rate']:.4f}",
        f"avg_ante_reached={summary['avg_ante_reached']:.4f}",
        f"median_ante={summary['median_ante']:.4f}",
        f"avg_money_end={summary['avg_money_end']:.4f}",
        f"avg_chips_end={summary['avg_chips_end']:.4f}",
        f"avg_rounds={summary['avg_rounds']:.4f}",
        f"avg_actions={summary['avg_actions']:.4f}",
        f"failure_breakdown={summary['failure_breakdown']}",
        f"stake_coverage={summary['stake_coverage']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("wrote %s", out_path)
    logger.info("wrote %s", txt_path)
    logger.info(
        "long_horizon: policy=%s stake=%s win_rate=%.4f avg_ante=%.3f median_ante=%.1f",
        summary["policy"],
        summary["stake"],
        summary["win_rate"],
        summary["avg_ante_reached"],
        summary["median_ante"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
