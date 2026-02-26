if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time

from trainer import action_space
from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
from trainer.risk_controller import PolicySignal, load_config as load_risk_config, select_policy as select_risk_policy
from trainer.search_expert import choose_action as choose_search_action
from trainer.utils import format_action, setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for infer_assistant.py") from exc
    return torch, nn


def parse_args():
    parser = argparse.ArgumentParser(description="Inference assistant for Balatro hand+shop decisions.")
    parser.add_argument("--backend", choices=["real", "sim"], default="real")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", default="")
    parser.add_argument("--policy-model", default="", help="Primary model path for pv/risk_aware.")
    parser.add_argument("--rl-model", default="", help="Optional RL model path for risk_aware.")
    parser.add_argument("--policy", choices=["bc", "pv", "risk_aware"], default="bc")
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())
    parser.add_argument("--max-shop-actions", type=int, default=action_space_shop.max_actions())
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--risk-config", default="trainer/config/p19_risk_controller.yaml")
    parser.add_argument("--search-budget-ms", type=float, default=10.0)
    parser.add_argument("--execute", action="store_true", help="Execute top-1 recommendation.")
    parser.add_argument("--once", action="store_true", help="Run one loop and exit.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--strict-errors",
        dest="strict_errors",
        action="store_true",
        default=True,
        help="Exit immediately when backend/state/action errors occur (default: true).",
    )
    parser.add_argument(
        "--no-strict-errors",
        dest="strict_errors",
        action="store_false",
        help="Keep polling on transient errors instead of exiting.",
    )
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
            rank = batch["rank"]
            suit = batch["suit"]
            chip = batch["chip"].unsqueeze(-1)
            enh = batch["enh"].unsqueeze(-1)
            edt = batch["edt"].unsqueeze(-1)
            seal = batch["seal"].unsqueeze(-1)
            pad = batch["pad"]

            r = self.rank_emb(rank)
            s = self.suit_emb(suit)
            card_x = __import__("torch").cat([r, s, chip, enh, edt, seal], dim=-1)
            card_h = self.card_proj(card_x)
            pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum
            ctx_h = self.ctx_proj(batch["context"])
            fused = __import__("torch").cat([pooled, ctx_h], dim=-1)
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
            rank = batch["rank"]
            suit = batch["suit"]
            chip = batch["chip"].unsqueeze(-1)
            enh = batch["enh"].unsqueeze(-1)
            edt = batch["edt"].unsqueeze(-1)
            seal = batch["seal"].unsqueeze(-1)
            pad = batch["pad"]

            r = self.rank_emb(rank)
            s = self.suit_emb(suit)
            card_x = __import__("torch").cat([r, s, chip, enh, edt, seal], dim=-1)
            card_h = self.card_proj(card_x)
            pad_sum = pad.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (card_h * pad.unsqueeze(-1)).sum(dim=1) / pad_sum
            ctx_h = self.ctx_proj(batch["context"])
            fused = __import__("torch").cat([pooled, ctx_h], dim=-1)
            return self.hand_head(fused)

        def forward_shop(self, batch):
            h = self.shop_proj(batch["shop_context"])
            return self.shop_head(h)

        def forward(self, batch):
            return self.forward_hand(batch)

    return BCMultiModel(max_actions, max_shop_actions)


def _resolve_model_path(args) -> str:
    primary = str(args.policy_model or args.model or "").strip()
    if not primary:
        raise RuntimeError("--model or --policy-model is required")
    return primary


def _load_single_model(model_path: str, policy_hint: str, args, torch, nn, device):
    state_dict = torch.load(model_path, map_location=device)
    if policy_hint in {"pv", "risk_aware"}:
        from trainer.models.policy_value import PolicyValueModel

        model = PolicyValueModel(args.max_actions, args.max_shop_actions).to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
            return model, True, "pv"
        except Exception:
            # Fallback to BC structures for compatibility with older checkpoints.
            pass

    is_multi = any(k.startswith("hand_head.") for k in state_dict.keys())
    if is_multi:
        model = _build_multi_model(nn, args.max_actions, args.max_shop_actions).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model, True, "bc"

    model = _build_hand_model(nn, args.max_actions).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, False, "bc"


def _load_model(args, torch, nn, device):
    # Backward-compatible wrapper used by real inference helpers.
    model_path = _resolve_model_path(args)
    return _load_single_model(model_path, args.policy, args, torch, nn, device)


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
    return {"shop_context": torch.tensor([sf["shop_context"]], dtype=torch.float32)}


def _forward_hand(model, batch, is_multi, policy_kind):
    if policy_kind == "pv":
        logits, _ = model.forward_hand(batch)
        return logits
    if is_multi:
        return model.forward_hand(batch)
    return model(batch)


def _forward_shop(model, batch, policy_kind):
    if policy_kind == "pv":
        logits, _ = model.forward_shop(batch)
        return logits
    return model.forward_shop(batch)


def _predict_topk(logits, legal_ids, max_actions, topk, torch, device):
    legal = [int(a) for a in legal_ids if 0 <= int(a) < max_actions]
    if not legal:
        return []
    legal_mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
    for aid in legal:
        legal_mask[0, aid] = 1.0
    masked = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
    k = min(max(1, topk), masked.shape[1])
    top_vals, top_ids = torch.topk(masked, k=k, dim=1)
    out = []
    for i in range(k):
        out.append((int(top_ids[0, i].item()), float(top_vals[0, i].item())))
    return out


def _predict_top1_with_conf(logits, legal_ids, max_actions, torch, device):
    legal = [int(a) for a in legal_ids if 0 <= int(a) < max_actions]
    if not legal:
        return None, 0.0, []
    legal_mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
    for aid in legal:
        legal_mask[0, aid] = 1.0
    masked = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
    probs = torch.softmax(masked, dim=1)
    top_vals, top_ids = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
    aid = int(top_ids[0, 0].item())
    conf = float(top_vals[0, 0].item())
    top2_gap = conf - float(top_vals[0, 1].item()) if top_vals.shape[1] > 1 else conf
    return aid, conf, [conf, top2_gap]


def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.infer")
    warn_if_unstable_python(logger)

    try:
        torch, nn = _require_torch()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    primary_path = _resolve_model_path(args)
    model, is_multi, policy_kind = _load_single_model(primary_path, args.policy, args, torch, nn, device)
    model.eval()

    rl_model = None
    rl_is_multi = False
    rl_kind = ""
    risk_cfg = {}
    if args.policy == "risk_aware":
        rl_path = str(args.rl_model or primary_path).strip()
        rl_model, rl_is_multi, rl_kind = _load_single_model(rl_path, "pv", args, torch, nn, device)
        rl_model.eval()
        risk_cfg = load_risk_config(args.risk_config)
        logger.info(
            "Loaded risk-aware models: primary=%s(%s) rl=%s(%s)",
            primary_path,
            policy_kind,
            rl_path,
            rl_kind,
        )
    else:
        logger.info("Loaded model=%s policy=%s multi_head=%s", primary_path, policy_kind, is_multi)

    backend = create_backend(
        args.backend,
        base_url=args.base_url if args.backend == "real" else None,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
        logger=logger,
    )

    try:
        if args.backend == "sim":
            backend.reset(seed=args.seed)

        seen_hand = False
        seen_shop = False
        max_loops = 300 if args.once else 10**9

        for _ in range(max_loops):
            try:
                state = backend.get_state()
            except Exception as exc:
                if args.strict_errors:
                    logger.error("Failed to fetch state (strict): %s", exc)
                    return 1
                logger.warning("Failed to fetch state: %s", exc)
                if args.once:
                    return 1
                time.sleep(args.poll_interval)
                continue

            phase = str(state.get("state") or "UNKNOWN")

            supports_shop = bool(is_multi or (args.policy == "risk_aware" and rl_is_multi))

            if phase == "SELECTING_HAND" and not seen_hand:
                cards = (state.get("hand") or {}).get("cards") or []
                hand_size = min(len(cards), action_space.MAX_HAND)
                if hand_size <= 0:
                    logger.info("SELECTING_HAND but no cards; waiting")
                else:
                    legal_ids = action_space.legal_action_ids(hand_size)
                    batch = _state_to_hand_batch(state, torch)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    decoded = []
                    selected_policy = args.policy
                    selected_action = None

                    if args.policy == "risk_aware":
                        with torch.no_grad():
                            pv_logits = _forward_hand(model, batch, is_multi, policy_kind)
                            rl_logits = _forward_hand(rl_model, batch, rl_is_multi, rl_kind)

                        pv_aid, pv_conf, _ = _predict_top1_with_conf(pv_logits, legal_ids, args.max_actions, torch, device)
                        rl_aid, rl_conf, _ = _predict_top1_with_conf(rl_logits, legal_ids, args.max_actions, torch, device)
                        pv_top3 = _predict_topk(pv_logits, legal_ids, args.max_actions, 3, torch, device)

                        pv_action = {"action_type": "WAIT", "indices": []}
                        rl_action = {"action_type": "WAIT", "indices": []}
                        if pv_aid is not None:
                            atype, mask_int = action_space.decode(hand_size, int(pv_aid))
                            pv_action = {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}
                        if rl_aid is not None:
                            atype, mask_int = action_space.decode(hand_size, int(rl_aid))
                            rl_action = {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}

                        heur = choose_action(state, start_seed=args.seed)
                        heur_action = {
                            "action_type": str(heur.action_type or "WAIT").upper(),
                            "indices": action_space.mask_to_indices(int(heur.mask_int or 0), hand_size) if heur.mask_int is not None else [],
                        }
                        heur_id = None
                        try:
                            heur_id = action_space.encode(hand_size, heur_action["action_type"], action_space.indices_to_mask(list(heur_action["indices"])))
                        except Exception:
                            heur_id = None

                        search_decision = choose_search_action(
                            state,
                            max_branch=80,
                            max_depth=2,
                            time_budget_ms=float(args.search_budget_ms),
                            seed=args.seed,
                        )
                        search_action = {"action_type": search_decision.action_type, "indices": list(search_decision.indices or [])}
                        search_id = None
                        try:
                            search_id = action_space.encode(hand_size, search_action["action_type"], action_space.indices_to_mask(list(search_action["indices"])))
                        except Exception:
                            search_id = None

                        hybrid_action = dict(pv_action)
                        hybrid_id = pv_aid
                        if search_id is not None and any(int(search_id) == int(x[0]) for x in pv_top3):
                            hybrid_action = dict(search_action)
                            hybrid_id = int(search_id)

                        pv_value = None
                        rl_value = None
                        if policy_kind == "pv":
                            with torch.no_grad():
                                _, pv_val_t = model.forward_hand(batch)
                            pv_value = float(pv_val_t.squeeze().item())
                        if rl_kind == "pv":
                            with torch.no_grad():
                                _, rl_val_t = rl_model.forward_hand(batch)
                            rl_value = float(rl_val_t.squeeze().item())

                        signals = [
                            PolicySignal("pv", None if pv_aid is None else int(pv_aid), float(pv_conf), pv_value, pv_aid is not None),
                            PolicySignal("rl", None if rl_aid is None else int(rl_aid), float(rl_conf), rl_value, rl_aid is not None),
                            PolicySignal("hybrid", None if hybrid_id is None else int(hybrid_id), float(max(pv_conf, rl_conf)), pv_value, hybrid_id is not None),
                            PolicySignal("heuristic", None if heur_id is None else int(heur_id), 0.50, None, heur_id is not None),
                        ]
                        context_vals = [float(x) for x in batch["context"][0].detach().cpu().tolist()]
                        decision = select_risk_policy(
                            phase_group="hand",
                            signals=signals,
                            context=context_vals,
                            cfg=risk_cfg,
                            available_policies={"pv", "rl", "hybrid", "heuristic", "search"},
                        )
                        selected_policy = str(decision.get("selected_policy") or "heuristic")
                        risk_score = float(decision.get("risk_score") or 0.0)
                        fallback_reason = str(decision.get("fallback_reason") or "")
                        policy_actions = {
                            "pv": pv_action,
                            "rl": rl_action,
                            "hybrid": hybrid_action,
                            "heuristic": heur_action,
                            "search": search_action,
                        }
                        selected_action = dict(policy_actions.get(selected_policy) or heur_action)
                        logger.info(
                            "risk-aware(hand): selected=%s risk=%.3f reason=%s",
                            selected_policy,
                            risk_score,
                            fallback_reason,
                        )
                        logger.info(
                            "risk-aware(hand) top1 ids: pv=%s rl=%s hybrid=%s heuristic=%s search=%s",
                            pv_aid,
                            rl_aid,
                            hybrid_id,
                            heur_id,
                            search_id,
                        )
                        ranked = []
                        for pid in ("rl", "hybrid", "pv", "heuristic", "search"):
                            act = policy_actions[pid]
                            ranked.append((pid, act))
                        logger.info("Hand suggestions by policy:")
                        for pid, act in ranked[: max(1, int(args.topk))]:
                            logger.info("  - %s => %s", pid, format_action(str(act.get("action_type") or "WAIT"), list(act.get("indices") or [])))
                    else:
                        with torch.no_grad():
                            logits = _forward_hand(model, batch, is_multi, policy_kind)
                        ranked = _predict_topk(logits, legal_ids, args.max_actions, args.topk, torch, device)

                        logger.info("Hand Top-%d suggestions:", len(ranked))
                        for rank_idx, (aid, score) in enumerate(ranked, start=1):
                            atype, mask_int = action_space.decode(hand_size, aid)
                            idxs = action_space.mask_to_indices(mask_int, hand_size)
                            decoded.append((aid, atype, idxs, score))
                            logger.info(
                                "  #%d action_id=%d score=%.4f action_type=%s indices=%s",
                                rank_idx,
                                aid,
                                score,
                                atype,
                                idxs,
                            )
                        if decoded:
                            _, atype, idxs, _ = decoded[0]
                            selected_action = {"action_type": atype, "indices": idxs}

                    if selected_action and (args.execute or (args.once and args.backend == "sim")):
                        _, _, _, _ = backend.step(selected_action)
                        logger.info(
                            "Executed hand top-1: policy=%s action=%s",
                            selected_policy,
                            format_action(str(selected_action.get("action_type") or "WAIT"), list(selected_action.get("indices") or [])),
                        )
                    seen_hand = True

            elif phase in action_space_shop.SHOP_PHASES and not seen_shop and supports_shop:
                legal_ids = action_space_shop.legal_action_ids(state)
                batch = _state_to_shop_batch(state, torch)
                batch = {k: v.to(device) for k, v in batch.items()}
                selected_policy = args.policy
                selected_action = None

                if args.policy == "risk_aware":
                    pv_aid, pv_conf = None, 0.0
                    rl_aid, rl_conf = None, 0.0
                    if is_multi:
                        with torch.no_grad():
                            pv_logits = _forward_shop(model, batch, policy_kind)
                        pv_aid, pv_conf, _ = _predict_top1_with_conf(pv_logits, legal_ids, args.max_shop_actions, torch, device)
                    if rl_is_multi:
                        with torch.no_grad():
                            rl_logits = _forward_shop(rl_model, batch, rl_kind)
                        rl_aid, rl_conf, _ = _predict_top1_with_conf(rl_logits, legal_ids, args.max_shop_actions, torch, device)

                    pv_action = action_space_shop.action_from_id(state, int(pv_aid)) if pv_aid is not None else {"action_type": "WAIT"}
                    rl_action = action_space_shop.action_from_id(state, int(rl_aid)) if rl_aid is not None else {"action_type": "WAIT"}
                    heur_action = dict((choose_shop_action(state).action) or {"action_type": "WAIT"})
                    heur_id = None
                    try:
                        heur_id = action_space_shop.encode(str(heur_action.get("action_type") or "WAIT"), dict(heur_action.get("params") or {}))
                    except Exception:
                        heur_id = None

                    signals = [
                        PolicySignal("pv", None if pv_aid is None else int(pv_aid), float(pv_conf), None, pv_aid is not None),
                        PolicySignal("rl", None if rl_aid is None else int(rl_aid), float(rl_conf), None, rl_aid is not None),
                        PolicySignal("hybrid", None if pv_aid is None else int(pv_aid), float(max(pv_conf, rl_conf)), None, pv_aid is not None),
                        PolicySignal("heuristic", None if heur_id is None else int(heur_id), 0.50, None, heur_id is not None),
                    ]
                    ctx_vals = [float(x) for x in batch["shop_context"][0].detach().cpu().tolist()]
                    decision = select_risk_policy(
                        phase_group="shop",
                        signals=signals,
                        context=ctx_vals,
                        cfg=risk_cfg,
                        available_policies={"pv", "rl", "hybrid", "heuristic"},
                    )
                    selected_policy = str(decision.get("selected_policy") or "heuristic")
                    selected_action = {
                        "pv": pv_action,
                        "rl": rl_action,
                        "hybrid": pv_action,
                        "heuristic": heur_action,
                    }.get(selected_policy, heur_action)
                    logger.info(
                        "risk-aware(shop): selected=%s risk=%.3f reason=%s",
                        selected_policy,
                        float(decision.get("risk_score") or 0.0),
                        str(decision.get("fallback_reason") or ""),
                    )
                else:
                    with torch.no_grad():
                        logits = _forward_shop(model, batch, policy_kind)
                    ranked = _predict_topk(logits, legal_ids, args.max_shop_actions, args.topk, torch, device)

                    logger.info("Shop Top-%d suggestions:", len(ranked))
                    decoded = []
                    for rank_idx, (aid, score) in enumerate(ranked, start=1):
                        action = action_space_shop.action_from_id(state, aid)
                        decoded.append((aid, action, score))
                        logger.info("  #%d action_id=%d score=%.4f action=%s", rank_idx, aid, score, action)
                    if decoded:
                        _, action, _ = decoded[0]
                        selected_action = action

                if selected_action and (args.execute or (args.once and args.backend == "sim")):
                    _, _, _, _ = backend.step(selected_action)
                    logger.info("Executed shop top-1: policy=%s action=%s", selected_policy, selected_action)
                seen_shop = True
            elif args.once and supports_shop and seen_hand and not seen_shop:
                # Once-mode fallback: emit one shop-head decision even if not currently in SHOP phase.
                action = dict((choose_shop_action(state).action) or {"action_type": "WAIT"})
                logger.info("Shop fallback suggestion (phase=%s): %s", phase, action)
                if args.execute or (args.once and args.backend == "sim"):
                    _, _, _, _ = backend.step(action)
                    logger.info("Executed shop top-1 fallback: %s", action)
                seen_shop = True

            if args.once:
                if seen_hand and (seen_shop or not is_multi):
                    return 0
                if args.backend == "sim":
                    try:
                        backend.step({"action_type": "AUTO"})
                    except Exception as exc:
                        if args.strict_errors:
                            logger.error("AUTO step failed (strict): %s", exc)
                            return 1
                        logger.warning("AUTO step failed: %s", exc)
                        state_now = backend.get_state()
                        phase_now = str(state_now.get("state") or "UNKNOWN")
                        if phase_now in {"SELECTING_HAND", "GAME_OVER", "MENU"}:
                            try:
                                backend.step({"action_type": "MENU"})
                                backend.step({"action_type": "START", "seed": args.seed})
                                continue
                            except Exception:
                                return 1
                        return 1
                else:
                    time.sleep(args.poll_interval)
            else:
                time.sleep(args.poll_interval)

        logger.warning("infer loop exhausted before reaching target phases")
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
