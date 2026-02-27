"""Generate distillation data from an ensemble teacher (pv+hybrid+rl+risk_aware) via sim rollouts."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer import action_space, action_space_shop
from trainer.ensemble_teacher import borda_rank_aggregate, decision_to_dict
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action
from trainer.expert_policy_shop import choose_shop_action
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
from trainer.utils import setup_logger, warn_if_unstable_python


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _load_model(model_path: str | None, torch, nn, max_actions: int, max_shop_actions: int, device):
    if not model_path or not Path(model_path).exists():
        return None
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        return None
    keys = list(state.keys())

    # PolicyValueModel: hand_policy_head, shop_policy_head
    if any(k.startswith("hand_policy_head") for k in keys):
        from trainer.models.policy_value import PolicyValueModel
        model = PolicyValueModel(max_actions, max_shop_actions)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model

    # BC multi-head: hand_head, shop_head
    if any(k.startswith("hand_head") or k.startswith("shop_head") for k in keys):
        from trainer.eval_long_horizon import _build_multi_model
        model = _build_multi_model(nn, max_actions, max_shop_actions)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model

    # BC hand-only
    from trainer.eval_long_horizon import _build_hand_model
    model = _build_hand_model(nn, max_actions)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _model_topk(model, state, phase: str, torch, device, k: int, max_actions: int, max_shop_actions: int):
    """Get top-k actions from a model for given state and phase."""
    if model is None:
        return []
    try:
        if phase == "HAND":
            f = extract_features(state)
            chip_hint = list(f.get("card_chip_hint") or [0] * action_space.MAX_HAND)
            batch = {
                "rank": torch.tensor([f["card_rank_ids"]], dtype=torch.long, device=device),
                "suit": torch.tensor([f["card_suit_ids"]], dtype=torch.long, device=device),
                "chip": torch.tensor([chip_hint], dtype=torch.float32, device=device),
                "enh": torch.tensor([f["card_has_enhancement"]], dtype=torch.float32, device=device),
                "edt": torch.tensor([f["card_has_edition"]], dtype=torch.float32, device=device),
                "seal": torch.tensor([f["card_has_seal"]], dtype=torch.float32, device=device),
                "pad": torch.tensor([f["hand_pad_mask"]], dtype=torch.float32, device=device),
                "context": torch.tensor([f["context"]], dtype=torch.float32, device=device),
            }
            with torch.no_grad():
                if hasattr(model, "forward_hand"):
                    out = model.forward_hand(batch)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    logits = model(batch)
            legal = state.get("legal_action_ids") or list(range(max_actions))
            valid = [a for a in legal if 0 <= a < max_actions]
            mask = torch.zeros((1, max_actions), dtype=torch.float32, device=device)
            for a in valid:
                mask[0, a] = 1.0
            masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
            topk_ids = torch.topk(masked, k=min(k, masked.shape[1]), dim=1).indices[0].tolist()
            return [int(a) for a in topk_ids if int(a) in set(valid)]
        else:
            sf = extract_shop_features(state)
            ctx = list(sf.get("shop_context") or [0.0] * SHOP_CONTEXT_DIM)
            if len(ctx) != SHOP_CONTEXT_DIM:
                ctx = (ctx + [0.0] * SHOP_CONTEXT_DIM)[:SHOP_CONTEXT_DIM]
            batch = {"shop_context": torch.tensor([ctx], dtype=torch.float32, device=device)}
            with torch.no_grad():
                if hasattr(model, "forward_shop"):
                    out = model.forward_shop(batch)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    return []
            legal = state.get("shop_legal_action_ids") or list(range(max_shop_actions))
            valid = [a for a in legal if 0 <= a < max_shop_actions]
            mask = torch.zeros((1, max_shop_actions), dtype=torch.float32, device=device)
            for a in valid:
                mask[0, a] = 1.0
            masked = torch.where(mask > 0, logits, torch.full_like(logits, -1e9))
            topk_ids = torch.topk(masked, k=min(k, masked.shape[1]), dim=1).indices[0].tolist()
            return [int(a) for a in topk_ids if int(a) in set(valid)]
    except Exception:
        return []


def _expert_to_action_dict(state: dict[str, Any], decision: Any) -> dict[str, Any]:
    if decision is None:
        return {"action_type": "WAIT"}

    action_type = str(getattr(decision, "action_type", "") or "").upper()
    mask_int = getattr(decision, "mask_int", None)
    if action_type and mask_int is not None:
        hand = (state.get("hand") or {}).get("cards") or []
        hand_size = min(len(hand), action_space.MAX_HAND)
        if hand_size > 0:
            return {
                "action_type": action_type,
                "indices": action_space.mask_to_indices(int(mask_int), hand_size),
            }

    macro_action = str(getattr(decision, "macro_action", "") or "").upper()
    if macro_action:
        out = {"action_type": macro_action}
        out.update(dict(getattr(decision, "macro_params", None) or {}))
        return out

    return {"action_type": "WAIT"}


def _shop_decision_to_action_dict(decision: Any) -> dict[str, Any]:
    action = getattr(decision, "action", None)
    if isinstance(action, dict) and action:
        return dict(action)
    return {"action_type": "WAIT"}


def _heuristic_topk(state, phase: str, k: int) -> list[int]:
    try:
        if phase == "HAND":
            decision = choose_action(state)
            hand = (state.get("hand") or {}).get("cards") or []
            hand_size = min(len(hand), action_space.MAX_HAND)
            if hand_size > 0 and getattr(decision, "action_type", None) and getattr(decision, "mask_int", None) is not None:
                aid = action_space.encode(hand_size, str(decision.action_type).upper(), int(decision.mask_int))
                return [int(aid)]
            legal = state.get("legal_action_ids") or action_space.legal_action_ids(hand_size)
            return [int(a) for a in legal[: max(1, k)] if isinstance(a, int) or str(a).isdigit()]
        else:
            decision = choose_shop_action(state)
            if getattr(decision, "action_id", None) is not None:
                return [int(decision.action_id)]
            legal = state.get("shop_legal_action_ids") or action_space_shop.legal_action_ids(state)
            return [int(a) for a in legal[: max(1, k)] if isinstance(a, int) or str(a).isdigit()]
    except Exception:
        return []


def main() -> int:
    p = argparse.ArgumentParser(description="Generate distillation data from ensemble teacher.")
    p.add_argument("--backend", choices=["sim"], default="sim")
    p.add_argument("--stake", default="gold")
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--max-steps-per-episode", type=int, default=280)
    p.add_argument("--hand-target-samples", type=int, default=10000)
    p.add_argument("--shop-target-samples", type=int, default=3000)
    p.add_argument("--pv-model", default="")
    p.add_argument("--hybrid-model", default="")
    p.add_argument("--rl-model", default="")
    p.add_argument("--risk-aware-config", default="")
    p.add_argument("--out", required=True, help="Output jsonl path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    logger = setup_logger("trainer.rollout_distill_p20")
    warn_if_unstable_python(logger)

    torch, nn = _require_torch()
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    max_a = action_space.max_actions()
    max_sa = action_space_shop.max_actions()

    models: dict[str, Any] = {}
    available: list[str] = []
    for name, path in [("pv", args.pv_model), ("hybrid", args.hybrid_model), ("rl", args.rl_model)]:
        m = _load_model(path, torch, nn, max_a, max_sa, device)
        models[name] = m
        if m is not None:
            available.append(name)

    models["risk_aware"] = models.get("pv")  # risk_aware reuses pv model with fallback
    if models["risk_aware"] is not None:
        available.append("risk_aware")

    logger.info(f"available ensemble members: {available}")
    if not available:
        logger.warning("no models available, using heuristic-only ensemble")
        available = ["heuristic"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = create_backend(args.backend, base_url="http://127.0.0.1:12346")

    hand_count = 0
    shop_count = 0
    total_steps = 0

    import random
    rng = random.Random(args.seed)

    with out_path.open("w", encoding="utf-8") as fp:
        for ep in range(args.episodes):
            if hand_count >= args.hand_target_samples and shop_count >= args.shop_target_samples:
                break

            seed_val = rng.randrange(1, 2**31 - 1)
            try:
                state = backend.reset(seed=str(seed_val))
                if str(args.stake or "").strip().lower() not in ("", "white"):
                    state, _, _, _ = backend.step({
                        "action_type": "START",
                        "seed": str(seed_val),
                        "stake": str(args.stake).upper(),
                    })
            except Exception as e:
                logger.warning(f"ep={ep} reset/start failed: {e}")
                continue

            for step in range(args.max_steps_per_episode):
                if state.get("done"):
                    break
                phase_raw = str(state.get("phase") or state.get("state") or "")
                if phase_raw == "SELECTING_HAND":
                    phase = "HAND"
                elif phase_raw in ("SHOP", "SMODS_BOOSTER_OPENED"):
                    phase = "SHOP"
                else:
                    try:
                        d = choose_action(state)
                        act_dict = _expert_to_action_dict(state, d)
                        state, _, _, _ = backend.step(act_dict)
                    except Exception:
                        break
                    continue

                if phase == "HAND" and hand_count >= args.hand_target_samples:
                    try:
                        d = choose_action(state)
                        act_dict = _expert_to_action_dict(state, d)
                        state, _, _, _ = backend.step(act_dict)
                    except Exception:
                        break
                    continue
                if phase == "SHOP" and shop_count >= args.shop_target_samples:
                    try:
                        state, _, _, _ = backend.step(_shop_decision_to_action_dict(choose_shop_action(state)))
                    except Exception:
                        break
                    continue

                member_rankings: dict[str, list[int]] = {}
                member_details: dict[str, dict[str, Any]] = {}

                for strat in available:
                    if strat == "heuristic":
                        topk = _heuristic_topk(state, phase, args.topk)
                    elif strat == "risk_aware":
                        topk = _model_topk(models.get("pv"), state, phase, torch, device, args.topk, max_a, max_sa)
                    else:
                        topk = _model_topk(models.get(strat), state, phase, torch, device, args.topk, max_a, max_sa)

                    if topk:
                        member_rankings[strat] = topk
                        member_details[strat] = {"top1": topk[0], "topk": topk[:args.topk]}

                legal = state.get("legal_action_ids") if phase == "HAND" else state.get("shop_legal_action_ids", [])
                if not legal:
                    legal = list(range(max_a if phase == "HAND" else max_sa))

                if member_rankings:
                    decision = borda_rank_aggregate(member_rankings, legal, k=args.topk)
                    teacher_topk = [decision.chosen_action] + [
                        a for a in legal if a != decision.chosen_action
                    ][:args.topk - 1]
                else:
                    heur = _heuristic_topk(state, phase, args.topk)
                    teacher_topk = heur if heur else legal[:args.topk]
                    decision = None

                features_raw: dict[str, Any] = {}
                if phase == "HAND":
                    features_raw = extract_features(state)
                else:
                    features_raw = extract_shop_features(state)

                record: dict[str, Any] = {
                    "schema": "distill_v1",
                    "phase": phase,
                    "state_features": features_raw,
                    "teacher_topk": teacher_topk[:args.topk],
                    "teacher_members": member_details,
                    "teacher_confidence": round(decision.confidence, 4) if decision else 0.0,
                    "disagreement": round(decision.disagreement, 4) if decision else 0.0,
                    "metadata": {
                        "stake": args.stake,
                        "ante": state.get("ante"),
                        "seed": seed_val,
                        "episode": ep,
                        "step": step,
                    },
                }

                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                if phase == "HAND":
                    hand_count += 1
                else:
                    shop_count += 1
                total_steps += 1

                try:
                    if phase == "HAND":
                        act_id = teacher_topk[0] if teacher_topk else None
                        if act_id is not None:
                            hand = (state.get("hand") or {}).get("cards") or []
                            hand_size = min(len(hand), action_space.MAX_HAND)
                            atype, mask_int = action_space.decode(hand_size, int(act_id))
                            act_dict = {"action_type": atype, "indices": action_space.mask_to_indices(mask_int, hand_size)}
                        else:
                            d = choose_action(state)
                            act_dict = _expert_to_action_dict(state, d)
                        state, _, _, _ = backend.step(act_dict)
                    else:
                        act_id = teacher_topk[0] if teacher_topk else None
                        if act_id is not None:
                            act_dict = action_space_shop.action_from_id(state, int(act_id))
                        else:
                            act_dict = _shop_decision_to_action_dict(choose_shop_action(state))
                        state, _, _, _ = backend.step(act_dict)
                except Exception:
                    break

    summary = {
        "schema": "distill_rollout_summary_v1",
        "generated_at": _now_iso(),
        "episodes_attempted": args.episodes,
        "hand_samples": hand_count,
        "shop_samples": shop_count,
        "total_steps": total_steps,
        "ensemble_members": available,
        "out_path": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
