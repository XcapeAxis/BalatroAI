if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time

from trainer import action_space
from trainer import action_space_shop
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.features_shop import SHOP_CONTEXT_DIM, extract_shop_features
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=["bc", "pv"], default="bc")
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())
    parser.add_argument("--max-shop-actions", type=int, default=action_space_shop.max_actions())
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--execute", action="store_true", help="Execute top-1 recommendation.")
    parser.add_argument("--once", action="store_true", help="Run one loop and exit.")
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


def _load_model(args, torch, nn, device):
    state_dict = torch.load(args.model, map_location=device)

    if args.policy == "pv":
        from trainer.models.policy_value import PolicyValueModel

        model = PolicyValueModel(args.max_actions, args.max_shop_actions).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model, True, "pv"

    is_multi = any(k.startswith("hand_head.") for k in state_dict.keys())
    if is_multi:
        model = _build_multi_model(nn, args.max_actions, args.max_shop_actions).to(device)
        model.load_state_dict(state_dict, strict=True)
        return model, True, "bc"

    model = _build_hand_model(nn, args.max_actions).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, False, "bc"


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

    model, is_multi, policy_kind = _load_model(args, torch, nn, device)
    model.eval()
    logger.info("Loaded model=%s policy=%s multi_head=%s", args.model, policy_kind, is_multi)

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
                logger.warning("Failed to fetch state: %s", exc)
                if args.once:
                    return 1
                time.sleep(args.poll_interval)
                continue

            phase = str(state.get("state") or "UNKNOWN")

            if phase == "SELECTING_HAND" and not seen_hand:
                cards = (state.get("hand") or {}).get("cards") or []
                hand_size = min(len(cards), action_space.MAX_HAND)
                if hand_size <= 0:
                    logger.info("SELECTING_HAND but no cards; waiting")
                else:
                    legal_ids = action_space.legal_action_ids(hand_size)
                    batch = _state_to_hand_batch(state, torch)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        logits = _forward_hand(model, batch, is_multi, policy_kind)
                    ranked = _predict_topk(logits, legal_ids, args.max_actions, args.topk, torch, device)

                    logger.info("Hand Top-%d suggestions:", len(ranked))
                    decoded = []
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

                    if decoded and (args.execute or (args.once and args.backend == "sim")):
                        _, atype, idxs, _ = decoded[0]
                        _, _, _, _ = backend.step({"action_type": atype, "indices": idxs})
                        logger.info("Executed hand top-1: %s", format_action(atype, idxs))
                    seen_hand = True

            elif phase in action_space_shop.SHOP_PHASES and not seen_shop and is_multi:
                legal_ids = action_space_shop.legal_action_ids(state)
                batch = _state_to_shop_batch(state, torch)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = _forward_shop(model, batch, policy_kind)
                ranked = _predict_topk(logits, legal_ids, args.max_shop_actions, args.topk, torch, device)

                logger.info("Shop Top-%d suggestions:", len(ranked))
                decoded = []
                for rank_idx, (aid, score) in enumerate(ranked, start=1):
                    action = action_space_shop.action_from_id(state, aid)
                    decoded.append((aid, action, score))
                    logger.info("  #%d action_id=%d score=%.4f action=%s", rank_idx, aid, score, action)

                if decoded and (args.execute or (args.once and args.backend == "sim")):
                    _, action, _ = decoded[0]
                    _, _, _, _ = backend.step(action)
                    logger.info("Executed shop top-1: %s", action)
                seen_shop = True
            elif args.once and is_multi and seen_hand and not seen_shop:
                # Once-mode fallback: emit one shop-head decision even if not currently in SHOP phase.
                legal_ids = action_space_shop.legal_action_ids(state)
                batch = _state_to_shop_batch(state, torch)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    logits = _forward_shop(model, batch, policy_kind)
                ranked = _predict_topk(logits, legal_ids, args.max_shop_actions, args.topk, torch, device)

                logger.info("Shop Top-%d suggestions (phase=%s fallback):", len(ranked), phase)
                decoded = []
                for rank_idx, (aid, score) in enumerate(ranked, start=1):
                    action = action_space_shop.action_from_id(state, aid)
                    decoded.append((aid, action, score))
                    logger.info("  #%d action_id=%d score=%.4f action=%s", rank_idx, aid, score, action)

                if decoded and (args.execute or (args.once and args.backend == "sim")):
                    _, action, _ = decoded[0]
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
