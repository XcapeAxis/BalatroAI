if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import time

from trainer import action_space
from trainer.env_client import create_backend
from trainer.features import extract_features
from trainer.utils import format_action, setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for infer_assistant.py") from exc
    return torch, nn


def parse_args():
    parser = argparse.ArgumentParser(description="Inference assistant for Balatro hand decisions.")
    parser.add_argument("--backend", choices=["real", "sim"], default="real")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--seed", default="AAAAAAA")
    parser.add_argument("--execute", action="store_true", help="Execute top-1 recommendation.")
    parser.add_argument("--once", action="store_true", help="Run one loop and exit.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _build_model(nn, max_actions: int):
    class BCModel(nn.Module):
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
            logits = self.head(fused)
            return logits

    return BCModel(max_actions)


def _state_to_batch(state, torch):
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

    model = _build_model(nn, args.max_actions).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

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

        while True:
            try:
                state = backend.get_state()
            except Exception as exc:
                logger.warning("Failed to fetch state: %s", exc)
                if args.once:
                    return 1
                time.sleep(args.poll_interval)
                continue

            phase = str(state.get("state") or "UNKNOWN")
            if phase != "SELECTING_HAND":
                logger.info("phase=%s (waiting for SELECTING_HAND)", phase)
                if args.once and args.backend == "sim":
                    progressed = False
                    for _ in range(30):
                        state, _, _, _ = backend.step({"action_type": "AUTO"})
                        phase = str(state.get("state") or "UNKNOWN")
                        if phase == "SELECTING_HAND":
                            progressed = True
                            break
                    if not progressed:
                        logger.warning("sim --once could not reach SELECTING_HAND within 30 AUTO steps")
                        return 1
                elif args.once:
                    return 0
                else:
                    time.sleep(args.poll_interval)
                    continue

            cards = (state.get("hand") or {}).get("cards") or []
            hand_size = min(len(cards), action_space.MAX_HAND)
            if hand_size <= 0:
                logger.info("No hand cards available.")
                if args.once:
                    return 0
                time.sleep(args.poll_interval)
                continue

            legal_ids = action_space.legal_action_ids(hand_size)
            batch = _state_to_batch(state, torch)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                logits = model(batch)

            legal_mask = torch.zeros((1, args.max_actions), dtype=torch.float32, device=device)
            for aid in legal_ids:
                legal_mask[0, aid] = 1.0
            masked = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))

            k = min(max(1, args.topk), masked.shape[1])
            top_vals, top_ids = torch.topk(masked, k=k, dim=1)

            decoded = []
            for i in range(k):
                aid = int(top_ids[0, i].item())
                score = float(top_vals[0, i].item())
                atype, mask_int = action_space.decode(hand_size, aid)
                idxs = action_space.mask_to_indices(mask_int, hand_size)
                decoded.append((aid, atype, idxs, score))

            logger.info("Top-%d suggestions:", k)
            for rank, (aid, atype, idxs, score) in enumerate(decoded, start=1):
                logger.info("  #%d action_id=%d score=%.4f action_type=%s indices=%s", rank, aid, score, atype, idxs)

            if args.execute and decoded:
                _, atype, idxs, _ = decoded[0]
                try:
                    _, _, _, _ = backend.step({"action_type": atype, "indices": idxs})
                    logger.info("Executed top-1: %s", format_action(atype, idxs))
                except Exception as exc:
                    logger.error("Execution failed: %s", exc)
                    if args.once:
                        return 1

            if args.once:
                return 0

            time.sleep(args.poll_interval)
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
