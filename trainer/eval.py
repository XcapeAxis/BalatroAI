if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from trainer import action_space
from trainer.dataset import iter_train_samples
from trainer.env_client import create_backend
from trainer.expert_policy import choose_action
from trainer.features import extract_features
from trainer.utils import setup_logger, warn_if_unstable_python


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for eval.py") from exc
    return torch, nn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC model offline or online.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="Run offline evaluation against labeled dataset.")
    mode.add_argument("--online", action="store_true", help="Run online evaluation in environment.")

    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--max-actions", type=int, default=action_space.max_actions())

    parser.add_argument("--dataset", default=None, help="Dataset jsonl for offline eval.")

    parser.add_argument("--backend", choices=["real", "sim"], default="real", help="Backend used in --online mode.")
    parser.add_argument("--base-url", default="http://127.0.0.1:12346", help="Base URL for online eval (real backend).")
    parser.add_argument("--episodes", type=int, default=5, help="Online episodes.")
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed-prefix", default="AAAAAAA")
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


def _record_to_batch(record, torch):
    f = record["features"]
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


def run_offline(args, model, device, torch):
    logger = setup_logger("trainer.eval.offline")
    if not args.dataset:
        logger.error("--dataset is required in --offline mode")
        return 2

    total = 0
    correct1 = 0
    correct3 = 0
    illegal_top1_masked_count = 0
    random_top3_baseline_sum = 0.0
    invalid_sample_count = 0

    model.eval()
    for record in iter_train_samples(args.dataset):
        legal_ids = list(record["legal_action_ids"])
        valid_legal_ids = [aid for aid in legal_ids if 0 <= aid < args.max_actions]
        if not valid_legal_ids:
            invalid_sample_count += 1
            continue

        target = int(record["expert_action_id"])

        batch = _record_to_batch(record, torch)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(batch)

        legal_mask = torch.zeros((1, args.max_actions), dtype=torch.float32, device=device)
        for aid in valid_legal_ids:
            legal_mask[0, aid] = 1.0

        masked = torch.where(legal_mask > 0, logits, torch.full_like(logits, -1e9))
        pred = int(masked.argmax(dim=1).item())
        top3_k = min(3, args.max_actions)
        top3 = torch.topk(masked, k=top3_k, dim=1).indices[0].tolist()

        total += 1
        if pred == target:
            correct1 += 1
        if target in top3:
            correct3 += 1
        if pred not in valid_legal_ids:
            illegal_top1_masked_count += 1

        random_k = min(3, len(valid_legal_ids))
        random_top3_baseline_sum += random_k / len(valid_legal_ids)

    if total == 0:
        logger.error(
            "No valid trainable records found in dataset: %s (invalid=%d)",
            args.dataset,
            invalid_sample_count,
        )
        return 2

    top1 = correct1 / total
    top3 = correct3 / total
    illegal_rate = illegal_top1_masked_count / total
    random_top3_baseline = random_top3_baseline_sum / total
    top3_vs_random_lift = top3 - random_top3_baseline

    logger.info(
        "Offline eval: total=%d top1=%.4f top3=%.4f illegal_rate=%.6f random_top3=%.4f top3_lift=%.4f invalid=%d",
        total,
        top1,
        top3,
        illegal_rate,
        random_top3_baseline,
        top3_vs_random_lift,
        invalid_sample_count,
    )
    return 0


def run_online(args, model, device, torch):
    logger = setup_logger("trainer.eval.online")
    wins = 0
    episodes_finished = 0
    money_sum = 0.0
    steps_sum = 0

    backend = create_backend(
        args.backend,
        base_url=args.base_url if args.backend == "real" else None,
        timeout_sec=args.timeout_sec,
        seed=args.seed_prefix,
        logger=logger,
    )

    try:
        for episode in range(args.episodes):
            try:
                state = backend.reset(seed=f"{args.seed_prefix}-{episode}")
            except Exception as exc:
                logger.warning("episode=%d failed to initialize: %s", episode, exc)
                continue

            ep_steps = 0
            ep_done = False
            for _ in range(args.max_steps_per_episode):
                phase = str(state.get("state") or "UNKNOWN")
                if phase == "GAME_OVER":
                    episodes_finished += 1
                    ep_done = True
                    if bool(state.get("won")):
                        wins += 1
                    money_sum += float(state.get("money") or 0.0)
                    break

                if phase == "SELECTING_HAND":
                    hand_size = min(len((state.get("hand") or {}).get("cards") or []), action_space.MAX_HAND)
                    if hand_size <= 0:
                        state, _, _, _ = backend.step({"action_type": "WAIT", "sleep": 0.05})
                        ep_steps += 1
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
                    aid = int(masked.argmax(dim=1).item())

                    atype, mask = action_space.decode(hand_size, aid)
                    indices = action_space.mask_to_indices(mask, hand_size)
                    state, _, _, _ = backend.step({"action_type": atype, "indices": indices})
                else:
                    decision = choose_action(state, start_seed=f"{args.seed_prefix}-{episode}")
                    macro = str(decision.macro_action or "wait").upper()
                    params = dict(decision.macro_params or {})
                    action = {"action_type": macro}
                    action.update(params)
                    if macro == "WAIT":
                        action["sleep"] = 0.05
                    state, _, _, _ = backend.step(action)

                ep_steps += 1

            steps_sum += ep_steps
            if not ep_done:
                episodes_finished += 1
                money_sum += float(state.get("money") or 0.0)
    finally:
        backend.close()

    if episodes_finished == 0:
        logger.error("Online eval completed with zero finished episodes")
        return 1

    logger.info(
        "Online eval: episodes_finished=%d win_rate=%.4f avg_money=%.2f avg_steps=%.2f",
        episodes_finished,
        wins / episodes_finished,
        money_sum / episodes_finished,
        steps_sum / max(1, episodes_finished),
    )
    return 0

def main() -> int:
    args = parse_args()
    logger = setup_logger("trainer.eval")
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

    if args.offline:
        return run_offline(args, model, device, torch)
    return run_online(args, model, device, torch)


if __name__ == "__main__":
    raise SystemExit(main())



